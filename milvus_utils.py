import streamlit as st
# Use MilvusClient for Lite version
from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema
import time
import os

# Import config variables including the global map
from config import (
    MILVUS_LITE_DATA_PATH, COLLECTION_NAME, EMBEDDING_DIM,
    MAX_ARTICLES_TO_INDEX, INDEX_METRIC_TYPE, INDEX_TYPE, INDEX_PARAMS,
    MILVUS_SEARCH_PARAMS, TOP_K, id_to_doc_map, save_id_map
)

@st.cache_resource
def get_milvus_client(max_retries=3, retry_delay=2):
    """Initializes and returns a MilvusClient instance for Milvus Lite with retry mechanism."""
    tries = 0
    while tries < max_retries:
        try:
            st.write(f"Initializing Milvus Lite client with data path: {MILVUS_LITE_DATA_PATH}")
            # 确保数据目录存在
            os.makedirs(os.path.dirname(MILVUS_LITE_DATA_PATH), exist_ok=True)
            
            # 检查文件是否被锁定
            if os.path.exists(MILVUS_LITE_DATA_PATH):
                try:
                    with open(MILVUS_LITE_DATA_PATH, 'r+b') as f:
                        # 尝试获取文件锁
                        import fcntl
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                except (IOError, OSError):
                    st.warning(f"数据库文件被锁定，尝试重新连接... (尝试 {tries + 1}/{max_retries})")
                    if tries == max_retries - 1:
                        # 最后一次尝试，删除并重建数据库文件
                        os.remove(MILVUS_LITE_DATA_PATH)
                        st.warning("已删除被锁定的数据库文件，将重新创建...")
            
            # 尝试连接
            client = MilvusClient(uri=MILVUS_LITE_DATA_PATH)
            st.success("Milvus Lite 客户端初始化成功！")
            return client
            
        except Exception as e:
            tries += 1
            if tries < max_retries:
                st.warning(f"连接失败，{retry_delay}秒后重试... (尝试 {tries}/{max_retries})")
                time.sleep(retry_delay)
            else:
                st.error(f"初始化 Milvus Lite 客户端失败: {e}")
                return None
def delete_collection_if_exists(client, collection_name):
    """Deletes an existing Milvus collection if it exists."""
    try:
        if collection_name in client.list_collections():
            st.warning(f"Collection '{collection_name}' already exists. Deleting it...")
            client.drop_collection(collection_name)
            st.success(f"Collection '{collection_name}' deleted successfully.")
        else:
            st.info(f"Collection '{collection_name}' does not exist.")
    except Exception as e:
        st.error(f"Failed to delete collection '{collection_name}': {e}")
        
@st.cache_resource
def setup_milvus_collection(_client):
    """Ensures the specified collection exists and is set up correctly in Milvus Lite."""
    if not _client:
        st.error("Milvus client not available.")
        return False
    try:
        collection_name = COLLECTION_NAME
        dim = EMBEDDING_DIM

        has_collection = collection_name in _client.list_collections()

        if not has_collection:
            st.write(f"Collection '{collection_name}' not found. Creating...")
            # Define fields using new API style if needed (older style might still work)
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
                # You can add other scalar fields directly here for storage
                FieldSchema(name="content_preview", dtype=DataType.VARCHAR, max_length=500), # Example
            ]
            schema = CollectionSchema(fields, f"PubMed Lite RAG (dim={dim})")

            _client.create_collection(
                collection_name=collection_name,
                schema=schema # Pass schema directly or define dimension/primary field name
                # Or simpler:
                # dimension=dim,
                # primary_field_name="id",
                # vector_field_name="embedding",
                # metric_type=INDEX_METRIC_TYPE
            )
            st.write(f"Collection '{collection_name}' created.")

            # Create an index
            st.write(f"Creating index ({INDEX_TYPE})...")
            index_params = _client.prepare_index_params()
            index_params.add_index(
                field_name="embedding",
                index_type=INDEX_TYPE,
                metric_type=INDEX_METRIC_TYPE,
                params=INDEX_PARAMS
            )
            _client.create_index(collection_name, index_params)
            st.success(f"Index created for collection '{collection_name}'.")
        else:
            st.write(f"Found existing collection: '{collection_name}'.")
            # Optional: Check schema compatibility if needed

        # Determine current entity count (fallback between num_entities and stats)
        try:
            if hasattr(_client, 'num_entities'):
                current_count = _client.num_entities(collection_name)
            else:
                stats = _client.get_collection_stats(collection_name)
                current_count = int(stats.get("row_count", stats.get("rowCount", 0)))
            st.write(f"Collection '{collection_name}' ready. Current entity count: {current_count}")
        except Exception:
            st.write(f"Collection '{collection_name}' ready.")

        return True # Indicate collection is ready

    except Exception as e:
        st.error(f"Error setting up Milvus collection '{COLLECTION_NAME}': {e}")
        return False


def index_data_if_needed(client, data, embedding_model):
    """Checks if data needs indexing and performs it using MilvusClient."""
    global id_to_doc_map # Modify the global map

    if not client:
        st.error("Milvus client not available for indexing.")
        return False

    collection_name = COLLECTION_NAME
    
    try:
        current_count = client.num_entities(collection_name)
    except Exception:
        st.write(f"Could not retrieve entity count, attempting to (re)setup collection.")
        if not setup_milvus_collection(client):
            return False
        current_count = 0

    st.write(f"当前集合中的文档数量: {current_count}")
    
    # 限制索引的数据量
    data_to_index = data[:MAX_ARTICLES_TO_INDEX]
    needed_count = len(data_to_index)
    
    # 准备数据
    docs_for_embedding = []
    data_to_insert = []
    temp_id_map = {}

    with st.spinner("准备数据进行索引..."):
        for doc in data_to_index:
            title = doc.get('title', '') or ""
            abstract = doc.get('abstract', '') or ""
            content = doc.get('content', '') or ""
            
            if not any([title, abstract, content]):
                continue
                
            # 生成整数 ID
            doc_id = needed_count + 1  # 使用自增 ID 确保唯一性
            doc_str_id = str(doc.get('id', f"doc_{doc_id}"))  # 保存原始字符串 ID
            needed_count += 1
            
            # 更新临时映射，使用字符串 ID 作为键
            temp_id_map[doc_str_id] = {
                'title': title,
                'abstract': abstract,
                'content': content,
                'milvus_id': doc_id  # 保存 Milvus ID 映射
            }
            
            # 准备嵌入的文本
            embedding_text = f"标题: {title}\n摘要: {abstract}"
            docs_for_embedding.append(embedding_text)
            
            # 准备插入数据 - 包含所有必要字段
            preview_text = f"{title[:100]}... {abstract[:100]}..."
            data_to_insert.append({
                "id": doc_id,  # 使用整数 ID
                "embedding": None,  # 稍后填充
                "content_preview": preview_text  # 添加预览文本
            })

    if current_count < needed_count and docs_for_embedding:
        st.warning(f"需要索引新数据 ({current_count}/{needed_count} 个文档)...")

        # 生成嵌入向量
        with st.spinner("生成文档向量..."):
            start_time = time.time()
            embeddings = embedding_model.encode(docs_for_embedding, show_progress_bar=True)
            st.write(f"向量生成完成，耗时: {time.time() - start_time:.2f} 秒")

        # 填充嵌入向量
        for i, emb in enumerate(embeddings):
            data_to_insert[i]["embedding"] = emb

        # 插入数据
        with st.spinner("将数据插入到 Milvus..."):
            try:
                client.insert(collection_name=collection_name, data=data_to_insert)
                
                # 更新并保存 ID 映射
                id_to_doc_map.update(temp_id_map)
                save_id_map()
                
                st.success(f"成功索引 {len(data_to_insert)} 个文档")
                return True
            except Exception as e:
                st.error(f"插入数据失败: {e}")
                return False
    elif current_count >= needed_count:
        st.write("数据已经完全索引")
        if not id_to_doc_map:
            id_to_doc_map.update(temp_id_map)
            save_id_map()
        return True
    else:
        st.error("没有找到有效的文档内容")
        return False

def search_similar_documents(client, query, embedding_model):
    """搜索相似文档"""
    if not client or not embedding_model:
        st.error("Milvus 客户端或嵌入模型不可用")
        return [], []

    try:
        # 生成查询向量
        query_embedding = embedding_model.encode([query])[0]

        # 执行搜索（增加搜索数量以提高召回率）
        search_k = min(TOP_K * 3, 20)  # 搜索更多文档，但不超过20个
        search_params = MILVUS_SEARCH_PARAMS.get("params", {"nprobe": 16})
        result = client.search(
            collection_name=COLLECTION_NAME,
            data=[query_embedding.tolist()],
            limit=search_k,
            output_fields=["id", "content_preview"],
            params=search_params  # 正确的参数名是 params 而不是 param
        )

        if not result or not result[0]:
            return [], []

        # 处理结果
        hit_ids = []
        distances = []
        
        for hits in result:
            for hit in hits:
                milvus_id = hit.get('id')  # 获取 Milvus ID
                # 查找对应的字符串 ID
                str_id = None
                # 构建 milvus_id 到 doc_id 的反向映射
                milvus_to_doc_id = {info.get('milvus_id'): doc_id for doc_id, info in id_to_doc_map.items()}
                
                # 直接查找对应的文档ID
                str_id = milvus_to_doc_id.get(milvus_id)
                if str_id:  # 找到了有效的映射
                    hit_ids.append(str_id)
                    distances.append(hit.get('distance', 0.0))

        return hit_ids, distances
    except Exception as e:
        st.error(f"搜索过程中出错: {e}")
        return [], []