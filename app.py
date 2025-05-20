import streamlit as st
from datetime import datetime

# 首先设置页面配置
st.set_page_config(layout="wide")

import time
import os
from datetime import datetime
import torch
import gc
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from query_enhancement import QueryEnhancer
from rag_core import rerank_documents
import asyncio
import nest_asyncio

# 禁用警告和设置环境变量
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 初始化异步支持
def setup_async():
    try:
        import sys
        if sys.platform == 'darwin':  # macOS specific handling
            import platform
            if platform.processor() == 'arm':  # M1/M2 处理器
                print("在M1/M2 Mac上禁用异步循环")
                return
        
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        nest_asyncio.apply(loop)
    except Exception as e:
        print(f"Warning: 异步初始化失败，这不会影响系统的主要功能: {e}")

# 在主线程中设置异步支持
setup_async()

# 禁用Streamlit的自动重新运行和文件监视
os.environ['STREAMLIT_SERVER_ENABLE_STATIC_SERVING'] = 'false'
os.environ['STREAMLIT_SERVER_ADDRESS'] = 'localhost'
os.environ['STREAMLIT_SERVER_PORT'] = '8501'
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.environ['STREAMLIT_SERVER_RUN_ON_SAVE'] = 'false'
os.environ['STREAMLIT_WATCHER_TYPE'] = 'none'
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = './hf_cache'
os.environ['TRANSFORMERS_CACHE'] = './model_cache'

# MPS 配置优化
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.8'
os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO'] = '0.5'
os.environ['PYTORCH_MPS_ALLOCATOR_POLICY'] = 'garbage_collection_conservative'

# 添加内存管理函数
def manage_memory():
    gc.collect()
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
            torch.mps.synchronize()
            allocated = torch.mps.current_allocated_memory() / 1024**3
            print(f"当前 MPS 内存使用: {allocated:.2f} GB")
        except Exception as e:
            print(f"MPS 内存清理失败: {e}")
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

# 智能设备选择
def get_device():
    if torch.backends.mps.is_available():
        print("使用 MPS 设备加速")
        return torch.device('mps')
    elif torch.cuda.is_available():
        print("使用 CUDA 设备加速")
        return torch.device('cuda')
    else:
        print("使用 CPU 设备")
        return torch.device('cpu')

DEVICE = get_device()

# 导入其他模块的函数和配置
from config import (
    DATA_FILE, EMBEDDING_MODEL_NAME, GENERATION_MODEL_NAME, RERANKER_MODEL_NAME, TOP_K,
    MAX_ARTICLES_TO_INDEX, MILVUS_LITE_DATA_PATH, COLLECTION_NAME,
    id_to_doc_map, EMBEDDING_DIM, HYBRID_SEARCH_CONFIG
)
from data_utils import load_data
from models import load_embedding_model, load_generation_model, load_reranker_model
from milvus_utils import get_milvus_client, setup_milvus_collection, index_data_if_needed, search_similar_documents, delete_collection_if_exists
from rag_core import generate_answer, reformulate_query

# 添加缓存装饰器的模型加载函数
@st.cache_resource
def load_models():
    manage_memory()
    try:
        print("开始加载模型...")
        # 加载 embedding 模型
        embedding_model = load_embedding_model(EMBEDDING_MODEL_NAME)
        manage_memory()
        
        # 加载 reranker 模型
        reranker_model, reranker_tokenizer = load_reranker_model(
            RERANKER_MODEL_NAME,
            device=DEVICE  # 使用选择的设备
        )
        manage_memory()
        
        # 加载生成模型
        generation_model, tokenizer = load_generation_model(GENERATION_MODEL_NAME)
        manage_memory()
        
        print("所有模型加载完成")
        return embedding_model, reranker_model, reranker_tokenizer, generation_model, tokenizer
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        return None, None, None, None, None

# 在 app.py 顶部的缓存装饰器部分
@st.cache_resource
def load_query_enhancer():
    try:
        return QueryEnhancer(model_name="Qwen/Qwen2.5-0.5B")  # 使用与生成模型相同的模型
    except Exception as e:
        st.error(f"加载查询增强器失败: {e}")
        return None

# 修改现有的模型加载代码
if 'models' not in st.session_state:
    st.session_state.models = load_models()

if 'query_enhancer' not in st.session_state:
    st.session_state.query_enhancer = load_query_enhancer()

embedding_model, reranker_model, reranker_tokenizer, generation_model, tokenizer = st.session_state.models
query_enhancer = st.session_state.query_enhancer

# Streamlit UI 设置
st.title("📄 医疗 RAG 系统 (Milvus Lite)")
st.markdown(f"使用 Milvus Lite, `{EMBEDDING_MODEL_NAME}`, `{GENERATION_MODEL_NAME}`, 和 `{RERANKER_MODEL_NAME}`。")

# --- 初始化 Milvus ---
st.title("Milvus 数据库管理")

# 获取 Milvus Lite 客户端
milvus_client = get_milvus_client()

if milvus_client:
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("⚠️ 重新创建 Collection", help="将删除现有 Collection 并重新创建"):
            with st.spinner("删除并重新创建 Collection..."):
                delete_collection_if_exists(milvus_client, COLLECTION_NAME)
                time.sleep(1)  # 确保删除完成
                success = setup_milvus_collection(milvus_client)
                if success:
                    st.success("✅ Collection 已重建")
                    time.sleep(1)
                    st.rerun()  # 重新加载页面
                else:
                    st.error("❌ Collection 重建失败")
    
    with col2:
        if st.button("🔄 刷新状态", help="刷新 Collection 状态"):
            st.rerun()

# --- 初始化与缓存继续 ---
if milvus_client:
    # 检查 Collection 的维度
    try:
        schema = milvus_client.describe_collection(COLLECTION_NAME)
        vec_field = next((f for f in schema["fields"] if f["name"] == "embedding"), None)
        if vec_field:
            collection_dim = int(vec_field["params"].get("dim", -1))
            st.info(f"📌 当前 Collection `{COLLECTION_NAME}` 的维度：{collection_dim}，你的模型配置维度：{EMBEDDING_DIM}")

            if collection_dim != EMBEDDING_DIM:
                st.error("维度不一致，请删除 Collection 后重建。")

    except Exception as e:
        st.warning(f"无法获取 Collection 维度信息: {e}")
        
    # 设置 Milvus Collection
    collection_is_ready = setup_milvus_collection(milvus_client)

    models_loaded = embedding_model and generation_model and tokenizer and reranker_model and reranker_tokenizer

    if collection_is_ready and models_loaded:
        pubmed_data = load_data(DATA_FILE)

        if pubmed_data:
            indexing_successful = index_data_if_needed(milvus_client, pubmed_data, embedding_model)
        else:
            st.warning(f"无法从 {DATA_FILE} 加载数据。跳过索引。")
            indexing_successful = False

        st.divider()

        if not indexing_successful and not id_to_doc_map:
            st.error("数据索引失败或不完整，且没有文档映射。RAG 功能已禁用。")
        else:
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []

            # 设置页面布局
            col1, col2 = st.columns([3, 1])
            with col1:
                query = st.text_input("请提出关于已索引医疗文章的问题:", key="query_input")
            with col2:
                enable_query_enhancement = st.toggle("启用查询优化", value=True, help="开启后会对查询进行优化，以提高检索效果")

            # 输出查询历史
            if st.session_state.chat_history:
                st.subheader("历史对话")
                for i, qa in enumerate(st.session_state.chat_history):
                    with st.container():
                        col1, col2 = st.columns([1, 11])
                        with col1:
                            st.write("👤")
                        with col2:
                            st.markdown(f"**Q{i+1}:** {qa.get('human', '')}")
                            
                        col1, col2 = st.columns([1, 11])
                        with col1:
                            st.write("🤖")
                        with col2:
                            st.markdown(f"**A{i+1}:** {qa.get('assistant', '')}")
                            
                        # 显示检索的文档数量
                        if qa.get('retrieved_docs'):
                            st.caption(f"参考文档数: {len(qa['retrieved_docs'])}")
                        st.divider()

            if st.button("获取答案", key="submit_button") and query:
                start_time = time.time()
                with st.spinner("搜索相关文档中..."):
                    try:
                        if enable_query_enhancement:
                            try:
                                enhanced_query = query_enhancer.enhance_query(
                                    query,
                                    st.session_state.chat_history if st.session_state.chat_history else None
                                )
                            except Exception as e:
                                st.error(f"查询优化失败: {e}")
                                enhanced_query = query
                            if enhanced_query and enhanced_query != query:
                                st.info(f"🔄 优化后的查询: {enhanced_query}")
                        else:
                            enhanced_query = query
                    except Exception as e:
                        st.warning(f"查询优化失败，使用原始查询。错误：{str(e)}")
                        enhanced_query = query

                # 创建混合检索器实例
                if 'hybrid_searcher' not in st.session_state:
                    from hybrid_search import HybridSearcher
                    st.session_state.hybrid_searcher = HybridSearcher()
                    # 预处理所有文档
                    all_docs = list(id_to_doc_map.values())
                    st.session_state.hybrid_searcher.preprocess_documents(all_docs)

                with st.spinner("正在进行混合检索..."):
                    # 获取稠密向量检索结果
                    retrieved_ids, distances = search_similar_documents(
                        milvus_client, 
                        enhanced_query,  # 使用增强后的查询
                        embedding_model
                    )
                    
                    # 将稠密检索的距离转换为分数（越大越好）
                    dense_scores = [-d for d in distances]  # 转换距离为分数
                    
                    # 进行混合检索
                    if retrieved_ids:
                        try:
                            # 确保有足够的文档进行混合检索
                            if len(retrieved_ids) >= 2:
                                retrieved_docs, hybrid_scores = st.session_state.hybrid_searcher.hybrid_search(
                                    enhanced_query,
                                    dense_scores,
                                    retrieved_ids
                                )
                                if retrieved_docs:
                                    distances = [-s for s in hybrid_scores]  # 更新距离分数
                                else:
                                    # 如果混合检索失败，回退到原始检索结果
                                    retrieved_docs = [id_to_doc_map[id] for id in retrieved_ids if id in id_to_doc_map]
                            else:
                                # 文档数量太少，使用原始检索结果
                                retrieved_docs = [id_to_doc_map[id] for id in retrieved_ids if id in id_to_doc_map]
                        except Exception as e:
                            st.warning(f"混合检索失败: {e}，使用原始检索结果")
                            retrieved_docs = [id_to_doc_map[id] for id in retrieved_ids if id in id_to_doc_map]

                if not retrieved_ids:
                    st.warning("在数据库中找不到相关文档。")
                else:
                    initial_docs = [id_to_doc_map[id] for id in retrieved_ids if id in id_to_doc_map]

                    retrieved_docs = rerank_documents(query, initial_docs, reranker_model, reranker_tokenizer, DEVICE)

                    if not retrieved_docs:
                        st.error("检索到的 ID 无法映射到加载的文档。请检查映射逻辑。")
                    else:
                        st.subheader("检索到的上下文文档:")
                        for i, doc in enumerate(retrieved_docs):
                            dist_str = f", 距离: {distances[i]:.4f}" if distances else ""
                            with st.expander(f"文档 {i+1} (ID: {retrieved_ids[i]}{dist_str}) - {doc['title'][:60]}"):
                                st.write(f"**标题:** {doc['title']}")
                                st.write(f"**摘要:** {doc['abstract']}")

                        st.divider()

                        st.subheader("生成的答案:")
                        with st.spinner("正在根据上下文生成答案..."):
                            answer = generate_answer(query, retrieved_docs, generation_model, tokenizer)
                            st.write(answer)

                        st.session_state.chat_history.append({
                            "human": query,
                            "assistant": answer,
                            "retrieved_docs": retrieved_docs,
                            "timestamp": datetime.now().isoformat()
                        })

                        # 反馈收集界面
                        col1, col2 = st.columns(2)
                        with col1:
                            satisfaction = st.slider("请对回答的质量进行评分(1-5分):", 1, 5, 3, key=f"satisfaction_{len(st.session_state.chat_history)}")
                        with col2:
                            feedback_type = st.radio("您对这个回答满意吗？", ["满意", "需要改进"], key=f"feedback_{len(st.session_state.chat_history)}")
                        
                        # 初始化反馈管理器
                        if 'feedback_manager' not in st.session_state:
                            from feedback_manager import FeedbackManager
                            st.session_state.feedback_manager = FeedbackManager()

                        # 文档相关性反馈
                        with st.expander("文档相关性反馈"):
                            for i, doc in enumerate(retrieved_docs[:3]):
                                st.write(f"**文档 {i+1}:** {doc['title'][:60]}")
                                doc_relevance = st.slider(f"此文档与问题的相关性(1-5分):", 1, 5, 3, key=f"doc_relevance_{i}")
                                doc_notes = st.text_input("相关性备注(可选):", key=f"doc_notes_{i}")
                                if st.button("提交文档反馈", key=f"submit_doc_feedback_{i}"):
                                    st.session_state.feedback_manager.add_document_feedback(
                                        doc_id=str(retrieved_ids[i]),
                                        query=query,
                                        relevance_score=doc_relevance,
                                        user_notes=doc_notes
                                    )
                                    st.success(f"文档 {i+1} 的反馈已保存")

                        if feedback_type == "需要改进":
                            user_feedback = st.text_area("请详细描述您的反馈:", key="user_feedback")
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("提交反馈", key="submit_feedback"):
                                    # 保存反馈
                                    st.session_state.feedback_manager.add_conversation_feedback(
                                        query=query,
                                        answer=answer,
                                        feedback=user_feedback,
                                        satisfaction=satisfaction,
                                        retrieved_docs=retrieved_docs,
                                        chat_history=st.session_state.chat_history,
                                        alpha=HYBRID_SEARCH_CONFIG["vector_weight"]
                                    )
                                    st.success("感谢您的反馈！")
                            with col2:
                                if st.button("重新生成答案", key="refine_button"):
                                    new_query = reformulate_query(
                                        query, 
                                        user_feedback, 
                                        answer,
                                        st.session_state.chat_history[-3:]  # 使用最近3轮对话作为上下文
                                    )
                                    st.session_state.chat_history.append({
                                        "human": new_query,
                                        "assistant": None,
                                        "retrieved_docs": None,
                                        "timestamp": datetime.now().isoformat()
                                    })
                                    # 基于反馈更新混合检索权重
                                    new_vector_weight = st.session_state.feedback_manager.get_optimal_alpha()
                                    HYBRID_SEARCH_CONFIG["vector_weight"] = new_vector_weight
                                    HYBRID_SEARCH_CONFIG["bm25_weight"] = 1.0 - new_vector_weight
                                    st.experimental_rerun()
                        else:
                            if st.button("提交反馈", key="submit_positive_feedback"):
                                st.session_state.feedback_manager.add_conversation_feedback(
                                    query=query,
                                    answer=answer,
                                    feedback="满意",
                                    satisfaction=satisfaction,
                                    retrieved_docs=retrieved_docs,
                                    chat_history=st.session_state.chat_history,
                                    alpha=HYBRID_SEARCH_CONFIG["vector_weight"]
                                )
                                st.success("感谢您的反馈！")

                end_time = time.time()
                st.info(f"总耗时: {end_time - start_time:.2f} 秒")

    else:
        st.error("加载模型或设置 Milvus Lite collection 失败。请检查日志和配置。")
else:
    st.error("初始化 Milvus Lite 客户端失败。请检查日志。")

# --- 页脚/信息侧边栏 ---
st.sidebar.header("系统配置")
st.sidebar.markdown(f"**向量存储:** Milvus Lite")
st.sidebar.markdown(f"**数据路径:** `{MILVUS_LITE_DATA_PATH}`")
st.sidebar.markdown(f"**Collection:** `{COLLECTION_NAME}`")
st.sidebar.markdown(f"**数据文件:** `{DATA_FILE}`")
st.sidebar.markdown(f"**嵌入模型:** `{EMBEDDING_MODEL_NAME}`")
st.sidebar.markdown(f"**生成模型:** `{GENERATION_MODEL_NAME}`")
st.sidebar.markdown("**重排序模型:** `BAAI/bge-reranker-v2-m3`")
st.sidebar.markdown(f"**最大索引数:** `{MAX_ARTICLES_TO_INDEX}`")
st.sidebar.markdown(f"**检索 Top K:** `{TOP_K}`")

def rerank_documents(query, docs, model, tokenizer, device):
    """对检索到的文档进行重排序"""
    if not docs:
        return []
        
    try:
        # 准备重排序的文本对
        text_pairs = []
        for doc in docs:
            title = doc.get('title', '')
            abstract = doc.get('abstract', '')
            content = doc.get('content', '')
            
            # 组合文档内容
            doc_text = f"标题: {title}\n摘要: {abstract}\n内容: {content}"
            text_pairs.append((query, doc_text))

        # 批处理重排序
        scores = []
        batch_size = 4  # 根据显存大小调整
        
        for i in range(0, len(text_pairs), batch_size):
            batch_pairs = text_pairs[i:i + batch_size]
            features = tokenizer(
                batch_pairs,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512
            )
            
            try:
                features = {k: v.to(device) for k, v in features.items()}
                with torch.no_grad():
                    scores.extend(model(**features).logits.flatten().tolist())
            except RuntimeError:  # 如果显存不足,回退到CPU
                model.to('cpu')
                features = {k: v.to('cpu') for k, v in features.items()}
                with torch.no_grad():
                    scores.extend(model(**features).logits.flatten().tolist())
                if str(device).startswith('cuda'):
                    model.to(device)  # 恢复到GPU

        # 结合文档和分数
        doc_scores = list(zip(docs, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in doc_scores]
    except Exception as e:
        st.error(f"文档重排序失败: {e}")
        return docs  # 发生错误时返回原始文档列表