import torch
import json
import os

def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
DEVICE = get_device()
BATCH_SIZE = 4 if DEVICE.type == 'mps' else 8
# Milvus Lite Configuration
MILVUS_LITE_DATA_PATH = "/Users/lujiayi/Downloads/data-mining-and-knowledge-processing-main/2025-spring/exp04-easy-rag-system/milvus_lite_data.db" # Path to store Milvus Lite data
COLLECTION_NAME = "medical_rag_lite" # Use a different name if needed

# Data Configuration
DATA_FILE = "./data/processed_data.json"

# Model Configuration
# Example: 'all-MiniLM-L6-v2' (dim 384), 'thenlper/gte-large' (dim 1024)
EMBEDDING_MODEL_NAME = 'BAAI/bge-m3'
GENERATION_MODEL_NAME = "Qwen/Qwen2.5-0.5B"
EMBEDDING_DIM = 1024  # BGE-m3 embedding dimension

RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

# Indexing and Search Parameters
MAX_ARTICLES_TO_INDEX = 500
TOP_K = 3
# Milvus index parameters (adjust based on data size and needs)
INDEX_METRIC_TYPE = "L2" # Or "IP"
INDEX_TYPE = "IVF_FLAT"  # Milvus Lite 支持的索引类型
# HNSW index params (adjust as needed)
INDEX_PARAMS = {"nlist": 128}
# Milvus search params
MILVUS_SEARCH_PARAMS = {
    "metric_type": "COSINE",  # 使用余弦相似度
    "params": {
        "nprobe": 16,         # 增加探测数量以提高召回率
        "ef": 64             # 增加搜索精度
    }
}

# Generation Parameters
MAX_NEW_TOKENS_GEN = 512
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.1
DO_SAMPLE = True

# 混合检索配置
HYBRID_SEARCH_CONFIG = {
    "vector_weight": 0.7,     # 向量检索权重
    "bm25_weight": 0.3,      # BM25权重
    "min_score": 0.1,        # 最小分数阈值
    "top_k": TOP_K,          # 返回的最大文档数
}

# 生成参数配置
GENERATION_CONFIG = {
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "do_sample": True,
    "no_repeat_ngram_size": 3
}

# 缓存配置
CACHE_CONFIG = {
    "doc_cache_size": 1000,      # 文档缓存大小
    "cache_ttl": 3600,           # 缓存生存时间（秒）
    "enable_cache": True         # 是否启用缓存
}

# ID映射配置
ID_MAP_FILE = "./data/id_map.json"
id_to_doc_map = {}

def save_id_map():
    """保存文档ID映射到文件"""
    try:
        os.makedirs(os.path.dirname(ID_MAP_FILE), exist_ok=True)
        with open(ID_MAP_FILE, 'w', encoding='utf-8') as f:
            json.dump(id_to_doc_map, f, ensure_ascii=False, indent=2)
        print(f"成功保存ID映射到: {ID_MAP_FILE}")
    except Exception as e:
        print(f"保存ID映射时出错: {str(e)}")

def load_id_map():
    """从文件加载文档ID映射"""
    global id_to_doc_map
    try:
        if os.path.exists(ID_MAP_FILE):
            with open(ID_MAP_FILE, 'r', encoding='utf-8') as f:
                id_to_doc_map = json.load(f)
            print(f"成功从 {ID_MAP_FILE} 加载ID映射")
    except Exception as e:
        print(f"加载ID映射时出错: {str(e)}")
        id_to_doc_map = {}