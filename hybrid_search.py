# 混合检索实现
from rank_bm25 import BM25Okapi
import numpy as np
from typing import List, Dict, Tuple
from sklearn.preprocessing import MinMaxScaler
from config import HYBRID_SEARCH_CONFIG, CACHE_CONFIG
from collections import OrderedDict
import time

class HybridSearcher:
    def __init__(self):
        self.bm25 = None
        self.tokenized_corpus = None
        self.doc_mapping = {}
        self.doc_index_mapping = {}
        self.scaler = MinMaxScaler()
        self.preprocessed_docs = []  # 存储预处理后的文档
        self._preprocessed = False   # 跟踪预处理状态
        
        # 使用 OrderedDict 作为简单缓存
        self.doc_cache = OrderedDict()
        self.cache_size = CACHE_CONFIG.get("doc_cache_size", 1000)

    def tokenize_text(self, text: str) -> List[str]:
        """简单的文本分词"""
        return text.lower().split()

    def preprocess_documents(self, documents: List[Dict]) -> None:
        """预处理文档用于BM25检索"""
        if self._preprocessed:
            print("文档已经预处理过，跳过...")
            return
            
        try:
            # 重置状态
            self.doc_mapping = {}
            self.doc_index_mapping = {}
            self.tokenized_corpus = []
            self.preprocessed_docs = []
            
            # 处理每个文档
            for i, doc in enumerate(documents):
                # 确保每个文档都有唯一ID
                doc_id = str(doc.get('id', i))
                if 'id' not in doc:
                    doc['id'] = doc_id
                    
                # 存储文档和映射
                self.doc_mapping[doc_id] = doc
                self.preprocessed_docs.append(doc)
                
                # 处理文档内容
                title = str(doc.get('title', '')).strip()
                abstract = str(doc.get('abstract', '')).strip()
                content = str(doc.get('content', '')).strip()
                text = ' '.join(filter(None, [title, abstract, content]))
                
                # 分词并存储
                tokens = self.tokenize_text(text)
                self.tokenized_corpus.append(tokens)
                self.doc_index_mapping[doc_id] = i
            
            # 初始化BM25
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            print(f"已处理 {len(documents)} 个文档用于混合检索")
            self._preprocessed = True
            
        except Exception as e:
            print(f"预处理文档时出错: {e}")
            raise

    def normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """归一化分数到 [0,1] 区间"""
        if len(scores) == 0:
            return scores
        if len(scores) == 1:
            return np.array([1.0])
        min_score = np.min(scores)
        max_score = np.max(scores)
        if max_score == min_score:
            return np.ones_like(scores)
        return (scores - min_score) / (max_score - min_score)

    def hybrid_search(
            self, 
            query: str,
            dense_scores: List[float],
            retrieved_indices: List[int]) -> Tuple[List[Dict], List[float]]:
        """
        执行混合检索，结合稠密检索和稀疏检索的结果。
        使用 HYBRID_SEARCH_CONFIG 中的权重配置。
        """
        try:
            if not retrieved_indices:
                return [], []
                
            # 1. 检查缓存
            cache_key = f"{query}_{hash(str(retrieved_indices))}"
            if cache_key in self.doc_cache:
                return self.doc_cache[cache_key]
                
            # 2. 对检索到的文档进行BM25计算
            docs = [self.doc_mapping[str(idx)] for idx in retrieved_indices if str(idx) in self.doc_mapping]
            if not docs:
                return [], []
                
            # 3. 计算BM25分数
            corpus = [' '.join([doc.get('title', ''), doc.get('abstract', '')]) for doc in docs]
            tokenized_corpus = [self.tokenize_text(text) for text in corpus]
            bm25 = BM25Okapi(tokenized_corpus)
            bm25_scores = bm25.get_scores(self.tokenize_text(query))
            
            # 4. 归一化分数
            dense_scores = np.array(dense_scores)
            bm25_scores = np.array(bm25_scores)
            dense_scores = self.normalize_scores(dense_scores)
            bm25_scores = self.normalize_scores(bm25_scores)
            
            # 5. 加权组合
            vector_weight = HYBRID_SEARCH_CONFIG["vector_weight"]
            combined_scores = vector_weight * dense_scores + (1 - vector_weight) * bm25_scores
            
            # 6. 过滤和排序结果
            indices = np.argsort(combined_scores)[::-1][:HYBRID_SEARCH_CONFIG.get("top_k", 3)]
            filtered_docs = [docs[i] for i in indices]
            filtered_scores = combined_scores[indices]
            
            # 7. 更新缓存
            self.doc_cache[cache_key] = (filtered_docs, filtered_scores.tolist())
            if len(self.doc_cache) > self.cache_size:
                self.doc_cache.popitem(last=False)  # 移除最早的项目
                
            return filtered_docs, filtered_scores.tolist()
        except Exception as e:
            print(f"混合检索失败: {str(e)}")
            return [], []