import json
from datetime import datetime
from typing import List, Dict, Optional
import os

class FeedbackManager:
    def __init__(self, feedback_file: str = "feedback_data.json"):
        """初始化反馈管理器"""
        self.feedback_file = feedback_file
        self.feedback_data = self._load_feedback_data()
        
    def _load_feedback_data(self) -> Dict:
        """加载反馈数据"""
        if os.path.exists(self.feedback_file):
            try:
                with open(self.feedback_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {"conversations": [], "document_feedback": {}, "alpha_history": []}
        return {"conversations": [], "document_feedback": {}, "alpha_history": []}
        
    def _save_feedback_data(self):
        """保存反馈数据"""
        with open(self.feedback_file, 'w', encoding='utf-8') as f:
            json.dump(self.feedback_data, f, ensure_ascii=False, indent=2)
            
    def add_conversation_feedback(self,
                                query: str,
                                answer: str,
                                feedback: str,
                                satisfaction: int,
                                retrieved_docs: List[Dict],
                                chat_history: List[Dict],
                                alpha: float) -> None:
        """添加对话反馈"""
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "answer": answer,
            "feedback": feedback,
            "satisfaction": satisfaction,  # 1-5分
            "retrieved_docs": [doc.get("id", "") for doc in retrieved_docs],
            "chat_history_length": len(chat_history),
            "alpha": alpha
        }
        self.feedback_data["conversations"].append(feedback_entry)
        self._save_feedback_data()
        
    def update_alpha_history(self, alpha: float, performance_score: float):
        """更新alpha参数历史"""
        self.feedback_data["alpha_history"].append({
            "timestamp": datetime.now().isoformat(),
            "alpha": alpha,
            "performance": performance_score
        })
        self._save_feedback_data()
        
    def get_optimal_alpha(self, window_size: int = 10) -> float:
        """获取最优的alpha值"""
        if not self.feedback_data["alpha_history"]:
            return 0.5
            
        # 获取最近的记录
        recent_history = self.feedback_data["alpha_history"][-window_size:]
        if not recent_history:
            return 0.5
            
        # 计算加权平均
        total_weight = 0
        weighted_sum = 0
        for i, record in enumerate(recent_history):
            weight = (i + 1) / len(recent_history)  # 越新的数据权重越大
            weighted_sum += record["alpha"] * record["performance"] * weight
            total_weight += weight
            
        return weighted_sum / total_weight if total_weight > 0 else 0.5
        
    def add_document_feedback(self, 
                            doc_id: str, 
                            query: str, 
                            relevance_score: int,
                            user_notes: Optional[str] = None) -> None:
        """添加文档相关性反馈"""
        if doc_id not in self.feedback_data["document_feedback"]:
            self.feedback_data["document_feedback"][doc_id] = []
            
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "relevance_score": relevance_score,  # 1-5分
            "user_notes": user_notes
        }
        self.feedback_data["document_feedback"][doc_id].append(feedback_entry)
        self._save_feedback_data()
        
    def get_document_feedback_stats(self, doc_id: str) -> Dict:
        """获取文档反馈统计信息"""
        if doc_id not in self.feedback_data["document_feedback"]:
            return {"avg_relevance": 0, "feedback_count": 0}
            
        feedbacks = self.feedback_data["document_feedback"][doc_id]
        if not feedbacks:
            return {"avg_relevance": 0, "feedback_count": 0}
            
        avg_relevance = sum(f["relevance_score"] for f in feedbacks) / len(feedbacks)
        return {
            "avg_relevance": avg_relevance,
            "feedback_count": len(feedbacks)
        }
        
    def analyze_conversation_patterns(self) -> Dict:
        """分析对话模式"""
        if not self.feedback_data["conversations"]:
            return {}
            
        total_conversations = len(self.feedback_data["conversations"])
        avg_satisfaction = sum(c["satisfaction"] for c in self.feedback_data["conversations"]) / total_conversations
        
        # 分析多轮对话效果
        multi_turn_conversations = [c for c in self.feedback_data["conversations"] if c["chat_history_length"] > 0]
        multi_turn_satisfaction = 0
        if multi_turn_conversations:
            multi_turn_satisfaction = sum(c["satisfaction"] for c in multi_turn_conversations) / len(multi_turn_conversations)
            
        return {
            "total_conversations": total_conversations,
            "avg_satisfaction": avg_satisfaction,
            "multi_turn_ratio": len(multi_turn_conversations) / total_conversations,
            "multi_turn_satisfaction": multi_turn_satisfaction
        }