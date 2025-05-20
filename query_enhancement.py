from typing import List, Dict, Optional
import re
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_huggingface import HuggingFacePipeline  # 更新导入
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    pipeline
)
import torch

class QueryEnhancer:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B"):
        print(f"初始化 QueryEnhancer, 模型: {model_name}")
        try:
            # 初始化 tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            # 智能设备选择
            if torch.backends.mps.is_available():
                device = 'mps'
                print("使用 MPS 设备加速")
            elif torch.cuda.is_available():
                device = 'cuda'
                print("使用 CUDA 设备加速")
            else:
                device = 'cpu'
                print("使用 CPU 设备")

            # 初始化模型
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                device_map=device if device == 'cuda' else None,  # 只在CUDA下使用device_map
                torch_dtype=torch.float32 if device == 'cpu' else torch.float16,  # MPS和CUDA使用float16
                low_cpu_mem_usage=True,
                use_cache=True
            )
            if device != 'cuda':  # 对于MPS和CPU，手动移动模型
                self.model.to(device)
            
            # 确保模型有正确的 token
            if not hasattr(self.tokenizer, 'pad_token') or self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # 禁用注意力警告
            import warnings
            warnings.filterwarnings("ignore", message=".*Sliding Window Attention.*")
            
            # 创建 pipeline 并配置生成参数
            pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=32,       # 减少生成长度
                temperature=0.3,         # 降低随机性
                do_sample=False,         # 禁用采样以获得确定性输出
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                device_map=None          # 让模型使用已分配的设备
            )
            
            # 使用新版 HuggingFacePipeline
            self.llm = HuggingFacePipeline(pipeline=pipe)
            
            # 优化提示模板
            self.expand_template = PromptTemplate(
                input_variables=["query"],
                template="""请分析下面这个医学查询，提取关键信息并生成优化后的查询：

用户查询：{query}

你的任务是：
1. 识别和保留最重要的医学症状、疾病名称或诊断相关的关键词
2. 删除无关的修饰词和口语化表达
3. 如果发现医学术语的错误用法，替换为正确的专业术语
4. 将口语化的症状描述转换为医学术语

要求：
- 生成一个精简的医学查询
- 控制在5-15个汉字之内
- 使用标准的医学术语
- 保持专业性和准确性
- 确保查询的完整性和可理解性

只需返回优化后的查询，不要添加任何解释或标点符号。"""
            )
            
            # 使用新版 RunnableSequence
            self.expand_chain = self.expand_template | self.llm
            
            print(f"QueryEnhancer 初始化成功")
            
        except Exception as e:
            print(f"QueryEnhancer 初始化失败: {e}")
            raise
    
    def expand_query(self, query: str) -> str:
        """将用户查询优化为标准的医学查询格式"""
        try:
            # 检查输入的合法性
            if not query or not query.strip():
                print("收到空查询，返回原始输入")
                return query

            # 清理和规范化查询
            cleaned_query = ' '.join(query.split())
            cleaned_query = re.sub(r'[?？!！。，,;；]', '', cleaned_query)  # 移除标点符号
            
            # 如果查询已经很规范（长度适中且包含关键医学术语），直接返回
            if 5 <= len(cleaned_query) <= 15 and any(term in cleaned_query for term in ['症', '病', '癌', '炎', '痛']):
                print(f"查询已经很规范，保持原样: {cleaned_query}")
                return cleaned_query
            
            # 生成优化后的查询
            result = self.expand_chain.invoke({
                "query": cleaned_query
            })
            
            # 验证并清理结果
            result = str(result).strip()
            if not result:
                print("模型返回空结果，使用原始查询")
                return cleaned_query
            
            # 标准化处理
            result = re.sub(r'\s+', '', result)  # 移除所有空白字符
            result = re.sub(r'[?？!！。，,;；]', '', result)  # 移除标点符号
            
            # 验证优化后的查询
            if not (5 <= len(result) <= 15 and any(c for c in result if '\u4e00' <= c <= '\u9fff')):
                print(f"优化结果不符合要求，使用原始查询: {result}")
                return cleaned_query
                
            print(f"原始查询: {cleaned_query}")
            print(f"优化结果: {result}")
            return result
            
            # 过滤并验证查询
            valid_queries = []
            for q in queries:
                # 移除可能的编号前缀和多余空格
                q = re.sub(r'^\d+[\.\s]+', '', q.strip())
                # 移除可能的起始标记，如"查询："
                q = re.sub(r'^(查询[:：]|问题[:：]|症状[:：])', '', q.strip())
                # 验证查询长度并确保查询有意义
                if 2 <= len(q) <= 20 and any(c for c in q if '\u4e00' <= c <= '\u9fff'):  # 包含中文字符
                    valid_queries.append(q)
            
            # 如果没有有效的优化查询，返回原始查询
            if not valid_queries:
                print("没有生成有效的优化查询，使用原始查询")
                return cleaned_query
                
            # 对优化查询进行排序和去重
            unique_queries = list(dict.fromkeys(valid_queries))  # 保持顺序的去重
            
            # 从有效查询中选择最相关的一个（默认使用第一个，因为它通常是最相关的）
            best_query = unique_queries[0]
            
            print(f"原始查询: {cleaned_query}")
            print(f"优化结果: {unique_queries}")
            print(f"选择查询: {best_query}")
            
            return best_query
            
        except Exception as e:
            print(f"查询优化失败: {e}")
            return query
            
    def enhance_query(self,
                    query: str,
                    chat_history: Optional[List[Dict]] = None) -> str:
        """进行查询优化"""
        try:
            return self.expand_query(query)
        except Exception as e:
            print(f"查询优化失败: {e}")
            return query