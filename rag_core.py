import streamlit as st
import torch
from config import MAX_NEW_TOKENS_GEN, TEMPERATURE, TOP_P, REPETITION_PENALTY
from torch.nn.functional import softmax
import numpy as np

def get_conversation_context(chat_history, max_turns=3):
    """
    从对话历史中获取上下文
    """
    if not chat_history:
        return ""
        
    # 只使用最近的几轮对话
    recent_history = chat_history[-max_turns:]
    context_parts = []
    
    for turn in recent_history:
        user_query = turn.get('human', '')
        system_response = turn.get('assistant', '')
        if user_query and system_response:
            context_parts.append(f"用户: {user_query}\n系统: {system_response}")
            
    return "\n\n".join(context_parts)

def generate_answer(query, context_docs, gen_model, tokenizer, device='cpu'):
    """生成答案"""
    if not context_docs:
        return "抱歉，我找不到相关的文档来回答您的问题。"
    if not gen_model or not tokenizer:
        st.error("生成模型或分词器不可用")
        return "错误：生成组件未加载。"

    try:
        # 整合上下文，确保文档的相关性
        context_parts = []
        for i, doc in enumerate(context_docs[:3], 1):  # 只使用前3个最相关的文档
            title = doc.get('title', '').strip()
            abstract = doc.get('abstract', '').strip()
            content = doc.get('content', '').strip()
            
            if title and (abstract or content):
                doc_text = f"文档{i}:\n标题: {title}\n"
                if abstract:
                    doc_text += f"摘要: {abstract}\n"
                if content:
                    doc_text += f"内容: {content}\n"
                context_parts.append(doc_text)
        
        context = "\n\n---\n\n".join(context_parts)

        # 优化中文提示词
        prompt = f"""请仅基于以下参考文档回答用户的问题。
如果答案在参考文档中找不到，请明确说明。请不要编造信息。

参考文档：
{context}

用户问题：{query}

回答："""

        # 分批处理以处理长文本
        max_context_length = 2048
        inputs = tokenizer(prompt, 
                         return_tensors="pt",
                         truncation=True, 
                         max_length=max_context_length)
        
        # 将输入移动到正确的设备
        inputs = {k: v.to(device) for k, v in inputs.items()}
        gen_model = gen_model.to(device)

        with torch.no_grad():
            outputs = gen_model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS_GEN,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                repetition_penalty=REPETITION_PENALTY,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,  # 启用采样
                no_repeat_ngram_size=3,  # 避免重复
                num_beams=1  # 使用简单生成
            )

        # 只解码新生成的标记
        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        # 清理和格式化响应
        response = response.strip()
        if not response:
            return "抱歉，我无法基于当前的参考文档生成有意义的回答。"

        return response

    except Exception as e:
        st.error(f"生成过程中出错: {e}")
        return "抱歉，在生成答案时遇到了错误。"

def rerank_documents(query, docs, reranker_model, reranker_tokenizer, device):
    """重新排序检索到的文档"""
    if not docs:
        return []
        
    try:
        # 准备重排序的文本对
        text_pairs = []
        for doc in docs:
            title = doc.get('title', '')
            abstract = doc.get('abstract', '')
            content = doc.get('content', '')
            
            # 优先使用摘要，如果没有则使用内容的前一部分
            doc_text = abstract if abstract else (content[:512] if content else title)
            if not doc_text:
                continue
                
            text_pairs.append((query, doc_text))

        if not text_pairs:
            return docs

        # 批处理重排序
        batch_size = 4  # 根据显存大小调整
        all_scores = []
        
        try:
            reranker_model = reranker_model.to(device)
            device_to_use = device
        except RuntimeError:
            print("设备内存不足，回退到 CPU")
            reranker_model = reranker_model.cpu()
            device_to_use = 'cpu'

        for i in range(0, len(text_pairs), batch_size):
            batch_pairs = text_pairs[i:i + batch_size]
            features = reranker_tokenizer(
                batch_pairs,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512
            )

            features = {k: v.to(device_to_use) for k, v in features.items()}
            
            with torch.no_grad():
                scores = reranker_model(**features).logits.flatten()
                all_scores.extend(scores.cpu().tolist())

        # 标准化分数
        if all_scores:
            scores_array = np.array(all_scores)
            scores_array = (scores_array - scores_array.mean()) / scores_array.std()
            
            # 结合文档和分数，过滤掉异常值
            valid_pairs = [
                (doc, score) 
                for doc, score in zip(docs, scores_array) 
                if not np.isnan(score) and abs(score) < 3  # 过滤掉3个标准差以外的异常值
            ]
            
            if valid_pairs:
                # 按分数排序
                valid_pairs.sort(key=lambda x: x[1], reverse=True)
                return [doc for doc, _ in valid_pairs]

        return docs  # 如果重排序失败，返回原始顺序

    except Exception as e:
        print(f"文档重排序失败: {e}")
        return docs  # 发生错误时返回原始文档列表

def reformulate_query(query, user_feedback, previous_answer, chat_history=None):
    """
    基于反馈和历史对话的查询改写方法。
    """
    context = get_conversation_context(chat_history) if chat_history else ""
    
    reformulated = (
        f"历史对话上下文:\n{context}\n\n" if context else ""
        f"原始问题：{query}\n"
        f"上次系统回答：{previous_answer}\n"
        f"用户反馈：{user_feedback}\n"
        f"请根据以上信息，重新提出更明确的问题以获得更好答案。"
    )
    return reformulated