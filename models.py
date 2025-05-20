import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers import AutoModelForSequenceClassification
import torch

def load_reranker_model(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    return model, tokenizer

    return model, tokenizer, device

@st.cache_resource
def load_embedding_model(model_name):
    """Loads the sentence transformer model."""
    st.write(f"Loading embedding model: {model_name}...")
    try:
        model = SentenceTransformer(model_name)
        st.success("Embedding model loaded.")
        return model
    except Exception as e:
        st.error(f"Failed to load embedding model: {e}")
        return None

def load_generation_model(model_name, **kwargs):
    print(f"加载生成模型: {model_name}, 参数: {kwargs}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # 根据模型名称选择合适的模型类
        model_kwargs = {
            'device_map': 'auto',
            'torch_dtype': torch.float16,
            'low_cpu_mem_usage': True
        }
        
        if "t5" in model_name.lower() or "bart" in model_name.lower():
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                **model_kwargs
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
        print(f"生成模型加载成功: {model_name}, 设备: {model.device}")
        return model, tokenizer
    except Exception as e:
        print(f"加载生成模型失败: {e}")
        raise