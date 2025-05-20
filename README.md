# 医疗RAG智能问答系统

## 项目简介

本项目实现了一个面向医疗领域的RAG（Retrieval-Augmented Generation）智能问答系统，结合了向量检索、BM25稀疏检索、查询优化、文档重排序和多轮对话反馈机制，能够为用户提供高质量的医学知识问答服务。

## 数据集说明

本项目所用医学原始数据集因涉及版权问题，暂不公开。如需测试请自备医学相关markdown文档。

## 主要功能

- **医学文档预处理**：支持Markdown批量清洗、分块、摘要生成。
- **向量数据库管理**：集成Milvus Lite，实现高效医学文档向量化检索。
- **混合检索**：结合向量检索与BM25，提升召回率和相关性。
- **查询优化**：对用户输入进行医学术语标准化和智能改写。
- **文档重排序**：基于BGE-Reranker对检索结果进行相关性再排序。
- **多轮对话与反馈**：支持上下文追踪、用户反馈采集与参数自适应。

## 目录结构

```text
├── app.py                  # Streamlit主入口
├── config.py               # 配置文件
├── preprocess_md.py        # 医学文档预处理
├── milvus_utils.py         # 向量数据库管理
├── hybrid_search.py        # 混合检索逻辑
├── query_enhancement.py    # 查询优化
├── rag_core.py             # RAG主流程（生成/重排序/多轮对话）
├── feedback_manager.py     # 用户反馈与参数自适应
├── requirements.txt        # 依赖包列表
├── README.md               # 项目说明
└── data/                   # 预处理数据与ID映射
```

## 快速开始

1. **环境准备**

   ```zsh
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **数据预处理**

   ```zsh
   python preprocess_md.py
   ```

3. **启动服务**

   ```zsh
   streamlit run app.py
   ```

## 依赖环境

- Python >= 3.8
- 主要依赖包：

  - streamlit
  - pymilvus
  - sentence-transformers
  - transformers
  - torch
  - accelerate
  - rank_bm25
  - scikit-learn
  - numpy
  - tqdm
  - beautifulsoup4
  - markdown

## 主要配置说明

- `config.py`：模型、检索、生成等参数配置
- `data/processed_data.json`：预处理后的医学知识库
- `data/id_map.json`：文档ID映射表

## 反馈与优化

- 支持用户对答案和文档相关性打分，系统自动记录并可自适应调整检索参数。

---
如有问题欢迎提交issue或联系作者。
