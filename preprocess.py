import os
import json
import re
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union
from bs4 import BeautifulSoup, Comment
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class ContentExtractor:
    """从 HTML 文件中提取内容的工具类。"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """清理文本内容。"""
        if not text:
            return ""
        patterns_to_remove = [
            r'微信号：[\w\-]+',
            r'公众号：[\w\-]+',
            r'作者：[\w\-]+',
            r'编辑：[\w\-]+',
            r'阅读原文',
            r'点击上方.*?蓝字',
            r'关注我们',
            r'扫描二维码',
            r'长按识别',
            r'返回顶部',
            r'参考文献[：:][^\n]*\n?',
            r'\[广告\]',
            r'\[[0-9-]+\]',
            r'ID[:：].*?\n',
            r'来源[:：].*?\n',
            r'https?://\S+',
        ]
        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        text = re.sub(r'([。！？；，.!?;,])\1+', r'\1', text)
        return text.strip()

    @staticmethod
    def generate_summary(content: str, max_length: int = 200) -> str:
        """通过 TF-IDF 权重和关键句提取生成摘要"""
        if not content:
            return ""
        sentences = re.split(r'([。!！？?])', content)
        sentences = [''.join(i) for i in zip(sentences[0::2], sentences[1::2] + [''])]
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return ""
        if len(sentences) == 1:
            return sentences[0]
        vectorizer = CountVectorizer(token_pattern=r'(?u)\b\w+\b')
        try:
            tf_matrix = vectorizer.fit_transform([s for s in sentences])
        except ValueError:
            return sentences[0]
        word_freq = np.asarray(tf_matrix.sum(axis=0)).ravel()
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            if len(sentence) < 5:
                score = 0
            else:
                words = vectorizer.get_feature_names_out()
                sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
                word_indices = [i for i, word in enumerate(words) if word in sentence_words]
                score = np.mean([word_freq[i] for i in word_indices]) if word_indices else 0
            position_weight = 1.5 if i < len(sentences) * 0.2 else 1.2 if i > len(sentences) * 0.8 else 1.0
            sentence_scores.append(score * position_weight)
        top_sentences = []
        total_length = 0
        for idx in np.argsort(sentence_scores)[::-1]:
            if total_length + len(sentences[idx]) <= max_length:
                top_sentences.append((idx, sentences[idx]))
                total_length += len(sentences[idx])
            else:
                break
        return ''.join(sent for _, sent in sorted(top_sentences, key=lambda x: x[0]))

    @staticmethod
    def clean_filename_for_summary(filename: str) -> str:
        """清理文件名，提取适合生成摘要的关键部分"""
        cleaned = re.sub(r'^\d{4}-\d{2}-\d{2}_', '', filename)
        cleaned = re.sub(r'ASH Poster[｜\s]*', '', cleaned)
        cleaned = re.sub(r'[_\|]', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    @staticmethod
    def extract_from_html(filepath: Path) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """从HTML文件中提取内容，失败时提取全文。"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f, 'lxml')
            title = None
            title_candidates = [
                soup.find('h1'),
                soup.find('title'),
                soup.find('meta', {'property': 'og:title'}),
                soup.find('meta', {'name': 'title'})
            ]
            for candidate in title_candidates:
                if candidate:
                    title = candidate.get_text().strip() if hasattr(candidate, 'get_text') else candidate.get('content', '').strip()
                    if title:
                        break
            if not title:
                title = ContentExtractor.clean_filename_for_summary(filepath.stem)
            content_selectors = [
                ('div', 'rich_media_content'),
                ('div', 'content'),
                ('article', None),
                ('div', 'post-content'),
                ('div', 'article-content'),
                ('main', None),
                ('section', None),
                ('div', 'entry-content'),
            ]
            content = None
            for tag, class_name in content_selectors:
                element = soup.find(tag, class_=class_name)
                if element:
                    for unwanted in element.find_all(['script', 'style', 'iframe', 'button', 'input', 'ins']):
                        unwanted.decompose()
                    for comment in element.find_all(text=lambda text: isinstance(text, Comment)):
                        comment.extract()
                    content = element.get_text(separator='\n', strip=True)
                    break
            if not content:
                # 提取全文
                for unwanted in soup.find_all(['script', 'style', 'iframe', 'button', 'input', 'ins']):
                    unwanted.decompose()
                for comment in soup.find_all(text=lambda text: isinstance(text, Comment)):
                    comment.extract()
                content = soup.get_text(separator='\n', strip=True)
            content = ContentExtractor.clean_text(content)
            summary = ContentExtractor.generate_summary(content) if content else ""
            return title, content, summary
        except Exception as e:
            print(f"Error processing HTML file {filepath}: {e}")
            return ContentExtractor.clean_filename_for_summary(filepath.stem), "", ""

class SemanticTextChunker:
    """基于语义的文本分块工具。"""
    
    def __init__(self, max_chunk_size: int = 512, min_chunk_size: int = 200):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        
    @staticmethod
    def is_sentence_complete(text: str) -> bool:
        """检查文本是否在完整句子处结束。"""
        if not text:
            return True
        return bool(re.search(r'[。！？.!?]$', text.strip()))

    def split_text(self, text: str) -> List[str]:
        """将文本分割成语义连贯的块。"""
        if not text:
            return [""]
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_length = 0
        for paragraph in paragraphs:
            sentences = re.split(r'([。！？.!?]+)', paragraph)
            sentences = [''.join(i) for i in zip(sentences[0::2], sentences[1::2] + [''])]
            sentences = [s.strip() for s in sentences if s.strip()]
            for sentence in sentences:
                sentence_length = len(sentence)
                if sentence_length > self.max_chunk_size:
                    if current_chunk:
                        chunks.append(''.join(current_chunk))
                        current_chunk = []
                        current_length = 0
                    sub_sentences = re.split(r'([，,；;、])', sentence)
                    sub_sentences = [''.join(i) for i in zip(sub_sentences[0::2], sub_sentences[1::2] + [''])]
                    current_sub_chunk = []
                    for sub_sent in sub_sentences:
                        if len(''.join(current_sub_chunk)) + len(sub_sent) <= self.max_chunk_size:
                            current_sub_chunk.append(sub_sent)
                        else:
                            if current_sub_chunk:
                                chunks.append(''.join(current_sub_chunk))
                            current_sub_chunk = [sub_sent]
                    if current_sub_chunk:
                        chunks.append(''.join(current_sub_chunk))
                    continue
                if current_length + sentence_length > self.max_chunk_size:
                    if current_chunk:
                        chunks.append(''.join(current_chunk))
                    current_chunk = [sentence]
                    current_length = sentence_length
                else:
                    current_chunk.append(sentence)
                    current_length += sentence_length
        if current_chunk:
            chunks.append(''.join(current_chunk))
        merged_chunks = []
        temp_chunk = []
        temp_length = 0
        for chunk in chunks:
            chunk_length = len(chunk)
            if temp_length + chunk_length <= self.max_chunk_size:
                temp_chunk.append(chunk)
                temp_length += chunk_length
            else:
                if temp_chunk:
                    merged_chunks.append(''.join(temp_chunk))
                temp_chunk = [chunk]
                temp_length = chunk_length
        if temp_chunk:
            merged_chunks.append(''.join(temp_chunk))
        return merged_chunks or [""]

class DocumentProcessor:
    """文档处理器，使用模型生成摘要。"""
    
    def __init__(self, input_dir: str, output_json: str, max_chunk_size: int = 512):
        self.input_dir = Path(input_dir)
        self.output_json = Path(output_json)
        self.chunker = SemanticTextChunker(max_chunk_size=max_chunk_size)
        self.extractor = ContentExtractor()
        print("正在加载文本生成模型...")
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        print("模型加载完成！")
    
    def generate_title_and_summary(self, content: str, filename: str = "") -> Tuple[str, str]:
        """使用模型生成标题和摘要，空内容时使用文件名作为后备"""
        input_text = content if content else ContentExtractor.clean_filename_for_summary(filename)
        if not input_text:
            input_text = "未知医学内容"
        
        title_prompt = f"""请为以下医学内容生成一个准确、专业的标题：

{input_text[:500]}...

标题："""
        summary_prompt = f"""请为以下医学内容生成一个简洁、专业的摘要，概述主要研究内容或意义：

{input_text[:1000]}

摘要："""
        try:
            title_output = self.pipeline(title_prompt, max_new_tokens=50)[0]['generated_text']
            title = title_output.split("标题：")[-1].strip()
            summary_output = self.pipeline(summary_prompt, max_new_tokens=150)[0]['generated_text']
            summary = summary_output.split("摘要：")[-1].strip()
            return title, summary
        except Exception as e:
            print(f"生成标题和摘要时出错: {e}")
            return input_text[:50], f"基于{input_text[:50]}的医学内容分析"

    def process_files(self) -> None:
        """处理所有HTML文件并生成JSON输出。"""
        print(f"开始处理 HTML 文件，目录：{self.input_dir}")
        self.output_json.parent.mkdir(parents=True, exist_ok=True)
        html_files = list(self.input_dir.glob('*.html'))
        total_files = len(html_files)
        print(f"找到 {total_files} 个 HTML 文件")
        all_data = []
        file_count = chunk_count = 0
        with tqdm(total=total_files, desc="处理文件") as pbar:
            for filepath in html_files:
                file_count += 1
                title, content, summary = self.extractor.extract_from_html(filepath)
                if content is None or not content.strip():
                    print(f"警告: 从 {filepath.name} 提取的内容为空，使用全文生成标题和摘要")
                title, summary = self.generate_title_and_summary(content, filepath.stem)
                chunks = self.chunker.split_text(content)
                for i, chunk in enumerate(chunks):
                    doc_data = {
                        "id": f"{filepath.stem}_{i}",
                        "title": title,
                        "abstract": summary if i == 0 else self.generate_title_and_summary(chunk, filepath.stem)[1],
                        "content": chunk,
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    }
                    all_data.append(doc_data)
                    chunk_count += 1
                pbar.update(1)
        print(f"\n处理完成。共处理 {file_count} 个文件，生成 {chunk_count} 个数据块。")
        try:
            with self.output_json.open('w', encoding='utf-8') as f:
                json.dump(all_data, f, ensure_ascii=False, indent=2)
            print(f"结果已保存到 {self.output_json}")
        except Exception as e:
            print(f"保存 JSON 文件时出错 {self.output_json}: {e}")

# Configuration
INPUT_DIR = '/Users/lujiayi/Downloads/leuka_data/html'
OUTPUT_JSON = './data/processed_data.json'
MAX_CHUNK_SIZE = 512

# Run
if __name__ == "__main__":
    processor = DocumentProcessor(INPUT_DIR, OUTPUT_JSON, MAX_CHUNK_SIZE)
    processor.process_files()