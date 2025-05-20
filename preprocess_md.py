import os
import json
import re
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from bs4 import BeautifulSoup
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from markdown import markdown
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ContentExtractor:
    """从 Markdown 文件中提取内容的工具类。"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """清理文本内容，移除 Markdown 标记、链接、网址、时间戳、声明等无关内容。"""
        if not text:
            return ""
        patterns_to_remove = [
            r'\[.*?\]\(.*?\)',                  # 移除 Markdown 链接，如 [text](url)
            r'https?://\S+',                   # 移除 http(s) 网址
            r'www\.\S+',                       # 移除 www 开头的网址
            r'\b\S+\.(com|org|net|gov|edu|cn|io)\b',  # 移除常见域名
            r'```.*?```',                      # 移除代码块
            r'`.*?`',                          # 移除行内代码
            r'!\[.*?\]\(.*?\)',                # 移除图片
            r'^\s*[-*+]\s+',                   # 移除列表符号
            r'^#{1,6}\s+',                     # 移除标题符号
            r'^\s*>\s+',                       # 移除引用符号
            r'---+|===+',                      # 移除分隔线
            r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}',  # 移除时间戳，如 2020-07-15 16:07:27
            r'陆道培医疗团队',                  # 移除团队名称
            r'申明：.*?侵犯。',                 # 移除声明
            r'了解更多内容请登陆：.*',          # 移除登陆提示
            r'\s+',                            # 规范化空格
            r'\n\s*\n+',                       # 规范化换行
        ]
        for pattern in patterns_to_remove:
            text = re.sub(pattern, ' ', text, flags=re.DOTALL)
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
            return sentences[0] if sentences else ""
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
    def simple_summary(content: str, max_length: int = 200) -> str:
        """快速生成摘要：提取前几句"""
        if not content:
            return ""
        sentences = re.split(r'([。!！？?])', content)
        sentences = [''.join(i) for i in zip(sentences[0::2], sentences[1::2] + [''])]
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return ""
        summary = ""
        total_length = 0
        for sentence in sentences[:3]:
            if total_length + len(sentence) <= max_length:
                summary += sentence
                total_length += len(sentence)
            else:
                break
        return summary

    @staticmethod
    def clean_filename_for_title(filename: str) -> str:
        """清理文件名，提取适合标题的部分"""
        cleaned = re.sub(r'^\d{4}-\d{2}-\d{2}_', '', filename)
        cleaned = re.sub(r'[_\|]', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    @staticmethod
    def extract_from_markdown(filepath: Path) -> Tuple[str, str, str]:
        """从 Markdown 文件中提取标题、正文和摘要"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                md_content = f.read()
            html_content = markdown(md_content, extensions=['extra', 'meta'])
            soup = BeautifulSoup(html_content, 'lxml')
            title = None
            title_match = re.search(r'^#\s+(.+?)\n', md_content, re.MULTILINE)
            if title_match:
                title = ContentExtractor.clean_text(title_match.group(1).strip())
            else:
                title_elem = soup.find('h1')
                title = ContentExtractor.clean_text(title_elem.get_text().strip()) if title_elem else ContentExtractor.clean_filename_for_title(filepath.stem)
            content = soup.get_text(separator='\n', strip=True)
            content = ContentExtractor.clean_text(content)
            if not content.strip():
                logging.warning(f"从 {filepath.name} 提取的内容为空，使用文件名作为默认内容")
                content = ContentExtractor.clean_filename_for_title(filepath.stem)
            summary = ContentExtractor.generate_summary(content)
            if not summary:
                summary = ContentExtractor.simple_summary(content)
            return title, content, summary
        except Exception as e:
            logging.error(f"处理 Markdown 文件 {filepath} 时出错: {e}")
            title = ContentExtractor.clean_filename_for_title(filepath.stem)
            content = title
            summary = ContentExtractor.simple_summary(content)
            return title, content, summary

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
    """文档处理器，处理 Markdown 文件。"""
    
    def __init__(self, input_dir: str, output_json: str, max_chunk_size: int = 512):
        self.input_dir = Path(input_dir)
        self.output_json = Path(output_json)
        self.chunker = SemanticTextChunker(max_chunk_size=max_chunk_size)
        self.extractor = ContentExtractor()
    
    def process_file(self, filepath: Path) -> List[Dict]:
        """处理单个 Markdown 文件"""
        title, content, _ = self.extractor.extract_from_markdown(filepath)
        chunks = self.chunker.split_text(content)
        results = []
        for i, chunk in enumerate(chunks):
            summary = ContentExtractor.generate_summary(chunk)
            if not summary:
                summary = ContentExtractor.simple_summary(chunk)
            doc_data = {
                "id": f"{filepath.stem}_{i}",
                "title": title,
                "abstract": summary,
                "content": chunk,
                "chunk_index": i,
                "total_chunks": len(chunks)
            }
            results.append(doc_data)
        return results

    def process_files(self) -> None:
        """处理所有 Markdown 文件并生成 JSON 输出，使用多线程加速"""
        logging.info(f"开始处理 Markdown 文件，目录：{self.input_dir}")
        self.output_json.parent.mkdir(parents=True, exist_ok=True)
        md_files = list(self.input_dir.glob('*.md'))
        total_files = len(md_files)
        logging.info(f"找到 {total_files} 个 Markdown 文件")
        all_data = []
        file_count = chunk_count = 0
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            future_to_file = {executor.submit(self.process_file, filepath): filepath for filepath in md_files}
            with tqdm(total=total_files, desc="处理文件") as pbar:
                for future in as_completed(future_to_file):
                    filepath = future_to_file[future]
                    try:
                        results = future.result()
                        all_data.extend(results)
                        file_count += 1
                        chunk_count += len(results)
                    except Exception as e:
                        logging.error(f"处理文件 {filepath} 时出错: {e}")
                    pbar.update(1)
        logging.info(f"\n处理完成。共处理 {file_count} 个文件，生成 {chunk_count} 个数据块。")
        try:
            with self.output_json.open('w', encoding='utf-8') as f:
                json.dump(all_data, f, ensure_ascii=False, indent=2)
            logging.info(f"结果已保存到 {self.output_json}")
        except Exception as e:
            logging.error(f"保存 JSON 文件时出错 {self.output_json}: {e}")

# Configuration
INPUT_DIR = '/Users/lujiayi/Downloads/leuka_data/md'
OUTPUT_JSON = './data/processed_data.json'
MAX_CHUNK_SIZE = 512

# Run
if __name__ == "__main__":
    processor = DocumentProcessor(INPUT_DIR, OUTPUT_JSON, MAX_CHUNK_SIZE)
    processor.process_files()