from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from src.config import CHUNK_SIZE, CHUNK_OVERLAP
import re
from typing import List
import jieba

class TextSplitter:
    """
    Handles text splitting with specified chunking strategy
    使用指定的分块策略处理文本分割
    """
    
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            is_separator_regex=False
        )
    
    def split_documents(self, documents):
        """
        Split documents into chunks
        将文档分割成块
        
        Args:
            documents: Documents to split
                      要分割的文档
        
        Returns:
            List of document chunks
            文档块列表
        """
        return self.splitter.split_documents(documents)

class ChineseTextSplitter(CharacterTextSplitter):
    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 26, **kwargs):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        self._chunk_size = chunk_size        # 修改这里，使用_chunk_size
        self._chunk_overlap = chunk_overlap  # 修改这里，使用_chunk_overlap
    
    def split_text(self, text: str) -> List[str]:
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'\s+', ' ', text)
        
        sentences = []
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
            sentences.extend(re.split('[。！？!?]', line))
        
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # 修改这里，使用_chunk_size
            if current_length + sentence_length > self._chunk_size:
                if current_chunk:
                    chunks.append("".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append("".join(current_chunk))
        
        # 修改这里，使用_chunk_overlap
        if self._chunk_overlap > 0 and len(chunks) > 1:
            overlapped_chunks = []
            for i in range(len(chunks)):
                if i == 0:
                    overlapped_chunks.append(chunks[i])
                else:
                    previous_chunk = chunks[i-1]
                    current_chunk = chunks[i]
                    
                    # 获取前一个块的末尾部分
                    overlap_start = len(previous_chunk) - self._chunk_overlap
                    if overlap_start > 0:
                        overlap_text = previous_chunk[overlap_start:]
                        overlapped_chunks.append(overlap_text + current_chunk)
                    else:
                        overlapped_chunks.append(current_chunk)
            
            chunks = overlapped_chunks
        
        return chunks

