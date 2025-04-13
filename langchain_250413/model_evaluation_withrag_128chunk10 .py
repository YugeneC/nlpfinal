import pandas as pd
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration, M2M100ForConditionalGeneration, M2M100Tokenizer
from tqdm import tqdm
import numpy as np
import json
from datetime import datetime
import gc
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import requests
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu
import jieba
from difflib import SequenceMatcher
from src.text_splitter import ChineseTextSplitter
import time

RELEVANCE_THRESHOLD = 0.5  
# Chunking Configuration
CHUNK_SIZE = 128  # Token size for each chunk
CHUNK_OVERLAP = 13  # 10% overlap (128 * 0.1)

class ModelEvaluator:
    def __init__(self):
        # Check if GPU is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load data
        print("Loading data...")
        self.df = pd.read_csv("chinese_simpleqa.csv")
        print(f"Loaded {len(self.df)} QA pairs")
        
        # Load translation model
        print("Loading translation model...")
        self.translator_model = M2M100ForConditionalGeneration.from_pretrained(
            "facebook/m2m100_418M",
            cache_dir="F:/huggingface_models"
        ).to(self.device)
        self.translator_tokenizer = M2M100Tokenizer.from_pretrained(
            "facebook/m2m100_418M",
            cache_dir="F:/huggingface_models"
        )
        
        # Initialize RAG components
        print("Initializing RAG system...")
        self.setup_rag()
        
    def setup_rag(self):
        """Set up RAG system with 256 token chunk size and 10% overlap"""
        # Initialize embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': self.device},
            cache_folder="F:/huggingface_models"
        )
        
        # Use configured chunk size
        text_splitter = ChineseTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        
        # Store QA pairs as knowledge base
        documents = []
        for _, row in self.df.iterrows():
            # Combine question and answer
            text = f"问题：{row['question']}\n答案：{row['answer']}"
            chunks = text_splitter.split_text(text)
            documents.extend([{
                "content": chunk,
                "metadata": {
                    "source": "chinese_simpleqa",
                    "question": row['question'],
                    "answer": row['answer']
                }
            } for chunk in chunks])
        
        # Create vector store
        self.vectorstore = FAISS.from_texts(
            [doc["content"] for doc in documents],
            self.embeddings,
            metadatas=[doc["metadata"] for doc in documents]
        )
        print(f"RAG system initialization complete! Processed {len(documents)} text chunks")
        
    def clear_gpu_memory(self):
        """Clear GPU memory"""
        if torch.cuda.is_available():
            # Ensure all GPU operations are completed
            torch.cuda.synchronize()
            # Clear cache
            torch.cuda.empty_cache()
            gc.collect()
            # Print cleanup status
            print("\nGPU Memory after clearing:")
            for i in range(torch.cuda.device_count()):
                used_mem = torch.cuda.memory_allocated(i) / 1024**3
                print(f"GPU {i} Used Memory: {used_mem:.2f}GB")
            
            # Consider adding special handling for Ollama
            try:
                # Send a simple request to reset Ollama's state
                requests.post('http://localhost:11434/api/generate', 
                    json={
                        "model": "none",
                        "prompt": "",
                        "stream": False
                    })
            except:
                pass
    
    def get_relevant_context(self, question, k=5):
        """获取相关上下文
        Args:
            question: 问题
            k: 返回的相关文档数量
        """
        docs_and_scores = self.vectorstore.similarity_search_with_score(question, k=k)
        
        # 使用较低的相似度阈值 (0.2 或 20%)
        relevant_docs = [doc for doc, score in docs_and_scores if score >= 0.2]
        
        if not relevant_docs:
            # 如果没有文档超过阈值，至少返回最相关的一个文档
            relevant_docs = [docs_and_scores[0][0]] if docs_and_scores else []
        
        context = "\n".join([doc.page_content for doc in relevant_docs])
        return context if context else "没有找到相关信息"
    
    def translate_zh_to_en(self, text):
        """Translate Chinese to English"""
        self.translator_tokenizer.src_lang = "zh"
        encoded = self.translator_tokenizer(text, return_tensors="pt").to(self.device)
        generated_tokens = self.translator_model.generate(
            **encoded,
            forced_bos_token_id=self.translator_tokenizer.get_lang_id("en"),
            max_length=128
        )
        return self.translator_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    
    def translate_en_to_zh(self, text):
        """Translate English to Chinese"""
        self.translator_tokenizer.src_lang = "en"
        encoded = self.translator_tokenizer(text, return_tensors="pt").to(self.device)
        generated_tokens = self.translator_model.generate(
            **encoded,
            forced_bos_token_id=self.translator_tokenizer.get_lang_id("zh"),
            max_length=128
        )
        return self.translator_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    
    def evaluate_t5(self, sample_size=100):
        """Evaluate T5 model"""
        print("\n=== Starting T5 Model Evaluation (with RAG) ===")
        
        # Load T5 model
        print("Loading T5 model...")
        t5_model = T5ForConditionalGeneration.from_pretrained(
            "google/flan-t5-base",
            cache_dir="F:/huggingface_models"
        ).to(self.device)
        t5_tokenizer = AutoTokenizer.from_pretrained(
            "google/flan-t5-base",
            cache_dir="F:/huggingface_models"
        )
        
        # Random sampling
        sample_data = self.df.sample(n=sample_size, random_state=42)
        results = []
        
        for idx, row in tqdm(sample_data.iterrows(), total=sample_size, desc="T5 Evaluation Progress"):
            question = row['question']
            actual_answer = row['answer']
            
            try:
                # Get relevant context
                context = self.get_relevant_context(question)
                
                # Translate question and context to English
                en_context = self.translate_zh_to_en(context)
                en_question = self.translate_zh_to_en(question)
                
                # Build English prompt
                input_text = f"""
Based on the following reference information:

Reference:
{en_context}

Current Question: {en_question}

Requirements:
1. Use the reference information to answer
2. Only provide the answer, no explanation
3. If no relevant information found, answer "unknown"
"""
                
                # Generate English answer
                inputs = t5_tokenizer(
                    input_text,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True
                ).to(self.device)
                outputs = t5_model.generate(**inputs, max_length=256)
                en_predicted_answer = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Translate answer back to Chinese
                predicted_answer = self.translate_en_to_zh(en_predicted_answer)
                
                # Calculate accuracy
                accuracy = self.calculate_accuracy(predicted_answer, actual_answer)
                
                # Display details every 10 questions
                if idx % 10 == 0:
                    print(f"\nProcessed question {idx}")
                    print(f"Question: {question}")
                    print(f"English Question: {en_question}")
                    print(f"English Answer: {en_predicted_answer}")
                    print(f"Predicted Answer: {predicted_answer}")
                    print(f"Actual Answer: {actual_answer}")
                    print(f"Accuracy: {accuracy:.4f}")
                
                results.append({
                    "question": question,
                    "actual_answer": actual_answer,
                    "predicted_answer": predicted_answer,
                    "en_question": en_question,
                    "en_answer": en_predicted_answer,
                    "context": context,
                    "accuracy": accuracy
                })
                
            except Exception as e:
                print(f"Error processing question: {str(e)}")
                continue
                
            finally:
                # Clear memory
                self.clear_gpu_memory()
        
        # Cleanup models
        del t5_model
        del t5_tokenizer
        self.clear_gpu_memory()
        
        return results
    
    def evaluate_mistral(self, sample_size=100):
        """Evaluate Mistral model with error handling and statistics"""
        sample_data = self.df.sample(n=sample_size, random_state=42)
        results = []
        error_cases = []  # 记录错误案例
        processed_count = 0  # 成功处理的数量
        
        for idx, row in tqdm(sample_data.iterrows(), total=sample_size):
            question = row['question']
            actual_answer = row['answer']
            
            try:
                context = self.get_relevant_context(
                    question=row['question'],
                    k=5
                )
                
                prompt = f"""
请根据以下参考信息回答问题。

参考信息：
{context}

当前问题：{question}

要求：
1. 参考上述信息回答问题
2. 只给出答案，不要解释
3. 如果参考信息中没有相关内容，直接回答"不知道"
"""
                response = requests.post('http://localhost:11434/api/generate', 
                    json={
                        "model": "mistral",
                        "prompt": prompt,
                        "stream": False,
                        "temperature": 0.1,
                        "top_p": 0.8,
                        "num_predict": 50
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    predicted_answer = response.json()['response'].strip()
                    accuracy = self.calculate_accuracy(predicted_answer, actual_answer)
                    
                    results.append({
                        "question": question,
                        "actual_answer": actual_answer,
                        "predicted_answer": predicted_answer,
                        "accuracy": accuracy,
                        "status": "success"
                    })
                    processed_count += 1
                    
                else:
                    error_cases.append({
                        "question": question,
                        "error_type": "api_error",
                        "status_code": response.status_code,
                        "index": idx
                    })
                    
            except Exception as e:
                error_cases.append({
                    "question": question,
                    "error_type": "processing_error",
                    "error_message": str(e),
                    "index": idx
                })
                print(f"Error processing question {idx}: {str(e)}")
                continue
                
            time.sleep(5)  # 请求间隔
        
        # 计算统计信息
        evaluation_stats = {
            "total_samples": sample_size,
            "successful_evaluations": processed_count,
            "failed_evaluations": len(error_cases),
            "success_rate": processed_count / sample_size * 100,
            "average_accuracy": np.mean([r["accuracy"] for r in results]) if results else 0,
            "error_summary": {
                "api_errors": len([e for e in error_cases if e["error_type"] == "api_error"]),
                "processing_errors": len([e for e in error_cases if e["error_type"] == "processing_error"])
            }
        }
        
        # 保存结果和错误信息
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"mistral_results_{timestamp}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                "results": results,
                "error_cases": error_cases,
                "statistics": evaluation_stats
            }, f, ensure_ascii=False, indent=2)
        
        # 打印评估报告
        print("\n=== Evaluation Report ===")
        print(f"Total samples: {sample_size}")
        print(f"Successful evaluations: {processed_count}")
        print(f"Failed evaluations: {len(error_cases)}")
        print(f"Success rate: {evaluation_stats['success_rate']:.2f}%")
        print(f"Average accuracy: {evaluation_stats['average_accuracy']:.4f}")
        print("\nError Summary:")
        print(f"API errors: {evaluation_stats['error_summary']['api_errors']}")
        print(f"Processing errors: {evaluation_stats['error_summary']['processing_errors']}")
        
        return results, error_cases, evaluation_stats
        
    def evaluate_qwen(self, sample_size=100):
        """评估Qwen模型"""
        print("\n=== 开始评估Qwen模型（带RAG）===")
        
        sample_data = self.df.sample(n=sample_size, random_state=42)
        results = []
        error_cases = []
        processed_count = 0
        failed_cases = []  # 新增：记录@@@@@类型的失败案例
        
        for idx, row in tqdm(sample_data.iterrows(), total=sample_size, desc="Qwen评估进度"):
            try:
                question = row['question']
                actual_answer = row['answer']
                
                # 获取相关上下文
                context = self.get_relevant_context(question, k=2)
                
                prompt = f"""
请根据以下参考信息回答问题。

参考信息：
{context}

当前问题：{question}

要求：
1. 必须基于参考信息中的内容回答
2. 只给出答案，不要解释
3. 如果参考信息中没有相关内容，直接回答"不知道"
"""
                
                response = requests.post('http://localhost:11434/api/generate', 
                    json={
                        "model": "qwen:7b-chat",
                        "prompt": prompt,
                        "stream": False,
                        "temperature": 0.1,
                        "top_p": 0.8,
                        "num_predict": 50
                    },
                    timeout=60
                )
                
                if response.status_code == 200:
                    predicted_answer = response.json()['response'].strip()
                    predicted_answer = predicted_answer.replace("答案：", "").strip()
                    predicted_answer = predicted_answer.split("\n")[0].strip()
                    
                    # 检查是否为@@@@@类型的答案
                    if "@" in predicted_answer:
                        failed_cases.append({
                            "question": question,
                            "actual_answer": actual_answer,
                            "predicted_answer": predicted_answer,
                            "context": context,
                            "failure_type": "placeholder_answer",
                            "index": idx
                        })
                        accuracy = 0.0
                        status = "failed"
                    else:
                        accuracy = self.calculate_accuracy(predicted_answer, actual_answer)
                        status = "success"
                    
                    results.append({
                        "question": question,
                        "actual_answer": actual_answer,
                        "predicted_answer": predicted_answer,
                        "context": context,
                        "accuracy": accuracy,
                        "status": status
                    })
                    processed_count += 1
                    
                    if idx % 10 == 0:
                        print(f"\n处理第 {idx} 个问题")
                        print(f"问题: {question}")
                        print(f"上下文: {context}")
                        print(f"预测答案: {predicted_answer}")
                        print(f"实际答案: {actual_answer}")
                        print(f"状态: {status}")
                        print(f"准确度: {accuracy:.4f}")
                
                else:
                    print(f"API Error: Status code {response.status_code}")
                    print(f"Response: {response.text}")
                    error_cases.append({
                        "question": question,
                        "error_type": "api_error",
                        "status_code": response.status_code,
                        "index": idx
                    })
                
            except requests.exceptions.RequestException as e:
                print(f"Request Error: {str(e)}")
                error_cases.append({
                    "question": question,
                    "error_type": "request_error",
                    "error_message": str(e),
                    "index": idx
                })
                continue
            
            time.sleep(5)
        
        # 计算统计信息（排除@@@@@类型的答案）
        valid_results = [r for r in results if r["status"] == "success"]
        
        evaluation_stats = {
            "total_samples": sample_size,
            "successful_evaluations": len(valid_results),
            "failed_evaluations": len(failed_cases),
            "error_evaluations": len(error_cases),
            "success_rate": len(valid_results) / sample_size * 100,
            "average_accuracy": np.mean([r["accuracy"] for r in valid_results]) if valid_results else 0,
            "error_summary": {
                "api_errors": len([e for e in error_cases if e["error_type"] == "api_error"]),
                "processing_errors": len([e for e in error_cases if e["error_type"] == "processing_error"]),
                "placeholder_answers": len(failed_cases)
            }
        }
        
        # 打印评估报告
        print("\n=== Qwen评估报告 ===")
        print(f"总样本数: {sample_size}")
        print(f"有效评估数: {len(valid_results)}")
        print(f"占位符答案数(@@@@@): {len(failed_cases)}")
        print(f"错误案例数: {len(error_cases)}")
        print(f"有效评估准确率: {evaluation_stats['average_accuracy']:.4f}")
        print(f"有效评估比例: {evaluation_stats['success_rate']:.2f}%")
        
        return results, failed_cases, error_cases, evaluation_stats
        
    def calculate_accuracy(self, predicted, actual):
        """Calculate answer accuracy using multiple evaluation methods"""
        # 4. First check containment (highest priority)
        if actual.strip() in predicted.strip():
            return 1.0
        
        # If no containment, calculate average of other metrics
        # 1. Calculate similarity using SequenceMatcher
        sequence_score = SequenceMatcher(None, predicted, actual).ratio()
        
        # 2. Calculate BLEU score
        try:
            predicted_tokens = list(jieba.cut(predicted))
            actual_tokens = list(jieba.cut(actual))
            bleu_score = sentence_bleu([actual_tokens], predicted_tokens)
        except:
            bleu_score = 0
        
        # 3. Calculate ROUGE score
        try:
            rouge = Rouge()
            rouge_scores = rouge.get_scores(' '.join(predicted_tokens), ' '.join(actual_tokens))
            rouge_l = rouge_scores[0]["rouge-l"]["f"]
        except:
            rouge_l = 0
        
        # Calculate average of the three metrics
        average_score = (sequence_score + bleu_score + rouge_l) / 3
        
        # Print scores for debugging (10% of cases)
        if random.random() < 0.1:
            print("\nEvaluation Details:")
            print(f"SequenceMatcher Score: {sequence_score:.4f}")
            print(f"BLEU Score: {bleu_score:.4f}")
            print(f"ROUGE-L Score: {rouge_l:.4f}")
            print(f"Average Score: {average_score:.4f}")
        
        return average_score

def main():
    evaluator = ModelEvaluator()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Evaluate T5 model
    t5_results = evaluator.evaluate_t5(sample_size=100)
    t5_avg_accuracy = np.mean([r["accuracy"] for r in t5_results])
    print(f"\nT5 Average Accuracy: {t5_avg_accuracy:.4f}")
    
    # Save T5 results
    with open(f"t5_results_withragg_128chunk10_{timestamp}.json", 'w', encoding='utf-8') as f:
        json.dump({
            "results": t5_results,
            "average_accuracy": float(t5_avg_accuracy)
        }, f, ensure_ascii=False, indent=2)
    
    # Clear GPU memory
    evaluator.clear_gpu_memory()
    print("\nClearing GPU memory, preparing for Mistral evaluation...")
    time.sleep(5)  # Wait for GPU resources to be released
    
    # Evaluate Mistral model
    mistral_results, error_cases, evaluation_stats = evaluator.evaluate_mistral(sample_size=100)
    mistral_avg_accuracy = np.mean([r["accuracy"] for r in mistral_results])
    print(f"\nMistral Average Accuracy: {mistral_avg_accuracy:.4f}")
    
    # Save Mistral results
    with open(f"mistral_results_withragg_128chunk10_{timestamp}.json", 'w', encoding='utf-8') as f:
        json.dump({
            "results": mistral_results,
            "error_cases": error_cases,
            "statistics": evaluation_stats
        }, f, ensure_ascii=False, indent=2)
    
    # Clear GPU memory
    evaluator.clear_gpu_memory()
    print("\nClearing GPU memory, preparing for Qwen evaluation...")
    time.sleep(5)  # Wait for GPU resources to be released
    
    # Evaluate Qwen model
    qwen_results, qwen_failed_cases, qwen_error_cases, qwen_evaluation_stats = evaluator.evaluate_qwen(sample_size=100)
    
    # Save Qwen results
    with open(f"qwen_results_withrag_128chunk10_{timestamp}.json", 'w', encoding='utf-8') as f:
        json.dump({
            "results": qwen_results,
            "failed_cases": qwen_failed_cases,
            "error_cases": qwen_error_cases,
            "statistics": qwen_evaluation_stats
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\nQwen有效评估准确率: {qwen_evaluation_stats['average_accuracy']:.4f}")
    print(f"Qwen占位符答案比例: {len(qwen_failed_cases)/100:.2%}")
    
    # Final cleanup
    evaluator.clear_gpu_memory()

if __name__ == "__main__":
    main() 