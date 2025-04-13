import pandas as pd
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration, M2M100ForConditionalGeneration, M2M100Tokenizer
from tqdm import tqdm
import numpy as np
import json
from datetime import datetime
import gc
from zhconv import convert
import requests
from difflib import SequenceMatcher
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu
import jieba
import time

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
        
    def clear_gpu_memory(self):
        """Clear GPU memory safely"""
        if torch.cuda.is_available():
            # Ensure that all GPU operations are completed
            torch.cuda.synchronize()
            # Clear cache
            torch.cuda.empty_cache()
            gc.collect()
            # Print the status after cleaning
            print("\nGPU Memory after clearing:")
            for i in range(torch.cuda.device_count()):
                used_mem = torch.cuda.memory_allocated(i) / 1024**3
                print(f"GPU {i} Used Memory: {used_mem:.2f}GB")
            
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
        print("\n=== Starting T5 Model Evaluation ===")
        
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
        
        for idx, row in tqdm(sample_data.iterrows(), total=sample_size, desc="T5 Progress"):
            question = row['question']
            actual_answer = row['answer']
            
            try:
                # Translate question to English
                en_question = self.translate_zh_to_en(question)
                
                # Generate English answer
                inputs = t5_tokenizer(
                    f"Answer this question: {en_question}",
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
                
                results.append({
                    "question": question,
                    "actual_answer": actual_answer,
                    "predicted_answer": predicted_answer,
                    "en_question": en_question,
                    "en_answer": en_predicted_answer,
                    "accuracy": accuracy
                })
                
            except Exception as e:
                print(f"Error processing question: {str(e)}")
                continue
                
            finally:
                # Clear GPU memory
                self.clear_gpu_memory()
        
        # Clean up models
        del t5_model
        del t5_tokenizer
        self.clear_gpu_memory()
        
        return results
    
    def evaluate_mistral(self, sample_size=100):
        """Evaluate Mistral model using Ollama"""
        print("\n=== Starting Mistral Model Evaluation ===")
        
        # Random sampling
        sample_data = self.df.sample(n=sample_size, random_state=42)
        results = []
        
        for idx, row in tqdm(sample_data.iterrows(), total=sample_size, desc="Mistral Progress"):
            question = row['question']
            actual_answer = row['answer']
            
            try:
                # Generate using Ollama API
                prompt = f"请用中文回答下面的问题，要简洁准确。\n\n问题：{question}"
                
                response = requests.post('http://localhost:11434/api/generate', 
                    json={
                        "model": "mistral",
                        "prompt": prompt,
                        "stream": False,
                        "temperature": 0.7,
                        "top_p": 0.95
                    })
                
                if response.status_code == 200:
                    predicted_answer = response.json()['response'].strip()
                    
                    # Calculate accuracy
                    accuracy = self.calculate_accuracy(predicted_answer, actual_answer)
                    
                    results.append({
                        "question": question,
                        "actual_answer": actual_answer,
                        "predicted_answer": predicted_answer,
                        "accuracy": accuracy
                    })
                    
                    # Print progress every 10 questions
                    if idx % 10 == 0:
                        print(f"\nProcessed {idx} questions")
                        print(f"Last question: {question}")
                        print(f"Predicted answer: {predicted_answer}")
                        print(f"Actual answer: {actual_answer}")
                        print(f"Accuracy: {accuracy:.4f}")
            
            except Exception as e:
                print(f"Error processing question {idx}: {str(e)}")
                continue
        
        return results
        
    def evaluate_qwen(self, sample_size=100):
        """Evaluate Qwen-7B-Chat model"""
        print("\n=== Starting Qwen-7B-Chat Model Evaluation ===")
        
        # Random sampling
        sample_data = self.df.sample(n=sample_size, random_state=42)
        results = []
        
        for idx, row in tqdm(sample_data.iterrows(), total=sample_size, desc="Qwen Evaluation Progress"):
            if idx > 0 and idx % 10 == 0:  # Pause every 10 questions
                print(f"\nProcessed {idx} questions")
                time.sleep(10)  # Add delay
            
            question = row['question']
            actual_answer = row['answer']
            
            try:
                prompt = f"""
请根据问题给出简短的中文答案。

问题：{question}

要求：
1. 只给出答案，不要解释
2. 如果不确定，直接回答"不知道"
"""
                
                response = requests.post('http://localhost:11434/api/generate', 
                    json={
                        "model": "qwen:7b-chat",  # Switch to qwen model
                        "prompt": prompt,
                        "stream": False,
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "num_predict": 50
                    })
                
                time.sleep(2)  # Add request interval
                
                if response.status_code == 200:
                    predicted_answer = response.json()['response'].strip()
                    predicted_answer = predicted_answer.replace("", "").strip()
                    predicted_answer = predicted_answer.split("\n")[0].strip()
                    
                    accuracy = self.calculate_accuracy(predicted_answer, actual_answer)
                    
                    if idx % 10 == 0:
                        print(f"\nQuestion: {question}")
                        print(f"Predicted Answer: {predicted_answer}")
                        print(f"Actual Answer: {actual_answer}")
                        print(f"Accuracy: {accuracy:.4f}")
                    
                    results.append({
                        "question": question,
                        "actual_answer": actual_answer,
                        "predicted_answer": predicted_answer,
                        "accuracy": accuracy
                    })
            
            except Exception as e:
                print(f"Error processing question {idx}: {str(e)}")
                continue
            
            time.sleep(2)  # Brief pause after each question
        
        return results
        
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
    with open(f"t5_results_norag_{timestamp}.json", 'w', encoding='utf-8') as f:
        json.dump({
            "results": t5_results,
            "average_accuracy": float(t5_avg_accuracy)
        }, f, ensure_ascii=False, indent=2)
    
    # Evaluate Mistral model
    mistral_results = evaluator.evaluate_mistral(sample_size=100)
    mistral_avg_accuracy = np.mean([r["accuracy"] for r in mistral_results])
    print(f"\nMistral Average Accuracy: {mistral_avg_accuracy:.4f}")
    
    # Save Mistral results
    with open(f"mistral_results_norag_{timestamp}.json", 'w', encoding='utf-8') as f:
        json.dump({
            "results": mistral_results,
            "average_accuracy": float(mistral_avg_accuracy)
        }, f, ensure_ascii=False, indent=2)
        
    # Clear GPU memory
    evaluator.clear_gpu_memory()
    print("\nClearing GPU memory, preparing for Qwen evaluation...")
    time.sleep(5)  # Wait for GPU resources to be released
    
    # Evaluate Qwen model
    qwen_results = evaluator.evaluate_qwen(sample_size=100)
    qwen_avg_accuracy = np.mean([r["accuracy"] for r in qwen_results])
    print(f"\nQwen Average Accuracy: {qwen_avg_accuracy:.4f}")
    
    # Save Qwen results
    with open(f"qwen_results_norag_{timestamp}.json", 'w', encoding='utf-8') as f:
        json.dump({
            "results": qwen_results,
            "average_accuracy": float(qwen_avg_accuracy)
        }, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main() 