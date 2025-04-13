import json
import jieba
from difflib import SequenceMatcher
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
from datetime import datetime
import os

class QwenEvaluator:
    def __init__(self):
        self.rouge = Rouge()
        
    def calculate_accuracy(self, predicted, actual):
        """Calculate answer accuracy using multiple evaluation methods"""
        if not predicted or not actual:
            return 0.0
            
        # Remove any formatting
        predicted = predicted.strip()
        actual = actual.strip()
        
        # Check for placeholder answers
        if "@" in predicted:
            return 0.0
            
        # Check for "不知道" response
        if predicted == "不知道":
            return 0.0
            
        # First check containment (highest priority)
        if actual in predicted:
            return 1.0
        
        try:
            # Calculate similarity using SequenceMatcher
            sequence_score = SequenceMatcher(None, predicted, actual).ratio()
            
            # Calculate BLEU score
            predicted_tokens = list(jieba.cut(predicted))
            actual_tokens = list(jieba.cut(actual))
            bleu_score = sentence_bleu([actual_tokens], predicted_tokens)
            
            # Calculate ROUGE score
            rouge_scores = self.rouge.get_scores(
                ' '.join(predicted_tokens), 
                ' '.join(actual_tokens)
            )
            rouge_l = rouge_scores[0]["rouge-l"]["f"]
            
            # Calculate average of the three metrics
            average_score = (sequence_score + bleu_score + rouge_l) / 3
            
            return average_score
            
        except Exception as e:
            print(f"Error calculating accuracy: {str(e)}")
            return 0.0

    def evaluate_results(self, json_file):
        """Evaluate Qwen results from JSON file"""
        try:
            # Read the JSON file
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            results = data.get('results', [])
            total_samples = len(results)
            valid_results = []
            failed_cases = []
            error_cases = []
            
            # Process each result
            for idx, result in enumerate(results, 1):
                predicted = result.get('predicted_answer', '')
                actual = result.get('actual_answer', '')
                
                # Skip empty answers
                if not predicted or not actual:
                    error_cases.append(result)
                    continue
                
                # Check for placeholder answers
                if "@" in predicted:
                    failed_cases.append(result)
                    result['status'] = 'failed'
                    result['accuracy'] = 0.0
                else:
                    # Calculate new accuracy
                    accuracy = self.calculate_accuracy(predicted, actual)
                    result['accuracy'] = accuracy
                    result['status'] = 'success'
                    if accuracy > 0:  # Only count as valid if accuracy is greater than 0
                        valid_results.append(result)
                
                # Print progress every 10 questions
                if idx % 10 == 0:
                    print(f"已处理 {idx}/{total_samples} 个问题")
            
            # Calculate statistics
            evaluation_stats = {
                "total_samples": total_samples,
                "successful_evaluations": len(valid_results),
                "failed_evaluations": len(failed_cases),
                "error_evaluations": len(error_cases),
                "success_rate": len(valid_results) / total_samples * 100,
                "average_accuracy": np.mean([r["accuracy"] for r in valid_results]) if valid_results else 0,
                "error_summary": {
                    "placeholder_answers": len(failed_cases),
                    "processing_errors": len(error_cases)
                }
            }
            
            # Create output filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"qwen_results_reevaluated_{timestamp}.json"
            
            # Save updated results
            output_data = {
                "results": results,
                "failed_cases": failed_cases,
                "error_cases": error_cases,
                "statistics": evaluation_stats
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            # Print evaluation report
            print("\n=== Qwen重新评估报告 ===")
            print(f"总样本数: {total_samples}")
            print(f"有效评估数: {len(valid_results)}")
            print(f"占位符答案数(@@@@@): {len(failed_cases)}")
            print(f"错误案例数: {len(error_cases)}")
            print(f"有效评估准确率: {evaluation_stats['average_accuracy']:.4f}")
            print(f"有效评估比例: {evaluation_stats['success_rate']:.2f}%")
            print(f"\n结果已保存至: {output_file}")
            
            return evaluation_stats
            
        except Exception as e:
            print(f"评估过程中出错: {str(e)}")
            return None

def main():
    # Initialize evaluator
    evaluator = QwenEvaluator()
    
    # Find the most recent qwen results file
    qwen_files = [f for f in os.listdir('.') if f.startswith('qwen_results_') and f.endswith('.json')]
    if not qwen_files:
        print("未找到Qwen评估结果文件！")
        return
        
    latest_file = max(qwen_files, key=os.path.getctime)
    print(f"正在处理文件: {latest_file}")
    
    # Evaluate results
    stats = evaluator.evaluate_results(latest_file)
    
    if stats:
        print("\n评估完成！")
    else:
        print("\n评估失败！")

if __name__ == "__main__":
    main()