#!/usr/bin/env python3
"""
Script to find common formula errors between Qwen3-4B and Qwen3-1.7B evaluators
"""

import json
import re
from collections import defaultdict

def parse_formula_result(formula_eval):
    """Parse formula evaluation to extract result and explanation"""
    result = ""
    explanation = ""
    
    if isinstance(formula_eval, dict):
        result = formula_eval.get("result", "")
        explanation = formula_eval.get("explanation", "")
    elif isinstance(formula_eval, str):
        # Try to parse JSON from string (may be wrapped in markdown)
        try:
            # Remove markdown code blocks if present
            cleaned = re.sub(r'```json\s*', '', formula_eval)
            cleaned = re.sub(r'```', '', cleaned)
            cleaned = cleaned.replace("▁", " ").strip()
            parsed = json.loads(cleaned)
            result = parsed.get("result", "")
            explanation = parsed.get("explanation", "")
        except json.JSONDecodeError:
            # Try to extract result using regex
            match = re.search(r'"result"\s*:\s*"(Correct|Incorrect)"', formula_eval)
            if match:
                result = match.group(1)
            exp_match = re.search(r'"explanation"\s*:\s*"([^"]*)"', formula_eval)
            if exp_match:
                explanation = exp_match.group(1)
    
    return result, explanation


def extract_formula_errors(filepath, model_name):
    """Extract all formula errors from an evaluation file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    formula_errors = {}
    
    for entry in data:
        # Use Row Number as unique identifier
        row_num = entry.get("Row Number")
        if not row_num:
            continue
        
        # Check overall result is Incorrect
        if entry.get("Result") != "Incorrect":
            continue
        
        llm_eval = entry.get("LLM Evaluation", {})
        if not llm_eval:
            continue
            
        formula_eval = llm_eval.get("formula", {})
        
        result, explanation = parse_formula_result(formula_eval)
        
        # Store if formula evaluation is Incorrect
        if result == "Incorrect":
            # Get formula from LLM Original Answer
            llm_original_answer = entry.get("LLM Original Answer", {})
            formula_used = llm_original_answer.get("formula", "") if isinstance(llm_original_answer, dict) else ""
            
            # Get the answer from LLM Original Answer
            if isinstance(llm_original_answer, dict):
                answer = llm_original_answer.get("answer", "")
            else:
                answer = str(llm_original_answer) if llm_original_answer else ""
            
            # Get ground truth
            ground_truth_answer = entry.get("Ground Truth Answer", "")
            ground_truth_explanation = entry.get("Ground Truth Explanation", "")
            
            formula_errors[row_num] = {
                "calculator_id": entry.get("Calculator ID", ""),
                "calculator_name": entry.get("Calculator Name", ""),
                "question": entry.get("Question", ""),
                "explanation": explanation,
                "formula_used": str(formula_used),
                "answer": str(answer),
                "ground_truth_answer": str(ground_truth_answer),
                "ground_truth_explanation": str(ground_truth_explanation),
                "category": entry.get("Category", "Unknown")
            }
    
    print(f"{model_name}: Found {len(formula_errors)} formula errors")
    return formula_errors


def main():
    
    qwen3_4b_file = "/work/pi_hongyu_umass_edu/sreevidyabol_umass_edu/MedRaC/EMNLP-2025-MedRaC/eval_output/code/Qwen_Qwen3-8B_modular_cot_code_ragQwen3-4B_eval.json"
    qwen3_1_7b_file = "/work/pi_hongyu_umass_edu/sreevidyabol_umass_edu/MedRaC/EMNLP-2025-MedRaC/eval_output/code/Qwen_Qwen3-8B_modular_cot_code_ragQwen3-1.7B_eval.json"
    
       
    errors_4b = extract_formula_errors(qwen3_4b_file, "Qwen3-4B")
    errors_1_7b = extract_formula_errors(qwen3_1_7b_file, "Qwen3-1.7B")
    
    common_ids = set(errors_4b.keys()) & set(errors_1_7b.keys())
    
    print(f"\n{'='*80}")
    print(f"COMMON FORMULA ERRORS: {len(common_ids)} questions")
    print(f"{'='*80}\n")
    
    common_errors = []
    for serial_no, row_num in enumerate(sorted(common_ids, key=lambda x: int(x)), 1):
        common_errors.append({
            "S.No.": serial_no,
            "Row_Number": row_num,
            "Calculator_ID": errors_4b[row_num]["calculator_id"],
            "Calculator_Name": errors_4b[row_num]["calculator_name"],
            "Category": errors_4b[row_num]["category"],
            "Question": errors_4b[row_num]["question"],
            "Ground_Truth_Answer": errors_4b[row_num]["ground_truth_answer"],
            "Ground_Truth_Explanation": errors_4b[row_num]["ground_truth_explanation"],
            "Formula_Used_by_Qwen3_4B": errors_4b[row_num]["formula_used"],
            "Answer_by_Qwen3_4B": errors_4b[row_num]["answer"],
            "Qwen3-4B_Explanation": errors_4b[row_num]["explanation"],
            "Formula_Used_by_Qwen3_1.7B": errors_1_7b[row_num]["formula_used"],
            "Answer_by_Qwen3_1.7B": errors_1_7b[row_num]["answer"],
            "Qwen3-1.7B_Explanation": errors_1_7b[row_num]["explanation"]
        })
    
    output_file = "common_formula_errors_qwen4b_qwen1_7b.json"
    with open(output_file, 'w') as f:
        json.dump(common_errors, f, indent=2, ensure_ascii=False)
    
    print(f"Detailed results saved to: {output_file}\n")
    
    print("SUMMARY STATISTICS:")
    print(f"  Qwen3-4B total formula errors: {len(errors_4b)}")
    print(f"  Qwen3-1.7B total formula errors: {len(errors_1_7b)}")
    print(f"  Common formula errors: {len(common_ids)}")
    if max(len(errors_4b), len(errors_1_7b)) > 0:
        print(f"  Agreement rate: {len(common_ids) / max(len(errors_4b), len(errors_1_7b)) * 100:.1f}%")
    
    # Category breakdown
    category_counts = defaultdict(int)
    for err in common_errors:
        category_counts[err["Category"]] += 1
    
    print(f"\nCOMMON ERRORS BY CATEGORY:")
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {count}")
    
    # Print first 10 examples
    print(f"\n{'='*80}")
    print("FIRST 10 EXAMPLES OF COMMON FORMULA ERRORS:")
    print(f"{'='*80}\n")
    
    for i, err in enumerate(common_errors[:10], 1):
        print(f"\n\nExample {i}:")
        print(f"\n  Row Number: {err['Row_Number']}")
        print(f"\n  Calculator ID: {err['Calculator_ID']}")
        print(f"\n  Calculator Name: {err['Calculator_Name']}")
        print(f"\n  Category: {err['Category']}")
        print(f"\n  Question: {err['Question']}")
        print(f"\n  Ground Truth Answer: {err['Ground_Truth_Answer']}")
        print(f"\n  Ground Truth Explanation: {err['Ground_Truth_Explanation']}")
        print(f"\n  Formula used by Qwen3-4B: {err['Formula_Used_by_Qwen3_4B']}")
        print(f"\n  Formula used by Qwen3-1.7B: {err['Formula_Used_by_Qwen3_1.7B']}")
        print(f"\n  Answer by Qwen3-4B: {err['Answer_by_Qwen3_4B']}")
        print(f"\n  Answer by Qwen3-1.7B: {err['Answer_by_Qwen3_1.7B']}")
        print(f"\n  Qwen3-4B explanation:")
        print(f"    {err['Qwen3-4B_Explanation']}")
        print(f"\n  Qwen3-1.7B explanation:")
        print(f"    {err['Qwen3-1.7B_Explanation']}")
        print()
    
    print(f"\nFull details in: {output_file}")

if __name__ == "__main__":
    main()