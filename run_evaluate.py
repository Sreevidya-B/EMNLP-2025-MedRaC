#!/usr/bin/env python3
"""
run_evaluate.py

Load only the smaller evaluator model and run error-type analysis on an existing eval.json.
This avoids loading both large models in one process (prevents OOM).
"""
import sys
sys.set_int_max_str_digits(0)

import argparse
from model import vllmModels
from evaluator import LLM_Evaluator, RegEvaluator
from utils.error_type import error_type_pipeline
from method.plain import Plain
from method.selfConsistency import SelfConsistency
from method.selfRefine import SelfRefine
from method.medPrompt import MedPrompt
from method.twoAgent import TwoAgent
from method.rag import RAG
from method.medRaC import MedRaC

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-json", required=True, help="Path to the eval json produced by run_generate.py")
    # p.add_argument("--evaluator", default="Qwen/Qwen2.5-3B-Instruct", help="Evaluator (smaller) model name")
    p.add_argument("--evaluator", default="Qwen/Qwen3-4B", help="Evaluator (smaller) model name")
    args = p.parse_args()

    # 1. Load only the evaluator (smaller) model
    try:
        evaluator_model = vllmModels(model_name=args.evaluator)
        print(f"Evaluator model ({args.evaluator}) loaded successfully")
    except Exception as e:
        print(f"[error] Could not load evaluator {args.evaluator}: {e}")
        raise

    # 2. Wrap evaluator if LLM_Evaluator is needed elsewhere
    llm_evaluator = LLM_Evaluator(evaluator_model)
    reg_evaluator = RegEvaluator()  # optional, if additional reg-based checks are needed

    # 3. Run error-type analysis (passes the vllmModels instance into pipeline)
    error_type_pipeline(input_json=args.eval_json, output_json_dir="ErrorTypes", model_name=evaluator_model)

    print("[Evaluation finished. Error-type outputs written under ErrorTypes/")

if __name__ == "__main__":
    main()