#!/usr/bin/env python3
"""
run_generate.py

Load only the generator model, produce raw.json and eval.json (no evaluator loaded).
"""
import sys
sys.set_int_max_str_digits(0)

import argparse
from model import vllmModels
from evaluator import RegEvaluator
from method.plain import Plain
from method.selfConsistency import SelfConsistency
from method.selfRefine import SelfRefine
from method.medPrompt import MedPrompt
from method.twoAgent import TwoAgent
from method.rag import RAG
from method.medRaC import MedRaC

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--generator", default="Qwen/Qwen2.5-7B-Instruct", help="Generator model name")
    p.add_argument("--test", action="store_true", help="Run fast test subset")
    args = p.parse_args()

    # 1. Load generator only
    model = vllmModels(model_name=args.generator)
    print(f"Generator model ({args.generator}) loaded successfully")

    # 2. Run chosen method to produce raw + eval jsons
    method = Plain(
        "cot",
        [model],
        [RegEvaluator()]  # reg_evaluator used only for compute_overall_accuracy_new later
    )

    raw = method.generate_raw(test=args.test)
    eval_json = method.evaluate(raw_json_file=raw)

    # 3. compute reg-only metrics (does not load LLM evaluator)
    reg_evaluator = RegEvaluator()
    reg_evaluator.compute_overall_accuracy_new(input_file_path=eval_json, output_dir_path="stats")

    print(f"[+] Generation finished. raw file: {raw}, eval file: {eval_json}")

if __name__ == "__main__":
    main()