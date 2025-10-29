#!/usr/bin/env python3
"""
run.py

A demo file
"""
import sys
sys.set_int_max_str_digits(0)   


import argparse
import os
import glob
from model import APIModel, vllmModels
from evaluator import RegEvaluator, LLM_Evaluator

from method.plain             import Plain
from method.selfConsistency   import SelfConsistency
from method.selfRefine        import SelfRefine
from method.medPrompt         import MedPrompt
from method.twoAgent          import TwoAgent
from method.rag               import RAG
from method.medRaC     import MedRaC

from utils.error_type import error_type_pipeline, error_type_pipeline_opt


# Note: by default, we set evaluator & model to gpt-4o-mini, in case deepseek API is not accessible. In our paper, we use deepseek-chat as LLM_Evaluator, and deepseek-reasoner to analyze error types. If you want to reproduce our results/stats, please use the two models

# deepseek = APIModel(
#     'DeepSeek/deepseek-chat',
#     rpm_limit=800,
#     tpm_limit=3000000,
#     temperature=0.0
# )
# gpt = APIModel(
#     'OpenAI/gpt-4o-mini',
#     rpm_limit=800,
#     tpm_limit=3000000,
#     temperature=1.0,
# )

# llm_evaluator = LLM_Evaluator(deepseek)
# reg_evaluator = RegEvaluator()



# If you want to use open-source models, uncomment this line and replace the gpt in method with model
# model = vllmModels(model_name="Qwen/Qwen3-8B")
eva_model = vllmModels(model_name="Qwen/Qwen3-4B")
# eva_model = vllmModels(model_name="Qwen/Qwen3-1.7B")
# eva_model = vllmModels(model_name="Qwen/Qwen3-0.6B")
# eva_model = vllmModels(model_name="Qwen/Qwen3-14B")
llm_evaluator = LLM_Evaluator(eva_model)
reg_evaluator = RegEvaluator()



# Here we give 3 example of Plain & RAG & MedRaC, You can also try other methods mentioned in our paper/imported above. Please check the corresponding file under method dir

# ------- CoT/Direct/Oneshot Method Example ------
# method = Plain(
#     "cot",
#     [gpt],
#     [reg_evaluator, llm_evaluator]
# )

# # set test=False to run the whole test set
# raw = method.generate_raw(test=True)
# eval_json = method.evaluate(raw_json_file=raw)
# reg_evaluator.compute_overall_accuracy_new(input_file_path=eval_json, output_dir_path="stats")
# ------- CoT/Direct/Oneshot Method Example ------  





# ------- RAG Method Example ------
# method = TwoAgent(
#     "cot",
#     [[gpt, gpt]],
#     [reg_evaluator, llm_evaluator]
# )
# raw = method.generate_raw(test=True, use_rag=True)
# eval_json = method.evaluate(raw_json_file=raw)
# reg_evaluator.compute_overall_accuracy_new(input_file_path= eval_json, output_dir_path="stats")
# ------- RAG Method Example ------



# -------- Our MedRaC Method Example -----------
# method = MedRaC(
#     llms=[model],
#     evaluators=[reg_evaluator],
#     model=model,
#     use_rag=False
# )
# raw = method.generate_raw(test=True)
# eval_json = method.evaluate(raw_json_file=raw)
# reg_evaluator.compute_overall_accuracy_new(input_file_path= eval_json, output_dir_path="stats")
# -------- Our MedRaC Method Example -----------

method_eval = MedRaC(
    llms=[eva_model],
    evaluators=[reg_evaluator, llm_evaluator], 
    model=eva_model,
    use_rag=False
)

# Step 2: Evaluate
raw_json = "/home/sreevidyabol_umass_edu/EMNLP-2025-MedRaC/raw_output/code/Qwen_Qwen3-8B_modular_cot_codeQwen3-8B_raw.json"
eval_json_llm = method_eval.evaluate(raw_json_file=raw_json)
# Outputs: eval_output/code/Qwen_Qwen3-8B_modular_cot_codeQwen3-8B_eval.json

# Step 3: Compute statistics (generates the results file)
reg_evaluator.compute_overall_accuracy_new(input_file_path= eval_json_llm, output_dir_path="stats")
# Outputs: stats/results_Qwen_Qwen3-8B_modular_cot_codeQwen3-8B_eval.json

# ------------ Error Type Analysis -------------
error_type_pipeline_opt(input_json=eval_json_llm, output_json_dir="ErrorTypes", model = eva_model)
# Outputs: ErrorTypes/Qwen_Qwen3-8B_error_eval.json

