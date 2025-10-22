#!/usr/bin/env python3
"""
run.py

A demo file
"""
import sys
sys.set_int_max_str_digits(0)   

import torch
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

from utils.error_type import error_type_pipeline


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
# 1. Initialize the open-source models
# model = vllmModels(model_name="Qwen/Qwen2.5-0.5B-Instruct")
# model = vllmModels(model_name="Qwen/Qwen2.5-1.5B-Instruct")
# model = vllmModels(model_name="Qwen/Qwen2.5-3B-Instruct")
model = vllmModels(model_name="Qwen/Qwen2.5-7B-Instruct")
# model = vllmModels(model_name="Qwen/Qwen2.5-14B-Instruct")
print("Model (Qwen/Qwen2.5-7B-Instruct) loaded successfully")


# 2. Keep a smaller API model for evaluation (optional - you can use the same model)
# gpt = APIModel(
#     'OpenAI/gpt-4o-mini',
#     rpm_limit=800,
#     tpm_limit=3000000,
#     temperature=0.0,
# )


# Require an independent smaller model for evaluation. Do NOT use the same model instance.
try:
    evaluator_model = vllmModels(model_name="Qwen/Qwen2.5-3B-Instruct")
    print("Evaluator model (Qwen/Qwen2.5-3B-Instruct) loaded successfully")
except Exception as e:
    print(f"[error] Could not load evaluator Qwen/Qwen2.5-3B-Instruct: {e}")
    print("[error] Aborting to avoid self-evaluation bias. Provide a working smaller evaluator model.")
    sys.exit(1)


# 3. Initialize evaluators
# llm_evaluator = LLM_Evaluator(deepseek) # You can also use gpt here if you don't have access to deepseek API

# Use the smaller independent evaluator
llm_evaluator = LLM_Evaluator(evaluator_model)
reg_evaluator = RegEvaluator()
print("Evaluators are setup successfully (independent smaller model)")

# Here we give 3 example of Plain & RAG & MedRaC, You can also try other methods mentioned in our paper/imported above. Please check the corresponding file under method dir

# ------- CoT/Direct/Oneshot Method Example ------
method = Plain(
    "cot",
    [model],  # Changed from [gpt] to [model]
    [reg_evaluator, llm_evaluator]
)

# set test=False to run the whole test set
raw = method.generate_raw(test=True)
eval_json = method.evaluate(raw_json_file=raw)
reg_evaluator.compute_overall_accuracy_new(input_file_path=eval_json, output_dir_path="stats")
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
#     llms=[gpt],
#     evaluators=[reg_evaluator, llm_evaluator],
#     model=gpt,
#     use_rag=True
# )
# raw = method.generate_raw(test=True)
# eval_json = method.evaluate(raw_json_file=raw)
# reg_evaluator.compute_overall_accuracy_new(input_file_path= eval_json, output_dir_path="stats")
# -------- Our MedRaC Method Example -----------



# ------------ Error Type Analysis -------------
# error_type_pipeline(input_json=eval_json, output_json_dir="ErrorTypes", model_name = 'OpenAI/gpt-4o-mini')
error_type_pipeline(input_json=eval_json, output_json_dir="ErrorTypes", model_name=evaluator_model)
