import os
import json
import re
import numpy as np
from typing import List, Tuple, Union, Optional
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

class Evaluator(ABC):
    
    @abstractmethod
    def check_correctness(
        responses: List[str], 
        ground_truth: List[str], 
        calid: List[Union[str, int]],
        upper_limit: List[Union[str, float]], 
        lower_limit: List[Union[str, float]],
        ground_truth_explanations: List[str], 
        relevant_entities: List[Union[dict, str]],
        formulas: Optional[List[str]] = None,
    ) -> Tuple[str, List[str]]:
        """
        Check correctness of answers against ground truth.
        
        Args:
            responses: List of LLM responses
            ground_truth: List of ground truth answers
            calid: List of calculator IDs for each answer
            upper_limit: Upper bounds for acceptable answers
            lower_limit: Lower bounds for acceptable answers
            ground_truth_explanations: List of explanations for the ground truth
            relevant_entities: List of relevant entities for each answer
            
        Returns:
            Tuple with key "Result" and list of "Correct"/"Incorrect" values.
        """
        pass


    @staticmethod
    def compute_overall_accuracy_new(input_file_path: str, output_dir_path: str):
        """
        Compute per-category, aggregated, and overall accuracy / error breakdown
        for both regular-expression evaluation and detailed LLM evaluation.

        Added in this version
        ---------------------
        • Two high-level aggregate buckets are reported, **Equation-based Question**
        and **Rule-based Question**, in addition to every original category and
        the overall totals.

            Equation-based Question  ← { "lab test", "physical",
                                        "date", "dosage conversion" }

            Rule-based Question      ← { "diagnosis", "risk", "severity" }

        The new buckets are calculated *after* all per-category tallies are finished,
        by pooling the underlying category statistics so that every metric
        (averages, conditional error rates, error-type shares, token means, etc.)
        is computed with exactly the same logic that is used elsewhere.
        """
        # -------------------------------------------------------------------------
        file_name   = os.path.basename(input_file_path)
        base_name   = os.path.splitext(file_name)[0]
        os.makedirs(output_dir_path, exist_ok=True)
        ext = Path(input_file_path).suffix.lower()

        # ------------- load -----------------------------------------------------------------
        datas = []
        with open(input_file_path, 'r', encoding='utf-8') as f:
            if ext == '.json':
                datas = json.load(f)
            elif ext == '.jsonl':
                for line in f:
                    line = line.strip()
                    if line:
                        datas.append(json.loads(line))
            else:
                raise ValueError(f"Unsupported file extension: {ext}")

        # ------------- accumulators ---------------------------------------------------------
        regular_eval = defaultdict(list)
        llm_eval = defaultdict(lambda: {
            "results": [],
            "first_error_type": [],
            "error_type_counts": Counter(),
            # conditional error bookkeeping
            "conditional_totals": Counter(),   # denom
            "conditional_errors": Counter()    # numer
        })
        answer_eval = defaultdict(list)

        input_tokens  = []
        output_tokens = []

        # The order matters for conditional error logic
        fields = ["formula", "extracted_values", "calculation", "answer"]

        # ------------- iterate rows ----------------------------------------------------------
        for data in datas:
            category = data.get("Category", "Unknown").strip().lower()
            result   = str(data.get("Result"))
            input_tokens.append(data.get("Input Tokens", -1))
            output_tokens.append(data.get("Output Tokens", -1))

            # 1) regular correctness
            is_correct = 1 if result in ["1", "Correct"] else 0
            regular_eval[category].append(is_correct)

            # 2) LLM-evaluation breakdown
            if "LLM Evaluation" not in data:
                continue
            llm_res = data["LLM Evaluation"]

            # 2a) raw answer correctness
            ans_ok = False
            raw_ans = llm_res.get("answer")
            if isinstance(raw_ans, dict):
                ans_ok = (raw_ans.get("result") == "Correct")
            elif isinstance(raw_ans, str):
                text = re.sub(r'^```json\s*|\s*```$', '', raw_ans, flags=re.IGNORECASE).strip()
                try:
                    parsed_ans = json.loads(text)
                    ans_ok = (parsed_ans.get("result") == "Correct")
                    # # --- BEGIN: ADDED CODE---
                    # # FIX: Check if parsed_ans is a dict before calling .get()
                    # if isinstance(parsed_ans, dict):
                    #     ans_ok = (parsed_ans.get("result") == "Correct")
                    # else:
                    #     logger.warning(f"Answer field parsed to non-dict type: {type(parsed_ans).__name__}")
                    #     ans_ok = False
                    # # --- END: ADDED CODE---
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse answer field JSON: {text[:100]!r}")
            answer_eval[category].append(1 if ans_ok else 0)

            bools = []
            for f in fields:
                raw = llm_res.get(f)
                parsed = {}
                if isinstance(raw, dict):
                    parsed = raw
                elif isinstance(raw, str):
                    text = re.sub(r'^```json\s*|\s*```$', '', raw, flags=re.IGNORECASE).strip()
                    try:
                        parsed = json.loads(text)
                    except json.JSONDecodeError:
                        logger.warning(f"Field `{f}` JSON parse failed (first 100 chars): {text[:100]!r}")
                        parsed = {}
                ok = (parsed.get("result", "") == "Correct")
                # # --- BEGIN: ADDED CODE---
                # # FIX: Handle cases where parsed might not be a dict
                # if isinstance(parsed, dict):
                #     ok = (parsed.get("result", "") == "Correct")
                # else:
                #     # If parsed is a list or other type, treat as incorrect
                #     logger.warning(f"Field `{f}` parsed to non-dict type: {type(parsed).__name__}")
                #     ok = False
                # # --- END: ADDED CODE ---
                bools.append(ok)

            all_correct = all(bools)
            llm_eval[category]["results"].append(1 if all_correct else 0)

            # 2c) capture first error type
            if not all_correct:
                for f, ok in zip(fields, bools):
                    if not ok:
                        llm_eval[category]["first_error_type"].append(f)
                        llm_eval[category]["error_type_counts"][f] += 1
                        break

            # 2d) conditional error bookkeeping
            prev_correct = True
            for f, ok in zip(fields, bools):
                if prev_correct:
                    llm_eval[category]["conditional_totals"][f] += 1
                    if not ok:
                        llm_eval[category]["conditional_errors"][f] += 1
                prev_correct = prev_correct and ok

        # ------------- helper ---------------------------------------------------------------
        def calc_stats(scores: list):
            n = len(scores)
            if n == 0:
                return {"average": 0.0, "std": 0.0, "count": 0}
            arr  = np.array(scores, dtype=float)
            mean = arr.mean()
            std_err = np.sqrt(mean * (1 - mean) / n)
            return {
                "average": round(mean * 100, 2),
                "std":     round(std_err, 2),
                "count":   n
            }

        output = {}

        # ---------- per-category -------------------------------------------------------------
        for category in sorted(set(regular_eval) | set(llm_eval)):
            reg_stats = calc_stats(regular_eval[category])
            llm_stats = calc_stats(llm_eval[category]["results"])
            raw_stats = calc_stats(answer_eval[category])

            llm_dict = {
                "raw_average": raw_stats["average"],
                **llm_stats
            }

            incorrect = llm_eval[category]["results"].count(0)
            if incorrect:
                counts = llm_eval[category]["error_type_counts"]
                for f in fields:
                    llm_dict[f"{f} error"] = round((counts[f] / incorrect) * 100, 2)

            totals = llm_eval[category]["conditional_totals"]
            errs   = llm_eval[category]["conditional_errors"]
            for f in fields:
                denom = totals[f]
                llm_dict[f"{f} error conditional"] = round((errs[f] / denom) * 100, 2) if denom else 0.0

            output[category] = {
                "regular expression evaluation": reg_stats,
                "llm evaluation": llm_dict
            }

        # ---------- NEW AGGREGATE BUCKETS ----------------------------------------------------
        # mapping of aggregate-bucket → set(original categories)
        AGG_MAP = {
            "equation-based question": {"lab test", "physical", "date", "dosage conversion"},
            "rule-based question":     {"diagnosis", "risk", "severity"}
        }

        for agg_name, cat_set in AGG_MAP.items():
            # collect all underlying categories that are actually present
            present = [c for c in cat_set if c in regular_eval]
            if not present:
                # skip bucket if no underlying data
                continue

            # ----- merge regular correctness ---------------------------------------------
            agg_reg  = [v for c in present for v in regular_eval[c]]

            # ----- merge LLM correctness & bookkeeping -----------------------------------
            agg_llm_results = [v for c in present for v in llm_eval[c]["results"]]
            agg_ans_results = [v for c in present for v in answer_eval[c]]

            # merge counters
            agg_error_counts  = Counter()
            agg_cond_totals   = Counter()
            agg_cond_errors   = Counter()
            for c in present:
                agg_error_counts.update(llm_eval[c]["error_type_counts"])
                agg_cond_totals.update(llm_eval[c]["conditional_totals"])
                agg_cond_errors.update(llm_eval[c]["conditional_errors"])

            # compute stats
            reg_stats  = calc_stats(agg_reg)
            llm_stats  = calc_stats(agg_llm_results)
            raw_stats  = calc_stats(agg_ans_results)

            llm_dict = {
                "raw_average": raw_stats["average"],
                **llm_stats
            }

            incorrect = agg_llm_results.count(0)
            if incorrect:
                for f in fields:
                    llm_dict[f"{f} error"] = round((agg_error_counts[f] / incorrect) * 100, 2)

            # conditional error rates
            for f in fields:
                denom = agg_cond_totals[f]
                llm_dict[f"{f} error conditional"] = round((agg_cond_errors[f] / denom) * 100, 2) if denom else 0.0

            output[agg_name] = {
                "regular expression evaluation": reg_stats,
                "llm evaluation": llm_dict
            }

        # ---------- overall -----------------------------------------------------------------
        all_reg    = [v for scores in regular_eval.values() for v in scores]
        all_llm    = [v for info in llm_eval.values() for v in info["results"]]
        all_answer = [v for scores in answer_eval.values() for v in scores]

        overall_reg = calc_stats(all_reg)
        overall_llm = calc_stats(all_llm)
        overall_raw = calc_stats(all_answer)

        overall_llm_dict = {
            "raw_average": overall_raw["average"],
            **overall_llm
        }

        total_incorrect = all_llm.count(0)
        if total_incorrect:
            overall_counts = Counter()
            for info in llm_eval.values():
                overall_counts.update(info["error_type_counts"])
            for f in fields:
                overall_llm_dict[f"{f} error"] = round((overall_counts[f] / total_incorrect) * 100, 2)

        overall_cond_totals = Counter()
        overall_cond_errs   = Counter()
        for info in llm_eval.values():
            overall_cond_totals.update(info["conditional_totals"])
            overall_cond_errs.update(info["conditional_errors"])
        for f in fields:
            denom = overall_cond_totals[f]
            overall_llm_dict[f"{f} error conditional"] = round((overall_cond_errs[f] / denom) * 100, 2) if denom else 0.0

        output["overall"] = {
            "regular expression evaluation": overall_reg,
            "llm evaluation": overall_llm_dict
        }

        # ---------- token stats -------------------------------------------------------------
        output["input_tokens_average"]  = int(round(np.mean(input_tokens)))  if input_tokens  else 0
        output["output_tokens_average"] = int(round(np.mean(output_tokens))) if output_tokens else 0

        # ---------- save --------------------------------------------------------------------
        out_file = os.path.join(output_dir_path, f"results_{base_name}.json")
        with open(out_file, 'w', encoding='utf-8') as wf:
            json.dump(output, wf, indent=4, ensure_ascii=False)

        return out_file, output


    @staticmethod
    def compute_multifile_overall_accuracy_new(input_dir_path: str, output_dir_path: str):
        """
        Compute accuracy for all JSON files in a directory using the updated Evaluator.compute_overall_accuracy_new.

        Args:
            input_dir_path: Directory containing the JSON files with evaluation results.
            output_dir_path: Directory where results will be written, each file will be named "results_<input_filename>.json".
        """
        os.makedirs(output_dir_path, exist_ok=True)

        # Get all JSON files in the directory
        input_files = [f for f in os.listdir(input_dir_path) if f.endswith('.json')]

        if not input_files:
            print(f"No JSON files found in directory: {input_dir_path}")
            return

        processed_files = 0
        for file_name in input_files:
            input_file_path = os.path.join(input_dir_path, file_name)

            try:
                Evaluator.compute_overall_accuracy_new(input_file_path, output_dir_path)
                processed_files += 1
                print(f"Processed file {processed_files}/{len(input_files)}: {file_name}")
            except Exception as e:
                print(f"Error processing file {file_name}: {str(e)}")

        print(f"Completed processing {processed_files} out of {len(input_files)} files.")


    
    