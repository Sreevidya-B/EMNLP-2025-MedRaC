
from pathlib import Path
import json
import pandas as pd
from typing import Dict,Any, Optional,Union,List, Tuple
from model import APIModel
import re
import os
import random
import time


# Re-usable system-message template
SYS_MSG = (
    "You are a medical-calculation analysis assistant. "
    "Based on the provided information, analyze the medical calculation answer "
    "and determine whether **{error_type}** is present in the given answer. "

)


def _parse_replies(raw_replies: List[Any]) -> List[Dict[str, Any]]:
    """
    Convert DeepSeek’s raw output into a uniform list of dictionaries:

        [{"error_present": str, "explanation": str}, …]

    Robust to all known return formats:

    • (dict, in_tok, out_tok)                 ← new DeepSeek format
    • (str,  in_tok, out_tok)                 ← legacy DeepSeek format
    • plain str
    • dicts / tuples / lists (OpenAI-style)
    • Markdown-wrapped JSON  ```json … ```
    • Free-text with key–value phrases (“error_present: Yes …”)
    """

    # ---------- regex compiled once -------------------------------
    KEY_RE = re.compile(
        r"error[_\s\-]?present\s*"          # key
        r"(?:['\"]\s*)?[:=]\s*(?:['\"]\s*)?"  # :  or = with optional quotes
        r"(?P<val>yes|no|true|false|present|absent)\b",
        re.I,
    )
    EXPL_RE = re.compile(
        r"explanation\s*(?:['\"]\s*)?[:=]\s*(?P<val>.+)",
        re.I | re.S,
    )

    # ---------- helper to flatten ---------------------------------
    def _to_text(r: Any) -> str:
        # (dict, int, int)  → handled outside
        if (
            isinstance(r, tuple)
            and len(r) == 3
            and isinstance(r[0], str)
            and all(isinstance(x, int) for x in r[1:])
        ):
            return r[0]

        if isinstance(r, str):
            return r

        if isinstance(r, (tuple, list)):
            for item in r:
                if isinstance(item, str):
                    return item
                if isinstance(item, dict) and "content" in item:
                    return str(item["content"])
            return str(r)

        if isinstance(r, dict):
            if "content" in r:
                return str(r["content"])
            if "message" in r and isinstance(r["message"], dict):
                return str(r["message"].get("content", ""))
            return str(r)

        return str(r)

    # ---------- main loop ------------------------------------------
    parsed: List[Dict[str, Any]] = []

    for r in raw_replies:
        # 0) fast-path – new tuple format with native dict
        if (
            isinstance(r, tuple)
            and len(r) == 3
            and isinstance(r[0], dict)
            and "error_present" in r[0]
        ):
            d = r[0]
            parsed.append(
                {
                    "error_present": d.get("error_present", "Unknown"),
                    "explanation":   d.get("explanation", ""),
                }
            )
            continue

        # 1) flatten to text
        text = _to_text(r).strip()

        # 2) strip markdown ```json fences, if any
        if text.startswith("```"):
            text = re.sub(r"^```[^\n]*\n", "", text)  # opening fence
            text = re.sub(r"\n```$", "", text).strip()  # closing fence

        # 3) try strict JSON
        try:
            obj = json.loads(text)
            if "error_present" in obj:
                parsed.append(
                    {
                        "error_present": obj.get("error_present", "Unknown"),
                        "explanation":   obj.get("explanation", ""),
                    }
                )
                continue
        except Exception:
            pass  # fall through to regex

        # 4) regex rescue
        err_val = "Unknown"
        expl_val = text

        if (m := KEY_RE.search(text)):
            err_val = (
                "Yes"
                if m.group("val").lower() in ("yes", "true", "present")
                else "No"
            )

        if (m := EXPL_RE.search(text)):
            expl_val = m.group("val").strip().split("\n\n")[0].strip()

        parsed.append({"error_present": err_val, "explanation": expl_val})

    return parsed


def error_type_pipeline(input_json: str, output_json_dir: str, model_name: str) -> None:
    """
    Evaluate eight classes of error types for every row in `input_json`
    and save the combined results under `output_json_dir`.

    All prompts are concatenated and sent to the model once.  Returned
    replies are partitioned back into per-error-type blocks.
    """

    raw_json_file = Path(input_json)
    if not raw_json_file.exists():
        raise FileNotFoundError(raw_json_file)

    df = pd.DataFrame(json.load(raw_json_file.open()))

    # # bookkeeping
    # model_name = df["Model Name"].iloc[0]
    # safe_model_name = model_name.replace("/", "_")
    
    # bookkeeping: model in the dataset (the generator being evaluated)
    dataset_model_name = df["Model Name"].iloc[0]
    safe_model_name = dataset_model_name.replace("/", "_")

    responses      = df["LLM Original Answer"].tolist()
    ground_truths  = df["Ground Truth Answer"].tolist()
    calids         = df["Calculator ID"].astype(str).tolist()
    extracted_vals = df["Relevant Entities"].tolist()
    notes          = df["Patient Note"].tolist()
    questions      = df["Question"].tolist()

    # deepseek = APIModel(
    #     model_name,
    #     # "OpenAI/gpt-4.1-mini",
    #     rpm_limit=600,
    #     tpm_limit=5_000_000,
    #     temperature=0.1,
    # )
    
    # Initialize evaluator WITHOUT using remote APIModel / DeepSeek.
    # Accept a vllmModels instance or a string model name to load locally.
    try:
        from model import vllmModels
    except Exception as e:
        raise RuntimeError("vllmModels not available; install or provide an evaluator instance") from e

    evaluator = None
    # If caller passed an LLM instance (vllmModels), use it directly.
    if not isinstance(model_name, str) and hasattr(model_name, "generate"):
        evaluator = model_name
    else:
        # model_name is a string (or None) -> instantiate local vllm model
        # local_name = model_name if isinstance(model_name, str) and model_name else "Qwen/Qwen2.5-3B-Instruct"
        local_name = model_name if isinstance(model_name, str) and model_name else "Qwen/Qwen3-4B"
        try:
            evaluator = vllmModels(model_name=local_name)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize local evaluator '{local_name}'. Pass a vllmModels instance.") from e

    # ---------- build prompts (functions defined elsewhere) -------------
    prompts_formula = build_formula_error_prompts(
        ground_truth_formulas=_get_formulas(
            calids=calids, json_path="data/formula_new.json"
        ),
        answers=responses,
    )

    prompts_var = build_variable_extraction_error_prompts(
        patient_notes=notes,
        questions=questions,
        ground_truth_Extracted_values=extracted_vals,
        answers=responses,
    )

    skip_cmis_mask = df["Category"].isin(
        ["lab test", "physical", "date", "dosage conversion"]
    )
    skip_units_mask = ~skip_cmis_mask

    prompts_cmis, cmis_index = [], []
    for i, (skip, n, q, gt, ans) in enumerate(
        zip(skip_cmis_mask, notes, questions, ground_truths, responses)
    ):
        if skip:
            continue
        prompts_cmis.extend(
            build_clinical_misinterpretation_prompts([n], [q], [gt], [ans])
        )
        cmis_index.append(i)

    prompts_miss_var = build_missing_variable_prompts(
        patient_notes=notes,
        questions=questions,
        ground_truth_Extracted_values=extracted_vals,
        answers=responses,
    )

    prompts_unit, unit_index = [], []
    for i, (skip, n, q, gt, ans) in enumerate(
        zip(skip_units_mask, notes, questions, ground_truths, responses)
    ):
        if skip:
            continue
        prompts_unit.extend(
            build_unit_conversion_error_prompts([n], [q], [gt], [ans])
        )
        unit_index.append(i)

    prompts_adj, adj_index = [], []
    for i, (skip, n, q, gt, ans) in enumerate(
        zip(skip_units_mask, notes, questions, ground_truths, responses)
    ):
        if skip:
            continue
        prompts_adj.extend(
            build_adjustment_coefficient_error_prompts([n], [q], [gt], [ans])
        )
        adj_index.append(i)

    prompts_arith = build_arithmetic_error_prompts(responses)

    prompts_round, round_index = [], []
    for i, (skip, gt, ans) in enumerate(zip(skip_units_mask, ground_truths, responses)):
        if skip:
            continue
        prompts_round.extend(build_rounding_error_prompts([gt], [ans]))
        round_index.append(i)

    # ---------- concat & remember slices --------------------------------
    all_prompts: List[Tuple[str, str]] = []
    slices: Dict[str, Tuple[int, int]] = {}
    start = 0

    def _add(name: str, block: List[Tuple[str, str]]) -> None:
        nonlocal start
        slices[name] = (start, start + len(block))
        all_prompts.extend(block)
        start += len(block)

    _add("formula", prompts_formula)
    _add("var",     prompts_var)
    _add("cmis",    prompts_cmis)
    _add("miss",    prompts_miss_var)
    _add("unit",    prompts_unit)
    _add("adj",     prompts_adj)
    _add("arith",   prompts_arith)
    _add("round",   prompts_round)
    
    # print(f"Total prompts to process: {len(all_prompts)}")

    # ---------- single generate ----------------------------------------
    # all_results = _parse_replies(deepseek.generate(prompts=all_prompts))
    
    # Use the local evaluator instance provided/created above.
    raw_outputs = evaluator.generate(prompts=all_prompts)
    all_results = _parse_replies(raw_outputs)
    
    # # ---------- CHUNKED GENERATION WITH RETRY --------------------------
    # all_results = []
    # failed_indices = []

    # chunk_size = 1  # Small chunks for long medical notes
    # max_retries = 3

    # total_chunks = (len(all_prompts) + chunk_size - 1) // chunk_size
    # print(f"Processing in {total_chunks} chunks of size {chunk_size}")

    # for chunk_idx, start_idx in enumerate(range(0, len(all_prompts), chunk_size)):
    #     chunk = all_prompts[start_idx : start_idx + chunk_size]
    #     success = False
    
    #     for attempt in range(1, max_retries + 1):
    #         try:
    #             raw_outputs = evaluator.generate(prompts=chunk)
    #             results = _parse_replies(raw_outputs)
    #             all_results.extend(results)
    #             success = True
    #             print(f"Chunk {chunk_idx + 1}/{total_chunks} completed ({len(chunk)} prompts)")
    #             break
    #         except Exception as e:
    #             wait = 2 ** (attempt - 1)
    #             print(f"[warning] Chunk {chunk_idx + 1}/{total_chunks} attempt {attempt}/{max_retries} failed: {e}")
    #             if attempt < max_retries:
    #                 print(f"[warning] Retrying in {wait}s...")
    #                 time.sleep(wait)
    
    #     if not success:
    #         # Fallback: process one-by-one
    #         print(f"[warning] Chunk {chunk_idx + 1}/{total_chunks} failed after {max_retries} attempts.")
    #         print(f"[warning] Falling back to single-prompt mode for {len(chunk)} prompts...")
        
    #         for j, single_prompt in enumerate(chunk):
    #             global_idx = start_idx + j
    #             try:
    #                 raw_single = evaluator.generate(prompts=[single_prompt])
    #                 res = _parse_replies(raw_single)
    #                 if res:
    #                     all_results.extend(res)
    #                 else:
    #                     all_results.append({"error_present": "N/A", "explanation": "Evaluation failed"})
    #                     failed_indices.append(global_idx)
    #             except Exception as e2:
    #                 print(f"[error] Single-prompt generation failed at index {global_idx}: {e2}")
    #                 all_results.append({"error_present": "N/A", "explanation": f"Error: {str(e2)}"})
    #                 failed_indices.append(global_idx)

    # # Validate result count
    # if len(all_results) != len(all_prompts):
    #     print(f"[warning] Generated {len(all_results)} results for {len(all_prompts)} prompts")
    #     while len(all_results) < len(all_prompts):
    #         all_results.append({"error_present": "N/A", "explanation": "Missing result"})
    #         failed_indices.append(len(all_results) - 1)
    #     all_results = all_results[:len(all_prompts)]

    # if failed_indices:
    #     print(f"[warning] {len(failed_indices)} prompts had permanent failures")
    #     print(f"[warning] Failed prompt indices: {failed_indices[:10]}..." if len(failed_indices) > 10 else f"[warning] Failed prompt indices: {failed_indices}")

    # # Print summary statistics
    # success_count = len(all_results) - len(failed_indices)
    # success_rate = (success_count / len(all_prompts)) * 100 if all_prompts else 0
    # print(f"Success rate: {success_count}/{len(all_prompts)} ({success_rate:.1f}%)")
    
    ##
    
    # ---------- slice results back into error-type blocks ---------------
    def _slice(name: str) -> List[Dict[str, Any]]:
        a, b = slices[name]
        return all_results[a:b]

    formula_res  = _slice("formula")
    var_res      = _slice("var")
    cmis_res     = _slice("cmis")
    miss_res     = _slice("miss")
    unit_res     = _slice("unit")
    adj_res      = _slice("adj")
    arith_res    = _slice("arith")
    round_res    = _slice("round")

    # ---------- attach back to df --------------------------------------
    col_pairs = {
        "formula":  (formula_res,  None),
        "var":      (var_res,      None),
        "cmis":     (cmis_res,     cmis_index),
        "miss":     (miss_res,     None),
        "unit":     (unit_res,     unit_index),
        "adj":      (adj_res,      adj_index),
        "arith":    (arith_res,    None),
        "round":    (round_res,    round_index),
    }


    error_name_map = {
        "formula": "Formula Error",
        "var":     "Incorrect Variable Extraction",
        "cmis":    "Clinical Misinterpretation (Rule-based Only)",
        "miss":    "Missing Variables",
        "unit":    "Unit Conversion Error",
        "adj":     "Missing or Misused Demographic/Adjustment Coefficients",
        "arith":   "Arithmetic Errors",
        "round":   "Rounding / Precision Errors",
    }

    n_rows = len(df)
    for key, (block, idx) in col_pairs.items():
        col_base = error_name_map[key]
        err_col = f"{col_base}"
        exp_col = f"{col_base} Explanation"

        errs = ["N/A"] * n_rows
        exps = [""]   * n_rows
        tgt  = idx if idx is not None else range(n_rows)
        for i, row_idx in enumerate(tgt):
            errs[row_idx] = block[i]["error_present"]
            exps[row_idx] = block[i]["explanation"]

        df[err_col] = errs
        df[exp_col] = exps

    # ---------- save ----------------------------------------------------
    out_dir  = Path(output_json_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{safe_model_name}_error_eval.json"
    out_file.write_text(
        json.dumps(df.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[+] Finished – results written to: {out_file.resolve()}")




def build_formula_error_prompts(
    ground_truth_formulas: List[str],
    answers: List[str],
) -> List[Tuple[str, str]]:
    prompts = []
    for gt, ans in zip(ground_truth_formulas, answers):
        system_message = SYS_MSG.format(error_type="Formula Error")
        user_message = (
            f"Ground-Truth Formula:\n{gt}\n\n"
            f"Answer to be evaluated:\n{ans}\n\n"
            "Task: Evaluate whether the formula or scoring system used in the answer is appropriate and correctly constructed for the given clinical context.\n\n"
            "You must check for the following issues:\n"
            "- **Incorrect Formula Selection**: A completely wrong formula is used for the clinical question (e.g., using Cockcroft-Gault for AKI instead of CKD-EPI).\n"
            "- **Internal Formula Construction Errors**: The selected formula appears intended to be correct but is flawed in structure or logic. Look for:\n"
            "   • Incorrect or missing coefficients or constants\n"
            "   • Wrong mathematical operators (e.g., `*` instead of `^`)\n"
            "   • Misused parentheses, terms in wrong places, or reversed logic\n"
            "   • Hallucinated or fabricated terms in formula\n"
            "   • Fabricated or omitted scoring items (e.g., omitting “recent surgery” in the Wells Score, or adding a non-existent item like “family history”)\n\n"
            "**Important Notes:**\n"
            "- Do NOT evaluate variable extraction correctness here.\n"
            "- Do NOT evaluate numerical calculation or rounding accuracy.\n"
            "- If multiple formula variants exist and the answer uses any valid one, it is acceptable.\n"
            "- If the answer includes extra details but the core formula is correct, that is acceptable.\n\n"
            "Return a STRICT JSON response: "
            '{"error_present": "Yes" or "No", "explanation": ""}.'
        )
        prompts.append((system_message, user_message))
    return prompts


def build_variable_extraction_error_prompts(
    patient_notes: List[str],
    questions: List[str],
    ground_truth_Extracted_values: List[str],
    answers: List[str],
) -> List[Tuple[str, str]]:
    prompts = []
    for note, q, gt, ans in zip(patient_notes, questions, ground_truth_Extracted_values, answers):
        system_message = SYS_MSG.format(error_type="Incorrect Variable Extraction Error")
        user_message = (
            f"Patient Note:\n{note}\n\n"
            f"Question:\n{q}\n\n"
            f"Ground-Truth Variable Extraction:\n{gt}\n\n"
            f"Answer to be evaluated:\n{ans}\n\n"
            "Task: Determine whether the answer incorrectly extracted key variables from the patient note.\n\n"
            "You should look for the following possible errors:\n"
            "1. **Wrong value**: The extracted value (e.g., heart rate, creatinine) does not match the patient note.\n"
            "2. **Wrong unit**: The extracted unit is misinterpreted (e.g., µmol/L mistaken for mg/dL).\n"
            "3. **Wrong instance**: Multiple similar values exist (e.g., lab values from different days), and the wrong one was selected.\n\n"
            "**Do NOT evaluate:**\n"
            "- Whether the formula chosen is appropriate and correct\n"
            "- Do NOT judge whether the final answer is correct, focus only on the value extraction part.\n"
            "- Whether the numerical calculation is accurate\n\n"
            "Return a STRICT JSON response: "
            '{"error_present": "Yes" or "No", "explanation": ""}.'
        )
        prompts.append((system_message, user_message))
    return prompts


def build_clinical_misinterpretation_prompts(
    patient_notes: List[str],
    questions: List[str],
    ground_truth_explanations: List[str],
    answers: List[str],
) -> List[Tuple[str, str]]:
    prompts = []
    for note, q, gt, ans in zip(patient_notes, questions, ground_truth_explanations, answers):
        system_message = SYS_MSG.format(error_type="Clinical Misinterpretation Error")
        user_message = (
            f"Patient Note:\n{note}\n\n"
            f"Question:\n{q}\n\n"
            f"Scoring Rubric and corresponding result:\n{gt}\n\n"
            f"Answer to be evaluated:\n{ans}\n\n"
            "Task: Evaluate whether the clinical meaning of each finding was interpreted correctly based on the scoring rubric and patient note.\n\n"
            "This error type reflects a misunderstanding of medical knowledge or common clinical reasoning, leading to incorrect interpretation of the patient's symptoms or findings.\n\n"
            "You should check for the following types of errors:\n"
            "1. **Incorrect severity classification** (e.g., mild vs. severe ascites)\n"
            "2. **Wrong presence/absence judgment** (e.g., assigning points for recent surgery when not present)\n"
            "3. **Incorrect threshold interpretation** (e.g., age >75 incorrectly treated as <75)\n"
            "4. **Misunderstanding clinical terms or context** (e.g., interpreting 'occasional alcohol use' as 'chronic alcohol abuse')\n\n"
            "**Important Notes:**\n"
            "- The variable values may be correctly extracted from the note, but the error lies in the clinical judgment or misclassification.\n"
            "- Do NOT evaluate the correctness of the scoring formula, numeric computation, or unit conversion.\n"
            "- If the clinical inference depends on subtle wording or ambiguity in the note, highlight that in your explanation.\n\n"
            "Return a STRICT JSON response: "
            '{"error_present": "Yes" or "No", "explanation": ""}.'
        )
        prompts.append((system_message, user_message))
    return prompts


def build_missing_variable_prompts(
    patient_notes: List[str],
    questions: List[str],
    ground_truth_Extracted_values: List[str],
    answers: List[str],
) -> List[Tuple[str, str]]:
    prompts = []
    for note, q, gt, ans in zip(patient_notes, questions, ground_truth_Extracted_values, answers):
        system_message = SYS_MSG.format(error_type="Missing Variable Extraction Error")
        user_message = (
            f"Patient Note:\n{note}\n\n"
            f"Question:\n{q}\n\n"
            f"Ground-Truth Variable Extraction:\n{gt}\n\n"
            f"Answer to be evaluated:\n{ans}\n\n"
            "Task: Identify whether the answer failed to extract or include one or more variables that are necessary to perform the correct calculation.\n\n"
            "You should look for cases where:\n"
            "1. A required input variable is completely missing.\n"
            "2. The model skipped over variables because they were ambiguous or not explicitly stated.\n"
            "3. The answer proceeds with partial information, leaving out fields that the formula or score requires.\n\n"
            "**Do NOT evaluate:**\n"
            "- Whether the formula used is correct\n"
            "- Whether the extracted variables are accurate\n"
            "- Whether the final calculation is numerically correct\n\n"
            "Focus only on whether the model omitted key inputs needed to properly execute the formula or scoring rule.\n"
            "Return a STRICT JSON response: "
            '{"error_present": "Yes" or "No", "explanation": ""}.'
        )
        prompts.append((system_message, user_message))
    return prompts



def build_unit_conversion_error_prompts(
    patient_notes: List[str],
    questions: List[str],
    ground_truth_explanations: List[str],
    answers: List[str],
) -> List[Tuple[str, str]]:
    prompts = []
    for note, q, gt, ans in zip(patient_notes, questions, ground_truth_explanations, answers):
        system_message = SYS_MSG.format(error_type="Unit Conversion Error")
        user_message = (
            f"Patient Note:\n{note}\n\n"
            f"Question:\n{q}\n\n"
            f"Ground-Truth Explanation:\n{gt}\n\n"
            f"Answer to be evaluated:\n{ans}\n\n"
            "Task: Evaluate whether any input variable was used with the wrong unit, or skipped unit conversion when required by the formula.\n\n"
            "You should look for the following types of errors:\n"
            "1. The value is used directly without converting to the expected unit (e.g., using creatinine 134 µmol/L directly in a formula that expects mg/dL).\n"
            "2. The conversion is attempted but the result is wrong (e.g., wrong conversion factor or direction).\n"
            "3. The unit label is misunderstood or misinterpreted (e.g., confusing mEq/L with mmol/L).\n\n"
            "**Do NOT evaluate:**\n"
            "- Whether the formula chosen is appropriate\n"
            "- Whether the correct value was extracted from the note\n"
            "- Whether the afterwards numerical computation was otherwise accurate\n\n"
            "Only evaluate whether the units used match those required by the formula, and whether any necessary conversions were done correctly.\n"
            "Return a STRICT JSON response: "
            '{"error_present": "Yes" or "No", "explanation": ""}.'
        )
        prompts.append((system_message, user_message))
    return prompts



def build_adjustment_coefficient_error_prompts(
    patient_notes: List[str],
    questions: List[str],
    ground_truth_explanations: List[str],
    answers: List[str],
) -> List[Tuple[str, str]]:
    prompts = []
    for note, q, gt, ans in zip(patient_notes, questions, ground_truth_explanations, answers):
        system_message = SYS_MSG.format(error_type="Missing or Misused Demographic/Adjustment Coefficient Error")
        user_message = (
            f"Patient Note:\n{note}\n\n"
            f"Question:\n{q}\n\n"
            f"Ground-Truth Explanation:\n{gt}\n\n"
            f"Answer to be evaluated:\n{ans}\n\n"
            "Task: Evaluate whether demographic- or context-based adjustment coefficients were properly applied in the formula.\n\n"
            "Specifically, check for:\n"
            "1. **Missing adjustment** — A coefficient required by the formula is missing (e.g., sex multiplier is omitted).\n"
            "2. **Incorrect coefficient used** — The formula includes a coefficient, but it does not match the patient's characteristics (e.g., using male factor for a female patient).\n"
            "3. **Incorrect demographic inference** — The model assumes the wrong demographic category (e.g., classifying patient as non-Black when clearly stated otherwise).\n\n"
            "**Common adjustment dimensions may include:**\n"
            "- Sex (e.g., male vs. female)\n"
            "- Race/ethnicity (e.g., Black vs. non-Black)\n"
            "- Age thresholds\n"
            "- Pregnancy status\n"
            "- Weight class (e.g., obese vs. normal weight)\n\n"
            "**Do NOT evaluate:**\n"
            "- Formula structure or selection\n"
            "- Variable extraction accuracy\n"
            "- Unit conversion correctness\n"
            "- Final numerical calculation\n\n"
            "Return a STRICT JSON response: "
            '{"error_present": "Yes" or "No", "explanation": ""}.'
        )
        prompts.append((system_message, user_message))
    return prompts



def build_arithmetic_error_prompts(
    answers: List[str],
) -> List[Tuple[str, str]]:
    prompts = []
    for  ans in answers:
        system_message = SYS_MSG.format(error_type="Arithmetic Error")
        user_message = (
            f"Answer to be evaluated:\n{ans}\n\n"
            "Task: All variables, units, and formula structure are assumed to be correct.\n"
            "Your task is to verify whether the **arithmetic computation** itself is correct.\n\n"
            "Check for:\n"
            "1. Basic arithmetic errors (e.g., 4 + 3 = 6)\n"
            "2. Wrong order of operations (e.g., using left-to-right instead of proper parentheses)\n"
            "3. Errors in exponentiation, multiplication, or division\n"
            "4. Missing or duplicated numeric terms\n\n"
            "**Do NOT evaluate:**\n"
            "- Formula selection or structure\n"
            "- Variable extraction\n"
            "- Unit conversion\n"
            "- Rounding or precision formatting\n\n"
            "If the calculation process is entirely accurate, a reasonable margin of error is acceptable."
            "Return a STRICT JSON response: "
            '{"error_present": "Yes" or "No", "explanation": ""}.'
        )
        prompts.append((system_message, user_message))
    return prompts


def build_rounding_error_prompts(
    ground_truth_explanations: List[str],
    answers: List[str],
) -> List[Tuple[str, str]]:
    prompts = []
    for gt, ans in zip(ground_truth_explanations, answers):
        system_message = SYS_MSG.format(error_type="Rounding / Precision Error")
        user_message = (
            f"Ground-Truth Explanation:\n{gt}\n\n"
            f"Answer to be evaluated:\n{ans}\n\n"
            "Task: Determine whether the numeric **final result** in the answer is imprecise due to **rounding or insufficient decimal precision**, "
            "even if the formula used and the overall arithmetic are mostly correct.\n\n"
            "This error type should be marked when rounding errors or insufficient precision in intermediate or final steps "
            "cause the final answer to fall outside the tolerance range.\n\n"
            "**Rules for evaluating precision:**\n"
            "- Use the number of decimal places in the LLM's final answer to determine the expected precision (up to a maximum of 2 decimal places).\n"
            "- If the answer is `10.65` → round to **2 decimal places**, tolerance ±0.005\n"
            "- If the answer is `10.7`  → round to **1 decimal place**, tolerance ±0.05\n"
            "- If the answer is `10.6512` → still round to **2 decimal places**, tolerance ±0.005 (overprecision beyond 2 d.p. does **not** increase accuracy expectations)\n\n"
            "**DO mark as a Rounding / Precision Error if:**\n"
            "- The calculation is mostly correct but rounding was done incorrectly (e.g., too few decimals)\n"
            "- The result deviates from the ground truth only because the final answer lacks the required precision (per above rules)\n\n"
            "**DO NOT mark as Rounding / Precision Error if:**\n"
            "- The formula used is incorrect\n"
            "- The arithmetic calculation is wrong\n"
            "- The answer is completely off due to conceptual misunderstanding\n\n"
            "Return a STRICT JSON response with:\n"
            '{"error_present": "Yes" or "No", "explanation": ""}'
        )
        prompts.append((system_message, user_message))
    return prompts



def _get_formulas(calids: List[str], json_path: str) -> List[str]:
    """
    Given a list of Calculator IDs, return a list of the corresponding formulas
    by looking them up in a JSON file. If any ID is not found, raise a ValueError.

    Parameters:
    calids (List[str]): A list of calculator ID strings to look up.
    json_path (str): Path to the JSON file containing formulas. Defaults to 'data/formula.json'.

    Returns:
    List[str]: A list of formulas matching each ID in `calids`.

    Raises:
    ValueError: If any ID in `calids` is not present in the JSON data.
    """
    # Load the JSON data from file
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Build a mapping from Calculator ID to Formula
    id_to_formula = {
        item["Calculator ID"]: item["Formula"]
        for item in data
    }

    # Lookup each ID, raising an error if it's missing
    formulas = []
    for cid in calids:
        if cid not in id_to_formula:
            raise ValueError(f"Calculator ID '{cid}' not found in {json_path}")
        formulas.append(id_to_formula[cid])

    return formulas

def sample_json_data(input_json_path: Union[str, Path], output_dir: Union[str, Path], n: int):
    """
    Randomly sample `n` entries from a JSON file (which is a list of objects) 
    and save to output_dir/sample_{n}.json.
    
    Args:
        input_json_path (str or Path): Path to the input JSON file.
        output_dir (str or Path): Directory to save the sampled JSON.
        n (int): Number of items to sample.
    
    Raises:
        ValueError: If the input JSON is not a list or n > len(data).
    """
    input_json_path = Path(input_json_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read JSON file
    with input_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of records.")

    if n > len(data):
        raise ValueError(f"Cannot sample {n} entries from only {len(data)} available records.")

    sampled_data = random.sample(data, n)

    output_path = output_dir / f"sampled_{n}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(sampled_data, f, indent=2, ensure_ascii=False)

    print(f"Sampled {n} records saved to {output_path}")