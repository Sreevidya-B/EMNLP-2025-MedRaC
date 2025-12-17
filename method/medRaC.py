# from method.open_source_rag import OpenSourceRAG
from model import LLM
from typing import Union, Tuple, List
import json
import asyncio
from textwrap import dedent
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from method import Method
import os
from evaluator import Evaluator
from schema.schemas import FormulaAndValues, Values
import logging
import pandas as pd
import re
import time
from method import RAG
# from method.open_source_rag import OpenSourceRAG
logger = logging.getLogger(__name__)

class MedRaC(Method):
    """
    A class to generate and execute code based on formulas and extracted values using an LLM model.
    """

    # Commonly available Python standard and preinstalled packages
    COMMON_PACKAGES = [
        'math', 'datetime', 'json', 're', 'os', 'sys',
        'numpy', 'pandas'
    ]

    def __init__(
        self,
        llms: List[LLM],
        evaluators: List[Evaluator],
        model: LLM,
        use_rag: bool = False,
    ):
        super().__init__(llms=llms)
        self.evaluators = evaluators
        self.model = model
        self.prompt_style = ("modular_cot_code_rag" if use_rag else "modular_cot_code") + self.model.model_name

        self.input_tokens_used = 0
        self.output_tokens_used = 0

        self.input_tokens = {}
        self.output_tokens = {}
        self.responses = {}
        self.correctness = {}
        self.formulas    = {}
        self.rag = RAG() if use_rag else None
        # self.rag = OpenSourceRAG() if use_rag else None

    
    def generate_raw(
        self,
        test: bool = False,
        raw_json_dir: str = "raw_output/code",
        formula_json_path: str = "data/formula_new.json",
    ) -> str:
        """Run the LLM(s) once, persist answers + token stats for each model."""

        # 1) load dataset
        # self.df = self.load_data_test() if test else self.load_dataset()
        
        # --- BEGIN ADDED CODE ---
        if test:
            target_row_numbers = [1030, 1034, 949, 950, 951, 468, 469, 476, 649, 651, 448, 449, 458, 467, 911, 914, 829, 830]
            full_df = self.load_dataset()
            self.df = full_df[full_df["Row Number"].isin(target_row_numbers)].reset_index(drop=True)
        else:
            self.df = self.load_dataset()
        # --- END ADDED CODE ---

        notes     = self.df["Patient Note"].tolist()
        questions = self.df["Question"].tolist()
        calids    = self.df["Calculator ID"].astype(str).tolist()

        # 3) ensure output dir
        os.makedirs(raw_json_dir, exist_ok=True)

        model_to_path = {}
        for llm in self.llm_list:
            model_name      = llm.model_name_full
            safe_model_name = model_name.replace("/", "_")

            if self.rag:
                # RAG: Retrives formulas first
                # formulas = [self.rag.retrieve(question, k=1)[0][0] for question in questions] 
                # --- BEGIN ADDED CODE ---
                formulas = []
                for question in questions:
                    time.sleep(0.7)  # Rate limit control
                    # formulas.append(self.rag.retrieve(question, k=1)[0][0])
                    results = self.rag.retrieve(question, k=3)
                    formulas.append(results[0][0] if results else "")
                # --- END ADDED CODE ---
            
            # # --- BEGIN ADDED CODE --- 
            # if self.rag:
            #     # RAG: Retrieve formulas with reranking + fallback to ground truth
            #     # Load ground-truth formulas as fallback
            #     calids = self.df["Calculator ID"].astype(str).tolist()
            #     expected_formulas = self._get_formulas(calids=calids, json_path=formula_json_path)
            #     id_to_expected = {cid: f for cid, f in zip(calids, expected_formulas)}
                
            #     formulas = []
            #     top_k = 3
            #     for i, (q, cid) in enumerate(zip(questions, calids)):
            #         time.sleep(0.7)  # Rate limit control
            #         results = self.rag.retrieve(q, k=top_k)  # List[(text, score)]
            #         candidate_texts = [t for t, _ in results] if results else []
            #         expected = id_to_expected.get(cid, "")
                    
            #         chosen = None
            #         # 1) exact match
            #         for t in candidate_texts:
            #             if t.strip() == expected.strip():
            #                 chosen = t
            #                 break
                    
            #         # 2) fuzzy containment (handles formatting differences)
            #         if chosen is None and expected:
            #             import re
            #             expected_norm = re.sub(r"\s+", " ", expected.lower()).strip()
            #             scores = []
            #             for t in candidate_texts:
            #                 t_norm = re.sub(r"\s+", " ", t.lower()).strip()
            #                 # token overlap score
            #                 exp_tokens = set(expected_norm.split())
            #                 t_tokens   = set(t_norm.split())
            #                 jaccard = len(exp_tokens & t_tokens) / max(1, len(exp_tokens | t_tokens))
            #                 scores.append((t, jaccard))
            #             if scores:
            #                 t_best, s_best = max(scores, key=lambda x: x[1])
            #                 # require a minimal overlap; else fallback
            #                 if s_best >= 0.35:
            #                     chosen = t_best
                    
            #         # 3) fallback to expected formula from formula_new.json
            #         if chosen is None:
            #             chosen = expected if expected else (candidate_texts[0] if candidate_texts else "")
                    
            #         formulas.append(chosen) 
            #     # --- END ADDED CODE ---          
                                

                self.formulas[model_name] = formulas
                
                # First LLM call to extract values (RAG mode)
                prompts = self._gen_extracted_values(notes, formulas, questions)
                schema = Values
                generations = llm.generate(prompts, schema=schema)
                extracted_values, in_toks, out_toks = map(list, zip(*generations))
                
                # --- BEGIN ADDED CODE ---
                # === DEBUG: Log 1st LLM call (RAG mode) ===
                debug_file_1st = os.path.join(raw_json_dir, f"{safe_model_name}_RAG_1st_call_debug.json")
                debug_records_1st = []
                for i, (prompt, ev, in_tok, out_tok, formula) in enumerate(zip(prompts, extracted_values, in_toks, out_toks, formulas)):
                    sys_msg, usr_msg = prompt
                    debug_records_1st.append({
                        "row_number": int(self.df["Row Number"].iloc[i]) if "Row Number" in self.df.columns else i,
                        # input = system message (fixed) + user message (patient notes + question + formula)
                        "input": {
                            "patient_note": notes[i],
                            "question": questions[i],
                            "retrieved_formula": formula,
                            "system_message": sys_msg,
                            "user_message": usr_msg
                        },
                        "output": {
                            "extracted_values": ev
                        },
                        "tokens": {
                            "input": in_tok,
                            "output": out_tok
                        }
                    })
                with open(debug_file_1st, "w", encoding="utf-8") as f:
                    json.dump(debug_records_1st, f, indent=2, ensure_ascii=False)
                logger.info(f"DEBUG: RAG 1st LLM call logged to {debug_file_1st}")
                # --- END ADDED CODE ---


                self.input_tokens[model_name]  = in_toks
                self.output_tokens[model_name] = out_toks
                
                # Second LLM Call (Code Generation): RAG mode
                answers, codes = self.generate_code(formulas, extracted_values, questions)
                
                # --- BEGIN ADDED CODE ---
                # === DEBUG: Log 2nd LLM call (RAG mode) ===
                debug_file_2nd = os.path.join(raw_json_dir, f"{safe_model_name}_RAG_2nd_call_debug.json")
                code_prompts = self._gen_code_prompt(formulas, extracted_values, questions)
                debug_records_2nd = []
                for i, (code_prompt, code, answer) in enumerate(zip(code_prompts, codes, answers)):
                    sys_msg_code, usr_msg_code = code_prompt
                    debug_records_2nd.append({
                        "row_number": int(self.df["Row Number"].iloc[i]) if "Row Number" in self.df.columns else i,
                        "input": {
                            "formula": formulas[i],
                            "extracted_values": extracted_values[i],
                            "question": questions[i],
                            "system_message": sys_msg_code,
                            "user_message": usr_msg_code
                        },
                        "output": {
                            "generated_code": code,
                            "executed_answer": answer
                        }
                    })
                with open(debug_file_2nd, "w", encoding="utf-8") as f:
                    json.dump(debug_records_2nd, f, indent=2, ensure_ascii=False)
                logger.info(f"DEBUG: RAG 2nd LLM call logged to {debug_file_2nd}")     
                # --- END ADDED CODE ---

            else:
                prompts = self._gen_formula_and_extracted_values(notes, questions)
                schema = FormulaAndValues
                # First LLM Call (Formula/Value Extraction)
                # This call takes the patient note and the question as input and asks the LLM to reason about the problem, identify the correct medical formula
                # and extract the necessary values from the note.
                generations = llm.generate(prompts, schema=schema) 
                responses_list, in_toks, out_toks = map(list, zip(*generations))
                
                # --- BEGIN ADDED CODE ---
                # Write the input and output tokens to a file for each row in the dataset
                with open("input_output_tokens.txt", "w", encoding="utf-8") as f:
                    for i, (prompt, response, in_tok, out_tok) in enumerate(zip(prompts, responses_list, in_toks, out_toks)):
                        system_msg, user_msg = prompt
                        row_number = self.df["Row Number"].iloc[i]
                        f.write(f"row number in the dataset: {row_number}\n")
                        f.write("-"*120 + "\n")
                        # input = system message (fixed) + user message (patient notes + question)
                        f.write("--- INPUT TOKENS (Input Prompt = System Message + User Message): ---\n")
                        f.write("-"*120 + "\n")
                        f.write(f"System Message remains constant across all rows in the dataset.\n")
                        f.write(f"User Message is tailored based on each data entry.\n")
                        f.write("="*120 + "\n")
                        f.write(f"System Message:\n\n{system_msg}\n\n")
                        f.write("-"*120 + "\n")
                        f.write(f"User Message:\n\n{user_msg}\n\n")
                        f.write("="*120 + "\n")
                        f.write("--- OUTPUT TOKENS (Model's generated answer to the input prompt): ---\n\n")
                        f.write(f"{response}\n\n")
                        f.write("="*120 + "\n")
                        f.write(f"Input Tokens: {in_tok}\n")
                        f.write(f"Output Tokens: {out_tok}\n")
                        f.write("="*120 + "\n\n\n")
                
                
                # # === DEBUG: Log 1st LLM call (non-RAG mode) ===
                # debug_file_1st = os.path.join(raw_json_dir, f"{safe_model_name}_1st_call_debug.json")
                # debug_records_1st = []
                # for i, (prompt, response, in_tok, out_tok) in enumerate(zip(prompts, responses_list, in_toks, out_toks)):
                #     sys_msg, usr_msg = prompt
                #     debug_records_1st.append({
                #         "row_number": int(self.df["Row Number"].iloc[i]) if "Row Number" in self.df.columns else i,
                #         "input": {
                #             "patient_note_snippet": notes[i][:300],
                #             "question": questions[i],
                #             "system_message": sys_msg,
                #             "user_message": usr_msg
                #         },
                #         "output": response,
                #         "tokens": {
                #             "input": in_tok,
                #             "output": out_tok
                #         }
                #     })
                # with open(debug_file_1st, "w", encoding="utf-8") as f:
                #     json.dump(debug_records_1st, f, indent=2, ensure_ascii=False)
                # logger.info(f"DEBUG: Non-RAG 1st LLM call logged to {debug_file_1st}")
                
                # --- END ADDED CODE ---
                
                self.input_tokens[model_name]  = in_toks
                self.output_tokens[model_name] = out_toks

                self.formulas[model_name] = self._get_formulas(
                    calids=calids,
                    json_path=formula_json_path,
                )

                formulas, extracted_values = [], []
                for r in responses_list:
                    try:
                        d = r if isinstance(r, dict) else json.loads(r)
                        formulas.append(d.get("formula", r))
                        extracted_values.append(d.get("extracted_values", r))
                    except:
                        formulas.append(r)
                        extracted_values.append(r)

                # --- BEGIN ADDED CODE: 1ST LLM OUTPUT ---
                first_llm_output = []
                for idx, (f, e, q, n) in enumerate(zip(formulas, extracted_values, questions, notes)):
                    first_llm_output.append({
                        "Row Number": int(self.df["Row Number"].iloc[idx]),
                        "question": q,
                        "patient_note_snippet": n,
                        "extracted_formula": f,
                        "extracted_values": e
                    })
                
                first_output_path = os.path.join(raw_json_dir, f"{safe_model_name}_1st_llm_output.json")
                with open(first_output_path, 'w', encoding='utf-8') as f:
                    json.dump(first_llm_output, f, indent=2, ensure_ascii=False)
                print(f"1st LLM output saved to: {first_output_path}")
                # --- END ADDED CODE ---
                
                # Second LLM Call (Code Generation and Execution)
                # This call takes the formula and values extracted from the first step and asks the LLM to generate a Python code snippet to perform the calculation.
                answers, codes = self.generate_code(formulas, extracted_values, questions)
                
                # --- BEGIN ADDED CODE: 2ND LLM INPUT/OUTPUT ---
                second_llm_data = []
                for idx, (f, e, q, code, answer) in enumerate(zip(formulas, extracted_values, questions, codes, answers)):
                    second_llm_data.append({
                        "Row Number": int(self.df["Row Number"].iloc[idx]),
                        "input_to_code_gen": {
                            "formula": f,
                            "extracted_values": e,
                            "question_context": q
                        },
                        "output_from_code_gen": {
                            "generated_code (calculation)": code,
                            "executed_answer (answer)": answer
                        }
                    })
                
                second_output_path = os.path.join(raw_json_dir, f"{safe_model_name}_2nd_llm_output.json")
                with open(second_output_path, 'w', encoding='utf-8') as f:
                    json.dump(second_llm_data, f, indent=2, ensure_ascii=False)
                print(f"2nd LLM input/output saved to: {second_output_path}")
                
                # # === DEBUG: Log 2nd LLM call (non-RAG mode) ===
                # debug_file_2nd = os.path.join(raw_json_dir, f"{safe_model_name}_2nd_call_debug.json")
                # code_prompts = self._gen_code_prompt(formulas, extracted_values, questions)
                # debug_records_2nd = []
                # for i, (code_prompt, code, answer) in enumerate(zip(code_prompts, codes, answers)):
                #     sys_msg_code, usr_msg_code = code_prompt
                #     debug_records_2nd.append({
                #         "row_number": int(self.df["Row Number"].iloc[i]) if "Row Number" in self.df.columns else i,
                #         "input": {
                #             "formula": formulas[i],
                #             "extracted_values": extracted_values[i],
                #             "question": questions[i],
                #             "system_message": sys_msg_code,
                #             "user_message": usr_msg_code
                #         },
                #         "output": {
                #             "generated_code": code,
                #             "executed_answer": answer
                #         }
                #     })
                # with open(debug_file_2nd, "w", encoding="utf-8") as f:
                #     json.dump(debug_records_2nd, f, indent=2, ensure_ascii=False)
                # logger.info(f"DEBUG: Non-RAG 2nd LLM call logged to {debug_file_2nd}")
                
                # --- END ADDED CODE ---
                
            self.responses[model_name] = [
                {"formula": f, "extracted_values": e, "calculation": c, "answer": a}
                for f, e, c, a in zip(formulas, extracted_values, codes, answers)
            ]
            # print(self.formulas[model_name])
            # 7) dump raw records
            raw_records = self._build_records(model_name=model_name, include_evaluation=False)
            raw_path = os.path.join(
                raw_json_dir,
                f"{safe_model_name}_{self.prompt_style}_raw.json"
            )
            self._dump_json(raw_records, raw_path)
            model_to_path[model_name] = raw_path

            logger.info("Raw generation complete – %s saved (%d rows)",
                        raw_path, len(raw_records))

        # return the first (or only) path, or a JSON map if many
        if len(model_to_path) == 1:
            return next(iter(model_to_path.values()))
        return json.dumps(model_to_path, indent=2)

    
    def evaluate(
        self,
        raw_json_file: str,
        eval_json_dir: str = "eval_output/code",
    ) -> str:
        """Read *raw* file, run evaluators, persist extended results."""

        # ensure raw file exists
        if not os.path.exists(raw_json_file):
            raise FileNotFoundError(raw_json_file)

        # 1) read raw data
        with open(raw_json_file, "r", encoding="utf-8") as f:
            records = json.load(f)
        self.df = pd.DataFrame(records)

        # 2) infer model + bookkeeping 
        model_name = self.df["Model Name"].iloc[0]
        safe_model_name = model_name.replace("/", "_")

        self.responses[model_name] = self.df["LLM Original Answer"].tolist()
        self.input_tokens[model_name] = self.df["Input Tokens"].tolist()
        self.output_tokens[model_name] = self.df["Output Tokens"].tolist()
        self.correctness[model_name] = {}
        calids = self.df["Calculator ID"].astype(str).tolist()
        self.formulas[model_name] = self._get_formulas(
            calids=calids,
            json_path="data/formula_new.json",
        )
        # 3) common fields for evaluators 
        ground_truths = self.df["Ground Truth Answer"].tolist()
        calids = self.df["Calculator ID"].astype(str).tolist()
        upper_limits = self.df["Upper Limit"].tolist()
        lower_limits = self.df["Lower Limit"].tolist()

        # 4) run every evaluator
        for evaluator in self.evaluators:
            if (
                "LLM_Evaluator" in evaluator.get_evaluator_name()
                and self.prompt_style == "direct"
            ):
                logger.debug("Skipping %s for direct style",
                            evaluator.get_evaluator_name())
                continue

            key, result_list = evaluator.check_correctness(
                responses=self.responses[model_name],
                ground_truths=ground_truths,
                calids=calids,
                upper_limits=upper_limits,
                lower_limits=lower_limits,
                relevant_entities=self.df["Relevant Entities"].tolist(),
                ground_truth_explanations=self.df["Ground Truth Explanation"].tolist(),
                formulas = self.formulas[model_name],
            )
            self.correctness[model_name][key] = result_list
            self.df[key] = result_list  # append column
            logger.info("Evaluator %s completed", evaluator.get_evaluator_name())

        # 5) token stats for LLM‑based evaluators 
        for evaluator in self.evaluators:
            if "LLM_Evaluator" in evaluator.get_evaluator_name():
                n_rows = len(self.df)
                logger.info(
                    "### LLM Evaluation Average Tokens - %s - input: %.2f | output: %.2f",
                    self.prompt_style,
                    evaluator.input_token_used / n_rows if n_rows else 0,
                    evaluator.output_token_used / n_rows if n_rows else 0,
                )

        # 6) persist full evaluation 
        os.makedirs(eval_json_dir, exist_ok=True)
        eval_path = os.path.join(
            eval_json_dir,
            f"{safe_model_name}_{self.prompt_style}_eval.json"
        )
        self._dump_json(self.df.to_dict(orient="records"), eval_path)
        logger.info("Evaluation written to %s", eval_path)

        return eval_path


    def generate_code(
        self,
        formulas: List[Union[str, dict]],
        extracted_values: List[Union[str, dict]],
        questions: List[str] = None
    ) -> Tuple[List[str], List[str]]:
        """
        Generates and executes code for each formula and extracted values.

        Args:
            formulas: A list of formulas (string or dict) to convert into code.
            extracted_values: A list of values (string or dict) associated with each formula.

        Returns:
            A list of results obtained by executing the generated code.
        """
        # Parse JSON strings if necessary
        parsed_formulas = []
        parsed_values = []
        for f, v in zip(formulas, extracted_values):
            if isinstance(f, str):
                try:
                    f = json.loads(f)
                except json.JSONDecodeError:
                    pass
            if isinstance(v, str):
                try:
                    v = json.loads(v)
                except json.JSONDecodeError:
                    pass
            parsed_formulas.append(f)
            parsed_values.append(v)

        # Generate prompts
        prompts = self._gen_code_prompt(parsed_formulas, parsed_values, questions)

        # Call the model to generate code
        gens = self.model.generate(prompts)  # List[Tuple[str, int, int]]
        
        codes = [r for r, _, _ in gens]

        # Unpack responses and update token counts
        responses = [r for r, _, _ in gens]
        self.input_tokens_used  += sum(it for _, it, _ in gens)
        self.output_tokens_used += sum(ot for _, _, ot in gens)

        # Extract code from responses and execute
        return self._run_responses(responses), codes



    def _gen_code_prompt(
        self,
        formulas: List[Union[str, dict, list]],
        extracted_values: List[Union[str, dict]],
        questions: List[str],
    ) -> List[Tuple[str, str]]:
        """
        Build high-quality prompts that coerce the LLM to return *only*
        executable Python code (wrapped in triple back-ticks) which computes
        `result` for each (formula, values) pair.

        Returns
        -------
        List[Tuple[str, str]]
            Sequence of (system_message, user_message) tuples.
        """

        # ---------- 1. system message ------------------------------------------------
        system_msg = dedent(f"""
            You are an expert clinical calculator that writes short, runnable Python
            snippets.  Follow these rules *exactly*:

            1. **Return ONLY Python code** wrapped in triple backticks, no
            explanations, markdown, or JSON metadata before/after.
            2. Use **only** these standard packages: {', '.join(self.COMMON_PACKAGES)}.
            3. Store the final numeric answer in a variable named `result`.
            Do NOT print anything else.
            4. Never prompt for user input; all required values are provided.

            ## Example (for reference only)

            Formula:
                CrCl (mL/min) = ((140 - age) * weight * gender_coeff) / (72 * Scr)

            Values:
                {{ "age": 65, "weight": 70, "gender_coeff": 0.85, "Scr": 1.3 }}

            Expected output format:

            ```python
            # constants
            age = 65
            weight = 70          # kg
            gender_coeff = 0.85
            Scr = 1.3            # mg/dL

            # computation
            result = ((140 - age) * weight * gender_coeff) / (72 * Scr)
            ```
        """).strip()

        # 2. build user messages 
        prompts: List[Tuple[str, str]] = []
        
        for formula_raw, vals_raw, q in zip(formulas, extracted_values, questions):

            # a) in case rag returns a list of [formulas, scores]
            if isinstance(formula_raw, list) and formula_raw:
                formula = formula_raw[0]
            else:
                formula = formula_raw

            # b) make sure vals is a dict
            try:
                if isinstance(vals_raw, str):
                    vals = json.loads(vals_raw)
                else:
                    vals = vals_raw
                assert isinstance(vals, dict)
                values_block = json.dumps(vals, ensure_ascii=False)
            except Exception:
                values_block = vals_raw
            
            # c) user message
            user_msg = dedent(f"""
                ### Formula
                {formula}

                ### Values
                {values_block}

                ### Related Question (context)
                {q}

                Now return *only* the Python code (triple-back-tick fenced) that
                assigns the final answer to `result`.
            """).strip()

            prompts.append((system_msg, user_msg))

        return prompts


    async def _execute_code(self, code: str) -> str:
        """
        Asynchronously executes the generated code and returns the result.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.execute_code, code)

    def execute_code(self, code: Union[str, dict]) -> str:
        """
        Executes the generated Python code in a sandboxed environment and returns the result.
        Accepts either a string or a dict (with a 'calculation' field).
        """
        # 1) pull out the code string
        if isinstance(code, dict):
            # adjust the key to whatever field holds your snippet
            code_str = code.get("calculation", "")
        else:
            code_str = code

        # 2) strip any ``` fences if present
        if code_str.strip().startswith("```"):
            # remove leading ```[lang]\n and trailing ```
            code_str = re.sub(r"^```[^\n]*\n", "", code_str)
            code_str = re.sub(r"```$", "", code_str)

        # 3) exec in sandbox
        local_vars = {}
        try:
            exec(code_str, {}, local_vars)
            result = local_vars.get("result", None)
            return str(result)
        except Exception as e:
            return f"Execution error: {e}"

    def _run_responses(self, responses, max_workers: int = 4) -> List[str]:
        """
        Executes the generated code in parallel using ThreadPoolExecutor.
        Args:
            responses: List of generated code strings to execute
            max_workers: Number of threads to use for parallel execution
        Returns:
            List of results from executing the code.
        """
        # Initialize results list with None
        results = [None] * len(responses)
        # Map each submitted future to its original index
        futures = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for idx, resp in enumerate(responses):
                if isinstance(resp, dict):
                    code_snippet = resp.get("calculation", "")
                else:
                    code_snippet = resp
            
                futures[executor.submit(self.execute_code, code_snippet.strip())] = idx

            # as_completed gives futures as they finish; tqdm tracks overall progress
            for future in tqdm(as_completed(futures),
                            total=len(futures),
                            desc="Executing code"):
                idx = futures[future]
                results[idx] = future.result()

        return results

    def  _gen_formula_and_extracted_values(self, notes, questions):

        prompts = []
        
        for note, question in zip(notes, questions):
            system_msg = (
                "You are a reasoning assistant that follows a chain-of-thought approach to find important information in a given patient note. "
                "Follow these steps to extract necessary information:\n"
                "1. Reason about which formula(s) are applicable. Then identify the correct formula required for the calculation and state it explicitly.\n"
                "2. First reason about what values are needed then explain where these values appear in the text. Then, explicitly extract the values and map them to the formula variables.\n"
                "{\"formula_reason\": str, \"formula\": str, \"extracted_values_reason\": str, \"extracted_values\": dict}\n\n"
                "- `formula_reason`: The reasons that a formula is applicable.\n"
                "- `formula`: The explicit mathematical equation used for the calculation (e.g., `BMI = weight / height^2`).\n"
                "- `extracted_values_reason`: Justification for how each value was identified.\n"
                "- `extracted_values`: A dictionary mapping variable names to extracted values from the note (e.g., {\"weight\": \"70kg\", \"height\": \"1.75m\"}).\n"
            )
            
            user_msg = (
                f"Here is the patient note:\n"
                f"{note}\n\n"
                f"Here is the task:\n"
                f"{question}\n\n"
                "Please reason through each step carefully, providing justifications before stating the formula and extracted values. Return the response in the specified JSON format."
            )
            
            prompts.append((system_msg, user_msg))
        
        return prompts

    def _gen_extracted_values(self, notes, formulas, questions):

        prompts = []
        
        for note, formula, question in zip(notes, formulas, questions):
            
            system_msg = (
                "You are a reasoning assistant that follows a chain-of-thought approach to find important information in a given patient note. "
                "Follow these steps to extract necessary information:\n"
                "First reason about what values are needed then explain where these values appear in the text. Then, explicitly extract the values and map them to the formula variables.\n"
                "{\"extracted_values_reason\": str, \"extracted_values\": dict}\n\n"
                "- `extracted_values_reason`: Justification for how each value was identified.\n"
                "- `extracted_values`: A dictionary mapping variable names to extracted values from the note (e.g., {\"weight\": \"70kg\", \"height\": \"1.75m\"}).\n"
            )
            
            user_msg = (
                f"Here is the patient note:\n"
                f"{note}\n\n"
                f"Here is the question:\n"
                f"{question}\n\n"
                f"Here is the formula:\n"
                f"{formula}\n\n"
                "Please reason through each step carefully, providing justifications before stating extracted values. Return the response in the specified JSON format."
            )
            
            prompts.append((system_msg, user_msg))
        
        return prompts
    


    @staticmethod
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
