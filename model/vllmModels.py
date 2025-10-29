# vllm_models.py
import time
import random
import numpy as np
import torch
from typing import List, Tuple, Optional, Dict, Type
from model.model import LLM
import logging
from transformers import AutoTokenizer
from pydantic import BaseModel
import warnings
import re,json
from tqdm import tqdm  
from transformers import AutoModelForCausalLM

logger = logging.getLogger(__name__)

class vllmModels(LLM):
    """
    Implementation of the LLM interface for models that can be run with vLLM.
    Supports models from providers like deepseek-ai, qwen, and llama.
    """
    def __init__(self, model_name: str,
                 max_tokens: int = 8000,
                 gpu_id: int = 0,
                 temperature: float = 0.6,
                 seed: Optional[int] = None,
                 trust_remote_code: bool = True,
                 top_p: float = 0.95,
                 repetition_penalty: float = 1.0,
                 **kwargs):
        """
        Initialize a vLLM-based model.
        
        Args:
            model_name: Name of the model in format 'Provider/ModelName'
            max_tokens: Maximum number of tokens to generate
            gpu_id: GPU ID to use for inference
            temperature: Sampling temperature for generation
            seed: Random seed for reproducibility
            trust_remote_code: Whether to trust remote code in the model repository
            top_p: Top-p sampling parameter
            repetition_penalty: Repetition penalty parameter
        """
        super().__init__(model_name, max_tokens, temperature, seed, top_p, repetition_penalty, **kwargs)
        
        # Check if model type is supported
        supported_models = ["deepseek", "qwen", "llama"]
        if not any(provider in self.model_type.lower() for provider in supported_models):
            warnings.warn(
                f"Unsupported model type: {self.model_type}. Expected one of: {supported_models}",
                category=UserWarning
            )
        
        self.trust_remote_code = trust_remote_code
        self.gpu_id = gpu_id
        
        # Initialize tokenizer 
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_full, trust_remote_code=self.trust_remote_code
        )
        
        # Set up seed for reproducibility
        if self.seed is None:
            self.seed = int(time.time() * 1000) % (2**32 - 1)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        # Set up device
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        # Import vLLM and initialize model
        try:
            from vllm import LLM as vLLM
            self.llm = vLLM(
                model=self.model_name_full,
                trust_remote_code=self.trust_remote_code,
                max_model_len=self.max_tokens,
                seed=self.seed,
                gpu_memory_utilization=0.96,
                enable_chunked_prefill=True,
                max_num_batched_tokens=4096,
                tokenizer_mode="slow",

            )
            logger.info(f"Model '{self.model_name_full}' loaded successfully.")
        except ImportError:
            logger.error("vLLM not found. Please install with 'pip install vllm'.")
            raise
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise
        
        self._hf_model  = None      
        self._hf_device = None

    
    def generate(
        self,
        prompts: List[Tuple[str, str]],
        batch_size: Optional[int] = 1,
        schema: Optional[Type[BaseModel]] = None
    ) -> List[Tuple[str, int, int]]:
        """
        Generate responses for a list of chat-based prompts using vLLM.

        Args:
            prompts: List of conversation prompts, where each prompt is a tuple of two strings:
                - The first element is the role ("system", "user", or "assistant").
                - The second element is the content of the message.
                Example:
                    [
                        ("system", "You are a helpful assistant."),
                        ("user", "How do I bake a cake?")
                    ]
            batch_size: Number of prompts to process at once (may be overridden based on model size).
            schema: Optional Pydantic model for structured output (unused in this implementation).

        Returns:
            A list of tuples, one per prompt, each containing:
                - response: the generated text response
                - input_tokens: the token count for the input prompt
                - output_tokens: the token count for the generated response
        """
        # Adjust batch size based on model size hints
        model_name_lower = self.model_name_full.lower()

        if any(x in model_name_lower for x in ["0.5b", "1.5b", "1b"]):
            batch_size = 1024
        elif any(x in model_name_lower for x in ["13b", "14b"]):
            batch_size = 64
        elif any(x in model_name_lower for x in ["3b", "4b","phi"]):
            batch_size = 256
        elif any(x in model_name_lower for x in ["7b", "8b"]):
            batch_size = 256
        elif any(x in model_name_lower for x in ["27b", "28b"]):
            batch_size = 128

        logger.info(f"Using batch size {batch_size} for model '{self.model_name_full}'")

        # Convert prompts to the format expected by the LLM and count input tokens
        prompt_texts = self.convert_prompts(prompts)
        input_tokens = [self.compute_tokens(text) for text in prompt_texts]

        responses: List[str] = []
        for i in range(0, len(prompt_texts), batch_size):
            batch = prompt_texts[i : i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1} / {(len(prompt_texts) + batch_size - 1)//batch_size}")
            try:
                from vllm import SamplingParams
                sampling_params = SamplingParams(
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    repetition_penalty=self.repetition_penalty
                )
                outputs = self.llm.generate(batch, sampling_params)

                # ---- parse thinking ----
                batch_responses = [
                    self.parse_think(o.outputs[0].text) for o in outputs
                ]

                responses.extend(batch_responses)
            except Exception as e:
                logger.error(f"Batch generation error: {e}")
                # Fill with error messages to keep list lengths consistent
                responses.extend([f"Error: {e}"] * len(batch))

        # Count tokens for each generated response
        output_tokens = [self.compute_tokens(resp) for resp in responses]

        # Combine into list of tuples
        return list(zip(responses, input_tokens, output_tokens))
    

    def convert_prompts(self, prompts: List[Tuple[str, str]]) -> List[str]:
        """
        Convert input prompts from the format List[Tuple[str, str]] to a list of prompt texts 
        using tokenizer.apply_chat_template.
        
        Each tuple contains a system message and a user message. For each tuple, a list of 
        dictionaries representing the messages is constructed and then converted into prompt text 
        using the tokenizer.
        
        Args:
            prompts: A list of tuples, where each tuple is (system_message, user_message).
        
        Returns:
            A list of prompt texts generated using tokenizer.apply_chat_template.
        """
        tokenizer = self.tokenizer    
        prompt_texts = []
        
        for system_msg, user_msg in prompts:
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            prompt_texts.append(prompt_text)
        
        return prompt_texts
    
    def _generate_single(self, system_msg: str, user_msg: str) -> str:
        """Generate a response for a single prompt pair"""
        # Convert the tuple (system_msg, user_msg) into a prompt text using the tokenizer
        converted_prompt = self.convert_prompts([(system_msg, user_msg)])[0]
        
        try:
            from vllm import SamplingParams
            sampling_params = SamplingParams(
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty
            )
            
            # Generate the response using the LLM; note that generate expects a list of prompt texts.
            outputs = self.llm.generate([converted_prompt], sampling_params)
            response = outputs[0].outputs[0].text
            response = self.parse_think(response)
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            response = f"Error: {str(e)}"
        
        return response


    @staticmethod
    def parse_think(llm_output: str):
        """
        Return the outer-most JSON block from an LLM response, or the raw string.

        • If the text contains a <think>…</think> chain-of-thought block, that block
        is stripped first, and the function scans forward from the first '{'
        to find the matching '}', ensuring a balanced JSON object.
        • After extraction, if the result is a JSON string (i.e. a string-wrapped dict),
        parse it and return a dict; otherwise return the string.

        Parameters
        ----------
        llm_output : str
            Raw text produced by the LLM.

        Returns
        -------
        Union[str, dict]
            The JSON object if found and parsed; otherwise the unmodified string.
        """
        text = llm_output.strip()

        # 1️⃣ Remove chain-of-thought if present
        if "<think>" in text and "</think>" in text:
            text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()
        else:
            # No <think> section → may still be a JSON-wrapped string
            trimmed = text
            if trimmed.startswith('{') and trimmed.endswith('}'):
                try:
                    return json.loads(trimmed)
                except json.JSONDecodeError:
                    return text
            return text

        # 2️⃣ Locate the outer-most JSON object
        start = text.find("{")
        if start == -1:
            return text

        depth = 0
        end = None
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break

        # Unbalanced braces: return cleaned text
        if end is None:
            return text

        snippet = text[start:end + 1]

        # 3️⃣ Attempt to parse the snippet to dict
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            # Fallback: return the raw snippet string
            return snippet
