import logging
import math
import re
from contextlib import nullcontext
from typing import List, Optional

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLM:
    def __init__(
        self,
        model_name: str,
        tokenizer_name: str,
        device: str = "cuda",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
    ):
        self.device = device
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        if model_name.startswith("osunlp"):
            # from llama_attn_replace import replace_llama_attn

            # replace_llama_attn(use_flash_attn=False)

            # Set RoPE scaling factor
            context_size = 8192
            config = transformers.AutoConfig.from_pretrained(model_name)

            orig_ctx_len = getattr(config, "max_position_embeddings", None)
            if orig_ctx_len and context_size > orig_ctx_len:
                scaling_factor = float(math.ceil(context_size / orig_ctx_len))
                config.rope_scaling = {"type": "linear", "factor": scaling_factor}

            # Load model and tokenizer
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                device_map=device,
            )
            self.model.resize_token_embeddings(32001)

            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_name,
                model_max_length=(
                    context_size if context_size > orig_ctx_len else orig_ctx_len
                ),
                padding_side="left",
                use_fast=False,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=self.dtype,
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                trust_remote_code=True,
            )

            # Tokenizer with optimized settings
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                padding_side="left",
                use_fast=True,  # Enable rust tokenizer
                truncation_side="left",
                trust_remote_code=True,
            )

        # Tokenizer with optimized settings
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Pre-compile regex patterns
        self.response_pattern = re.compile(r"<(.*?)>")
        self.split_pattern = re.compile(r"### Response:")

        # Generation config
        self.generation_config = {
            "max_new_tokens": 128,
            "do_sample": False,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "pad_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }

    def tokenize(self, texts: List[str]) -> dict:
        """Optimized tokenization with attention to memory"""
        return self.tokenizer(
            texts,
            return_tensors="pt",
            pad_to_multiple_of=4,
            padding=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length
            if self.tokenizer.model_max_length < 32768
            else 32768,
            return_token_type_ids=False,
        ).to(self.device)

    def get_response(self, generated_output: str) -> str:
        """Optimized response extraction"""
        response = self.split_pattern.split(generated_output)
        if len(response) > 1:
            result = self.response_pattern.search(response[1])
            if result:
                return f"<{result.group(1)}>"
        return generated_output

    @torch.inference_mode()
    def generate(self, texts: List[str]) -> List[str]:
        """Optimized batch generation with memory management"""
        try:
            model_inputs = self.tokenize(texts)
            generated_ids = self.model.generate(
                **model_inputs, **self.generation_config
            )

            # Process outputs in chunks if large batch
            batch_size = len(texts)
            chunk_size = 32  # Process 32 at a time to prevent memory issues
            responses = []

            for i in range(0, batch_size, chunk_size):
                chunk_ids = generated_ids[i : i + chunk_size]
                decoded = self.tokenizer.batch_decode(
                    chunk_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                responses.extend(self.get_response(output) for output in decoded)

            return responses

        finally:
            # Ensure cleanup even if error occurs
            if "model_inputs" in locals():
                del model_inputs
            if "generated_ids" in locals():
                del generated_ids
            if self.device == "cuda":
                torch.cuda.empty_cache()
