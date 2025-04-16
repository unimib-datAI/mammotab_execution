import re
import os
import torch
from typing import List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from contextlib import nullcontext

test_locally = os.getenv("TEST_LOCALLY", "False").lower() == "true"
shared_cache = "/scratch_share/datai/`whoami`"
cache_dir = shared_cache if os.path.exists(shared_cache) else None


class LLM:
    def __init__(
        self,
        model_name: str,
        tokenizer_name: str,
        device: Optional[str] = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
    ):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        # Model configuration
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=self.dtype,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            cache_dir=cache_dir
        ).to(self.device)

        # Tokenizer with optimized settings
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            padding_side="left",
            use_fast=True,  # Enable Rust-based tokenizer
            truncation_side="left",
            cache_dir=cache_dir
        )
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
        }

    def tokenize(self, texts: List[str]) -> dict:
        """Optimized tokenization with attention to memory"""
        return self.tokenizer(
            texts,
            return_tensors="pt",
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
    def generate(self, texts: List[str], chunk_size: int = 64) -> List[str]:
        """Optimized batch generation with memory management"""
        responses = []
        try:
            # Process in chunks to manage memory
            for i in range(0, len(texts), chunk_size):
                chunk_texts = texts[i: i + chunk_size]

                # Tokenize chunk
                model_inputs = self.tokenize(chunk_texts)

                # Generate responses
                generated_ids = self.model.generate(
                    **model_inputs, **self.generation_config
                )

                # Decode outputs
                decoded = self.tokenizer.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                responses.extend(self.get_response(output)
                                 for output in decoded)

            return responses

        finally:
            # Ensure cleanup even if error occurs
            if "model_inputs" in locals():
                del model_inputs
            if "generated_ids" in locals():
                del generated_ids
            if self.device == "cuda":
                torch.cuda.empty_cache()
