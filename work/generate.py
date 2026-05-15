import re
import os
import torch
import logging
from pathlib import Path
from typing import List, Optional
from time import perf_counter
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from model_utils import normalize_adapter_path

test_locally = os.getenv("TEST_LOCALLY", "False").lower() == "true"
shared_cache = "/scratch_share/datai/`whoami`"
cache_dir = shared_cache if os.path.exists(shared_cache) else None
logger = logging.getLogger(__name__)

DTYPE_ALIASES = {
    "auto": "auto",
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def resolve_dtype(dtype: Optional[str], device: str):
    if dtype is None or not dtype.strip():
        return "auto" if device == "cuda" else torch.float32

    normalized = dtype.strip().lower()
    if normalized not in DTYPE_ALIASES:
        valid_values = ", ".join(sorted(DTYPE_ALIASES))
        raise ValueError(
            f"Invalid model dtype: {dtype}. Valid values: {valid_values}")

    return DTYPE_ALIASES[normalized]


def resolve_max_input_tokens(model_max_length: int) -> int:
    configured = os.getenv("MAX_INPUT_TOKENS")
    hard_cap = model_max_length if model_max_length < 32768 else 32768
    if configured is None or not configured.strip():
        return hard_cap

    value = int(configured)
    if value <= 0:
        raise ValueError("MAX_INPUT_TOKENS must be a positive integer.")
    return min(value, hard_cap)


class LLM:
    def __init__(
        self,
        model_name: str,
        tokenizer_name: str,
        device: Optional[str] = None,
        adapter_path: Optional[str] = None,
        offload_dir: Optional[str] = None,
        model_dtype: Optional[str] = None,
    ):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = resolve_dtype(
            model_dtype or os.getenv("MODEL_DTYPE"), self.device)
        adapter_path = normalize_adapter_path(adapter_path)
        offload_dir = offload_dir or os.getenv("OFFLOAD_DIR")
        if offload_dir:
            Path(offload_dir).mkdir(parents=True, exist_ok=True)

        device_map = "auto" if self.device == "cuda" else None

        # Model configuration
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            dtype=self.dtype,
            cache_dir=cache_dir,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            offload_folder=offload_dir,
            offload_state_dict=offload_dir is not None,
        )

        if adapter_path:
            try:
                from peft import PeftModel
            except ImportError as e:
                raise ImportError(
                    "Loading adapters requires the peft package. "
                    "Install it with `pip install peft`."
                ) from e

            self.model = PeftModel.from_pretrained(
                self.model,
                adapter_path,
                is_trainable=False,
                offload_folder=offload_dir,
                low_cpu_mem_usage=True,
            )

        self.model.eval()

        # Tokenizer with optimized settings
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            padding_side="left",
            use_fast=True,  # Enable Rust-based tokenizer
            truncation_side="left",
            cache_dir=cache_dir,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_input_tokens = resolve_max_input_tokens(
            self.tokenizer.model_max_length
        )

        # Pre-compile regex patterns
        self.response_pattern = re.compile(r"<(.*?)>")
        self.split_pattern = re.compile(r"### Response:")

        # Generation config
        self.generation_config = {
            "max_new_tokens": 128,
            "do_sample": False,
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
            max_length=self.max_input_tokens,
            return_token_type_ids=False,
        )

    def move_inputs_to_device(self, model_inputs: dict) -> dict:
        if self.device != "cuda":
            return model_inputs

        return {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in model_inputs.items()
        }

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
        total_chunks = (len(texts) + chunk_size - 1) // chunk_size
        heartbeat_every = max(
            1, int(os.getenv("GEN_HEARTBEAT_EVERY_CHUNKS", "1")))
        try:
            # Process in chunks to manage memory
            for i in range(0, len(texts), chunk_size):
                chunk_texts = texts[i: i + chunk_size]
                chunk_idx = (i // chunk_size) + 1
                chunk_start = perf_counter()

                # Tokenize chunk
                model_inputs = self.move_inputs_to_device(
                    self.tokenize(chunk_texts))
                if chunk_idx % heartbeat_every == 0:
                    token_count = int(model_inputs["input_ids"].shape[1])
                    logger.info(
                        "Generating sub-chunk %s/%s (batch=%s, prompt_tokens=%s)",
                        chunk_idx,
                        total_chunks,
                        len(chunk_texts),
                        token_count,
                    )

                # Generate responses
                generated_ids = self.model.generate(
                    **model_inputs, **self.generation_config
                )
                if chunk_idx % heartbeat_every == 0:
                    logger.info(
                        "Completed sub-chunk %s/%s in %.2fs",
                        chunk_idx,
                        total_chunks,
                        perf_counter() - chunk_start,
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
