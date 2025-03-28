import re
import torch
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLM:

    def __init__(self, model_name: str, tokenizer_name: str, load_in_4bit: bool = False):
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name,
                                                          device_map="auto",
                                                          load_in_4bit=load_in_4bit)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def tokenize(self, texts: List[str]) -> List[torch.Tensor]:
        return self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)

    def get_response(self, generated_output: str) -> str:
        response = generated_output.split("### Response:")
        if response is not None and len(response) > 1:
            reuslts_extracted = re.findall(r"<(.*?)>", response[1])
            if len(reuslts_extracted) > 0:
                annotation = f"<{reuslts_extracted[0]}>"
                return annotation
        return generated_output

    @torch.inference_mode()
    def generate(self, texts: List[str]):
        model_inputs = self.tokenize(texts=texts)
        generated_ids = self.model.generate(
            **model_inputs, max_new_tokens=128)
        decoded_outputs = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True)
        responses = []
        for output in decoded_outputs:
            responses.append(self.get_response(generated_output=output))
        return responses
