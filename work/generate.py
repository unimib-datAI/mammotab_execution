from typing import List
from openai import OpenAI
import re
import logging
from time import sleep

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLM:
    def __init__(self,
                 model_name: str,
                 api_key: str,
                 base_url: str = None,
                 max_retries: int = 3,
                 retry_delay: int = 5):

        self.model_name = model_name
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url if base_url else "https://api.openai.com/v1"
        )
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.response_pattern = re.compile(r"<(.*?)>")
        self.split_pattern = re.compile(r"### Response:")

    def get_response(self, generated_output: str) -> str:
        """Extract the response from the generated output"""
        response = self.split_pattern.split(generated_output)
        if len(response) > 1:
            result = self.response_pattern.search(response[1])
            if result:
                return f"<{result.group(1)}>"
        return generated_output

    def generate(self, prompts: List[str]) -> List[str]:
        """Generate responses for a batch of prompts"""
        responses = []
        error_flags = []

        for prompt in prompts:
            for attempt in range(self.max_retries):
                try:
                    chat_completion = self.client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant"},
                            {"role": "user", "content": prompt}
                        ],
                        model=self.model_name,
                        temperature=0.7,
                        max_tokens=128
                    )

                    response = chat_completion.choices[0].message.content
                    processed_response = self.get_response(response)
                    responses.append(processed_response)
                    error_flags.append(False)
                    break

                except Exception as e:
                    if attempt == self.max_retries - 1:
                        logger.error(
                            f"Failed after {self.max_retries} attempts for prompt: {prompt[:100]}...")
                        responses.append("")  # Append empty string on failure
                        error_flags.append(True)
                        break
                    logger.warning(
                        f"Attempt {attempt + 1} failed, retrying in {self.retry_delay} seconds...")
                    sleep(self.retry_delay)

        return responses, error_flags
