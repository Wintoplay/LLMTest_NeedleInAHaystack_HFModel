from typing import Optional

from .model import ModelProvider

try:
    import hfer
    from hfer import LLModel, create_model, is_model
except Exception as e:
    print('Try to pip install hfer, check more: https://pypi.org/project/hfer/')
    raise e

class HFer(ModelProvider):

    DEFAULT_MODEL_KWARGS: dict = dict(max_new_tokens=50, temperature=0.1)

    def __init__(self, model_name, repo_or_path, model_kwargs: dict = DEFAULT_MODEL_KWARGS):
        if is_model(model_name):
            self.hf: LLModel = create_model(model_name, repo_or_path)
        else:
            raise ValueError(f"Invalid hf model name: {model_name}")

        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.tokenizer = self.hf.get_tokenizer()

    def evaluate_model(self, prompt: str, config: dict = {}) -> str:
        """
        Evaluates a given prompt using the Huggingface model and retrieves the model's response.

        Args:
            prompt (str): The prompt to send to the model.

        Returns:
            str: The content of the model's response to the prompt.
        """
        # update additional config
        config.update({k: v for k, v in self.model_kwargs.items()})

        response = self.hf.chat(messages=prompt, config=config)

        return response

    def generate_prompt(self, context: str, retrieval_question: str) -> str | list[dict[str, str]]:

        prompt_template = "{context}\n\nQuestion:\n{retrieval_question}\nAnswer:"

        return prompt_template.format(context=context, retrieval_question=retrieval_question)

    def encode_text_to_tokens(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode_tokens(self, tokens: list[int], context_length: Optional[int] = None) -> str:
        return self.tokenizer.decode(tokens[:context_length])
