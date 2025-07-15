from .util_solid import Singleton
from .util_torch import getTorchDevice
import torch
import time
import typing
import os
import re
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    pipeline,
    DynamicCache,
)

device = getTorchDevice()


class LlmApi:
    def __init__(self, modelName):
        self.modelName = modelName

    def instruct(self, prompt: str, callback=None): ...


try:

    import ollama

    class LlmApiOllama(LlmApi):

        def instruct(self, prompt: str, callback=None):
            response = ollama.generate(model=self.modelName, prompt=prompt, stream=True)
            resultJoiner = []
            for chunk in response:
                result = chunk["response"]
                resultJoiner.append(result)
                if callback:
                    callback(result)
            return "".join(resultJoiner)

        def chat(self, prompt: list[dict[str, typing.Any]], callback=None):
            response = ollama.chat(model=self.modelName, messages=prompt, stream=True)
            resultJoiner = []
            for chunk in response:
                result = chunk["message"]["content"]
                resultJoiner.append(result)
                if callback:
                    callback(result)
            return "".join(resultJoiner)

except ImportError:
    ...


class LlmApiHuggingface(LlmApi):
    ModelDir = None

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        if self.ModelDir is not None:
            self.modelName = os.path.join(self.ModelDir, self.modelName)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.modelName,
            # load_in_4bit=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.modelName,
            # load_in_4bit=True,
        )

    def applyChatTemplate(self, prompt):
        return self.tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": "",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    def instruct(self, prompt: str, callback=None):
        if callback:
            # return pipeline(
            #     "text-generation",
            #     model=self.model,
            #     tokenizer=self.tokenizer,
            #     device=device,
            #     stream_complete=True,
            # )(prompt, callback=callback)
            print("not support stream")
        model_inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        t0 = time.perf_counter()
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=1024,
            #     num_beams=3,
            #     no_repeat_ngram_size=5,
            #     early_stopping=True,
        )
        t1 = time.perf_counter()
        generated_ids = generated_ids[:, model_inputs["input_ids"].shape[1] :]
        ret = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        print(f"time cost {t1-t0:.3f}")
        ret = ret[0]
        return ret

    def instruct_stream(self, prompt: str, callback=None):
        model_inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        tokens = model_inputs["input_ids"]
        t0 = time.perf_counter()
        kv = DynamicCache()
        tkBuf = ""
        while True:
            out = self.model(
                input_ids=tokens,
                past_key_values=kv,
                use_cache=True,
            )
            next_token = torch.multinomial(
                torch.softmax(out.logits[0, -1, :], dim=-1), 1
            ).item()
            kv = out.past_key_values

            if callback:
                tk = self.tokenizer.decode([next_token], skip_special_tokens=True)
                tkBuf += tk
                callback(tk)
            if next_token == self.model.config.eos_token_id:
                break
            tokens = torch.tensor([[next_token]]).to(tokens.device)
        t1 = time.perf_counter()
        print(f"time cost {t1-t0:.3f}")
        ret = tkBuf
        return ret


LLMApiUsing = LlmApiOllama


@Singleton
class Llm(LLMApiUsing): ...


def remove_thinking(s):
    return re.sub(r"\s*<think>.*</think>\s*", "", s, flags=re.DOTALL).strip()


def extract_codeblock(s, lang):
    return re.sub(
        rf"^.*?```{lang}(?P<c>.*)```.*$", lambda m: m.group("c"), s, flags=re.DOTALL
    ).strip()
