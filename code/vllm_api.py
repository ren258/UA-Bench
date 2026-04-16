import openai
import httpx
import time
import os
from openai import APITimeoutError

import itertools
import threading


MODEL_NAME = "my_model"

BACKENDS = [
    {"base_url": "http://localhost:11546/v1", "model": f"{MODEL_NAME}-gpu0"},
    {"base_url": "http://localhost:11547/v1", "model": f"{MODEL_NAME}-gpu1"},
    {"base_url": "http://localhost:11548/v1", "model": f"{MODEL_NAME}-gpu2"},
    {"base_url": "http://localhost:11549/v1", "model": f"{MODEL_NAME}-gpu3"},
    {"base_url": "http://localhost:11550/v1", "model": f"{MODEL_NAME}-gpu4"},
    {"base_url": "http://localhost:11551/v1", "model": f"{MODEL_NAME}-gpu5"},
    {"base_url": "http://localhost:11552/v1", "model": f"{MODEL_NAME}-gpu6"},
    {"base_url": "http://localhost:11553/v1", "model": f"{MODEL_NAME}-gpu7"},
]

_backend_cycle = itertools.cycle(BACKENDS)
_backend_lock = threading.Lock()

def _pick_backend():
    with _backend_lock:
        return next(_backend_cycle)
    

def openai_call(prompt, temperature=0.0, model="qwen3-8b"):
    backend = _pick_backend()

    base_url = backend["base_url"]
    
    model = MODEL_NAME
    model = backend["model"]

    client = openai.OpenAI(
        base_url=base_url,
        api_key="",
    )

    messages =[{'role': 'system', 'content': 'You are an intelligent AI assistant.'},
                    {'role': 'user', 'content': prompt}]
    
    timeout_seconds = 6000
    max_tokens = 39000

    fails = 0
    while True:
        if fails > 30:
            print('Failed too many times, exiting...')
            return None
        try:
            completion = client.chat.completions.create(
                model = model,
                messages = messages,
                temperature = temperature,
                timeout = timeout_seconds,
                max_tokens = max_tokens,
                # extra_body={
                #     "chat_template_kwargs": {"enable_thinking": False},
                # },
            )
            response = completion.choices[0].message.content
            break
        except APITimeoutError as e:
            print(f"[Timeout after {timeout_seconds}s] {e}")
            return None
        except Exception as e:
            print(f'Error: {e}')
            print("Exception type:", type(e))
            print("Exception repr:", repr(e))
            fails += 1
            time.sleep(fails * 0.1 + 1)
    return response

if __name__ == "__main__":
    prompt = "Who are you?"
    response = openai_call(prompt)
    
    print(response)
