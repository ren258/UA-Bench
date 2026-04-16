import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


API_MODE = os.getenv("LLM_API_MODE", "network").lower()
assert API_MODE in ("network", "local"), f"Unknown LLM_API_MODE: {API_MODE}"


def parallel_openai_calls(
    prompts,
    temperature=0.0,
    model="gpt-4o-2024-11-20",
    max_workers=128,
):
    if API_MODE == "network":
        from gpt_api import openai_call
    else:
        from vllm_api import openai_call
    
    print(f"[parallel_openai_calls] API_MODE={API_MODE}, model={model}, temperature={temperature}, num_prompts={len(prompts)}")

    results = [None] * len(prompts)

    if API_MODE == "network":

        def _call_single(i, prompt):
            try:
                return i, openai_call(
                    prompt,
                    temperature=temperature,
                    model=model,
                )
            except Exception as e:
                return i, f"ERROR: {e}"

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_call_single, i, prompt)
                for i, prompt in enumerate(prompts)
            ]

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Parallel API calls (network)",
            ):
                idx, output = future.result()
                results[idx] = output

    else:
        def _call_single(i, prompt):
            try:
                return i, openai_call(
                    prompt,
                    temperature=temperature,
                    model=model,
                )
            except Exception as e:
                return i, f"ERROR: {e}"

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_call_single, i, prompt)
                for i, prompt in enumerate(prompts)
            ]

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Parallel API calls (local vLLM)",
            ):
                idx, output = future.result()
                results[idx] = output

    return results
