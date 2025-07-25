import gc
import torch
import time
import platform
import os
from typing import List, Any

# --- vLLM or Transformers ---
IS_MACOS = platform.system() == "Darwin"

if not IS_MACOS:
    from vllm import LLM, SamplingParams
    from vllm.distributed import destroy_model_parallel
else:
    from transformers import AutoModelForCausalLM, AutoTokenizer

# --- ヘルパー関数 -----------------------------------------------------------------

def _load_model_from_settings(settings: Any, prefix: str) -> Any:
    """
    settingsオブジェクトとプレフィックスに基づき、vLLMモデルまたはHugging Face Transformersモデルをロードする。
    """
    model_name = getattr(settings, f"{prefix}_model_name")
    
    # Get the absolute path to the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the absolute path to the model
    model_path = os.path.join(script_dir, '..', model_name)

    if IS_MACOS:
        print(f"macOS環境を検出しました。'{model_path}' をHugging Face Transformersでロードします...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        print("モデルのロードが完了しました。")
        return model, tokenizer
    else:
        quantization = getattr(settings, f"{prefix}_model_quantization", None)
        gpu_memory_utilization = getattr(settings, f"{prefix}_gpu_memory_utilization", 0.9)
        dtype = getattr(settings, f"{prefix}_dtype", "auto")
        
        if quantization and str(quantization).lower() == 'null':
            quantization = None

        seed = getattr(settings, "seed", int(time.time()))

        print(f"モデル '{model_path}' をロードしています...")
        print(f"  - quantization: {quantization}")
        print(f"  - tensor_parallel_size: {settings.tensor_parallel_size}")
        print(f"  - seed: {seed}")

        llm = LLM(
            model=model_path,
            quantization=quantization,
            tensor_parallel_size=settings.tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            trust_remote_code=settings.trust_remote_code,
            max_model_len=settings.max_tokens,
            seed=seed
        )
        print("モデルのロードが完了しました。")
        return llm

def _execute_inference_with_error_handling(
    model: Any, 
    prompts: List[str], 
    settings: Any,
    prefix: str,
    tokenizer: Any = None
) -> List[str]:
    """
    例外処理を強化した推論実行の内部関数。
    """
    print(f"{len(prompts)}件のバッチ推論を実行中...")
    try:
        if IS_MACOS:
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
            outputs = model.generate(**inputs, max_new_tokens=settings.max_tokens)
            results = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        else:
            sampling_params = SamplingParams(
                temperature=getattr(settings, f"{prefix}_temperature"),
                top_p=getattr(settings, f"{prefix}_top_p"),
                max_tokens=settings.max_tokens,
                seed=getattr(settings, "seed", int(time.time())),
                stop=[model.get_tokenizer().eos_token] if prefix != "base" else ["<stop>"]
            )
            outputs = model.generate(prompts, sampling_params)
            results = [output.outputs[0].text for output in outputs]
        
        print("バッチ推論が完了しました。")
        return results
    except Exception as e:
        print(f"推論中にエラーが発生しました: {e}")
        return [""] * len(prompts)
    finally:
        if not IS_MACOS:
            gc.collect()
            torch.cuda.empty_cache()

def unload_model(model_path_for_log: str):
    """
    モデルオブジェクトを削除した後に、GPUメモリを解放する。
    """
    if not IS_MACOS:
        destroy_model_parallel()
        gc.collect()
        torch.cuda.empty_cache()
        print(f"モデル '{model_path_for_log}' に関連するリソースを解放し、GPUキャッシュをクリアしました。")
    else:
        print(f"macOS環境では、モデル '{model_path_for_log}' のリソース解放は自動的に行われます。")


# --- 1. モデルロード系関数（3種類） --------------------------------------------------

def inst_model_load(settings: Any) -> Any:
    """
    settings.yamlから 'Instruct_' プレフィックスの設定を読み込み、Instructモデルをロードする。
    """
    return _load_model_from_settings(settings, "Instruct")

def base_model_load(settings: Any) -> Any:
    """
    settings.yamlから 'base_' プレフィックスの設定を読み込み、ベースモデルをロードする。
    """
    return _load_model_from_settings(settings, "base")

def think_model_load(settings: Any) -> Any:
    """
    settings.yamlから 'think_' プレフィックスの設定を読み込み、長考モデルをロードする。
    """
    return _load_model_from_settings(settings, "think")


# --- 2. 推論系関数（5種類） ------------------------------------------------------

def inst_model_inference(model_and_tokenizer: Any, prompts: List[str], settings: Any) -> List[str]:
    """
    Instructモデルを用いてバッチ推論を行う。プロンプトはChatML形式に変換される。
    """
    if IS_MACOS:
        model, tokenizer = model_and_tokenizer
        formatted_prompts = []
        for p in prompts:
            messages = [{"role": "user", "content": p}]
            prompt_str = tokenizer.apply_chat_template(
                conversation=messages, tokenize=False, add_generation_prompt=True
            )
            formatted_prompts.append(prompt_str)
        return _execute_inference_with_error_handling(model, formatted_prompts, settings, "Instruct", tokenizer)
    else:
        llm = model_and_tokenizer
        tokenizer = llm.get_tokenizer()
        formatted_prompts = []
        for p in prompts:
            messages = [{"role": "user", "content": p}]
            prompt_str = tokenizer.apply_chat_template(
                conversation=messages, tokenize=False, add_generation_prompt=True
            )
            formatted_prompts.append(prompt_str)
        return _execute_inference_with_error_handling(llm, formatted_prompts, settings, "Instruct")


def base_model_inference(model_and_tokenizer: Any, prompts: List[str], settings: Any) -> List[str]:
    """
    ベースモデルを用いてバッチ推論を行う。
    """
    if IS_MACOS:
        model, tokenizer = model_and_tokenizer
        return _execute_inference_with_error_handling(model, prompts, settings, "base", tokenizer)
    else:
        llm = model_and_tokenizer
        return _execute_inference_with_error_handling(llm, prompts, settings, "base")


def think_model_inference(model_and_tokenizer: Any, prompts: List[str], settings: Any) -> List[str]:
    """
    長考モデルを用いてバッチ推論を行う。出力には<think>タグが含まれる。
    """
    if IS_MACOS:
        model, tokenizer = model_and_tokenizer
        formatted_prompts = []
        for p in prompts:
            messages = [{"role": "user", "content": p}]
            prompt_str = tokenizer.apply_chat_template(
                conversation=messages, tokenize=False, add_generation_prompt=True
            )
            formatted_prompts.append(prompt_str)
        return _execute_inference_with_error_handling(model, formatted_prompts, settings, "think", tokenizer)
    else:
        llm = model_and_tokenizer
        tokenizer = llm.get_tokenizer()
        formatted_prompts = []
        for p in prompts:
            messages = [{"role": "user", "content": p}]
            prompt_str = tokenizer.apply_chat_template(
                conversation=messages, tokenize=False, add_generation_prompt=True
            )
            formatted_prompts.append(prompt_str)
        return _execute_inference_with_error_handling(llm, formatted_prompts, settings, "think")
