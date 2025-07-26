import ollama
import sys
from typing import List, Any
from ollama import ResponseError
import openai # openaiライブラリをインポート

# --- ヘルパー関数 -----------------------------------------------------------------

def _get_model_name(settings: Any, prefix: str) -> str:
    """
    settingsオブジェクトとプレフィックスに基づき、モデル名を取得する。
    """
    return getattr(settings, f"{prefix}_model_name")

def _get_sampling_params(settings: Any, prefix: str) -> dict:
    """
    settingsオブジェクトとプレフィックスに基づき、Ollama用のサンプリングパラメータを生成する。
    """
    temperature = getattr(settings, f"{prefix}_temperature", 0.7)
    top_p = getattr(settings, f"{prefix}_top_p", 0.9)
    max_tokens = getattr(settings, "max_tokens", 4096)
    seed = getattr(settings, "seed", 42)
    
    return {
        "temperature": temperature,
        "top_p": top_p,
        "num_predict": max_tokens,
        "seed": seed,
    }

def _execute_inference(model_name: str, prompts: List[str], options: dict, is_chat: bool, ollama_url: str, settings: Any) -> List[str]:
    """
    OllamaまたはOpenAI互換APIを使用して推論を実行する内部関数。
    """
    results = []
    backend_name = getattr(settings, 'inference_backend', 'ollama')
    print(f"モデル '{model_name}' を使用して{len(prompts)}件の推論を実行中 (URL: {ollama_url}, バックエンド: {backend_name})...")

    try:
        if backend_name == "lmstudio": # LM StudioなどのOpenAI互換APIを検出
            client = openai.OpenAI(base_url=ollama_url, api_key="lm-studio") # api_keyはダミーでOK
            for prompt in prompts:
                if is_chat:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=options.get("temperature"),
                        top_p=options.get("top_p"),
                        max_tokens=options.get("num_predict"),
                        seed=options.get("seed"),
                        stop=options.get("stop")
                    )
                    results.append(response.choices[0].message.content)
                else:
                    # LM Studioの/v1/completionsエンドポイントは非推奨だが、互換性のため残す
                    response = client.completions.create(
                        model=model_name,
                        prompt=prompt,
                        temperature=options.get("temperature"),
                        top_p=options.get("top_p"),
                        max_tokens=options.get("num_predict"),
                        seed=options.get("seed"),
                        stop=options.get("stop")
                    )
                    results.append(response.choices[0].text)
        else: # OllamaネイティブAPI
            client = ollama.Client(host=ollama_url)
            for prompt in prompts:
                if is_chat:
                    response = client.chat(
                        model=model_name,
                        messages=[{'role': 'user', 'content': prompt}],
                        options=options
                    )
                    results.append(response['message']['content'])
                else:
                    response = client.generate(
                        model=model_name,
                        prompt=prompt,
                        options=options
                    )
                    results.append(response['response'])
        print("推論が完了しました。")
        return results
    except ResponseError as e:
        print(f"Ollamaでの推論中にエラーが発生しました: {e.error}")
        if e.status_code == 404:
            print(f"エラー: モデル '{model_name}' が見つかりません。")
            print(f"解決策: 'ollama pull {model_name}' を実行してモデルをダウンロードしてください。")
            sys.exit(1)
        return [""] * len(prompts)
    except openai.APIStatusError as e:
        print(f"OpenAI互換APIでの推論中にエラーが発生しました: {e.status_code} - {e.response}")
        sys.exit(1)
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
        sys.exit(1)

# --- 1. モデルロード/アンロード系関数（ダミー） --------------------------------------------------
# Ollamaではクライアント側での明示的なロード/アンロードは不要なため、ダミー関数を定義します。

def inst_model_load(settings: Any) -> str:
    model_name = _get_model_name(settings, "Instruct")
    ollama_url = getattr(settings, "ollama_url", "http://localhost:11434")
    backend_name = getattr(settings, 'inference_backend', 'ollama')
    print(f"モデル '{model_name}' を使用します (URL: {ollama_url}, バックエンド: {backend_name})。")
    return model_name

def base_model_load(settings: Any) -> str:
    model_name = _get_model_name(settings, "base")
    ollama_url = getattr(settings, "ollama_url", "http://localhost:11434")
    backend_name = getattr(settings, 'inference_backend', 'ollama')
    print(f"モデル '{model_name}' を使用します (URL: {ollama_url}, バックエンド: {backend_name})。")
    return model_name

def think_model_load(settings: Any) -> str:
    model_name = _get_model_name(settings, "think")
    ollama_url = getattr(settings, "ollama_url", "http://localhost:11434")
    backend_name = getattr(settings, 'inference_backend', 'ollama')
    print(f"モデル '{model_name}' を使用します (URL: {ollama_url}, バックエンド: {backend_name})。")
    return model_name

def unload_model(model_path_for_log: str, settings: Any):
    backend_name = getattr(settings, 'inference_backend', 'ollama')
    print(f"{backend_name}モデル '{model_path_for_log}' のセッションを終了しました。")
    pass

# --- 2. 推論系関数（5種類） ------------------------------------------------------

def inst_model_inference(llm: str, prompts: List[str], settings: Any) -> List[str]:
    """
    Instructモデル（チャット形式）を用いてバッチ推論を行う。
    """
    model_name = llm
    options = _get_sampling_params(settings, "Instruct")
    ollama_url = getattr(settings, "ollama_url", "http://localhost:11434")
    return _execute_inference(model_name, prompts, options, is_chat=True, ollama_url=ollama_url, settings=settings)

def base_model_inference(llm: str, prompts: List[str], settings: Any) -> List[str]:
    """
    ベースモデル（生成形式）を用いてバッチ推論を行う。
    """
    model_name = llm
    options = _get_sampling_params(settings, "base")
    options["stop"] = ['<stop>']
    ollama_url = getattr(settings, "ollama_url", "http://localhost:11434")
    return _execute_inference(model_name, prompts, options, is_chat=False, ollama_url=ollama_url, settings=settings)

def think_model_inference(llm: str, prompts: List[str], settings: Any) -> List[str]:
    """
    長考モデル（チャット形式）を用いてバッチ推論を行う。
    """
    model_name = llm
    options = _get_sampling_params(settings, "think")
    ollama_url = getattr(settings, "ollama_url", "http://localhost:11434")
    return _execute_inference(model_name, prompts, options, is_chat=True, ollama_url=ollama_url, settings=settings)

def curation_model_inference(llm: str, prompts: List[str], settings: Any) -> List[str]:
    """
    キュレーション用の推論を行う。ベースモデルを使用するが、サンプリングパラメータが異なる。
    """
    model_name = llm
    options = _get_sampling_params(settings, "base")
    options["temperature"] = 0.01
    options["top_p"] = 1.0
    options["stop"] = ['<stop>']
    ollama_url = getattr(settings, "ollama_url", "http://localhost:11434")
    return _execute_inference(model_name, prompts, options, is_chat=False, ollama_url=ollama_url, settings=settings)

def evolution_model_inference(llm: str, prompts: List[str], settings: Any) -> List[str]:
    """
    思考を進化させるための推論を行う。ベースモデルを使用するが、サンプリングパラメータが異なる。
    """
    model_name = llm
    options = _get_sampling_params(settings, "base")
    options["temperature"] = 0.6
    options["top_p"] = 0.9
    options["stop"] = ['<stop>']
    ollama_url = getattr(settings, "ollama_url", "http://localhost:11434")
    return _execute_inference(model_name, prompts, options, is_chat=False, ollama_url=ollama_url, settings=settings)
