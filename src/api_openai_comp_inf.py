import sys
from typing import List, Any
import openai
from openai import OpenAI
from openai import APIError
import numpy as np
import time


# --- ヘルパー関数 -----------------------------------------------------------------

def _get_model_name(settings: Any, prefix: str) -> str:
    """
    settingsオブジェクトとプレフィックスに基づき、モデル名を取得する。
    """
    return getattr(settings, f"{prefix}_model_name")

def _get_sampling_params(settings: Any, prefix: str) -> dict:
    """
    settingsオブジェクトとプレフィックスに基づき、OpenAI互換API用のサンプリングパラメータを生成する。
    """
    temperature = getattr(settings, f"{prefix}_temperature", 0.7)
    top_p = getattr(settings, f"{prefix}_top_p", 0.9)
    max_tokens = getattr(settings, "max_tokens", 4096)
    
    return {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }

def _execute_inference(model_name: str, prompts: List[str], options: dict, is_chat: bool, settings: Any, endpoint: str | None = None, api_key: str | None = None) -> List[str]:
    """
    OpenAI互換APIを使用して推論を実行する内部関数。
    """
    print(f"OpenAI互換モデル '{model_name}' を使用して{len(prompts)}件の推論を実行中...")
    results = []
    max_retries = 5
    base_delay = 1  # seconds

    client = OpenAI(
        base_url=endpoint or getattr(settings, "openai_comp_endpoint", "https://api.openai.com/v1"),
        api_key=api_key or getattr(settings, "openai_comp_api_key"),
    )

    for prompt in prompts:
        for attempt in range(max_retries):
            try:
                if is_chat:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=options.get("temperature"),
                        top_p=options.get("top_p"),
                        max_tokens=options.get("max_tokens"),
                        stop=options.get("stop"),
                    )
                    results.append(response.choices[0].message.content)
                else:
                    response = client.completions.create(
                        model=model_name,
                        prompt=prompt,
                        temperature=options.get("temperature"),
                        top_p=options.get("top_p"),
                        max_tokens=options.get("max_tokens"),
                        stop=options.get("stop"),
                    )
                    results.append(response.choices[0].text)
                break  # Success, break out of retry loop
            except APIError as e:
                if e.status_code == 429 and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"レート制限に達しました。{delay}秒待機してリトライします... (試行 {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                else:
                    print(f"OpenAI互換APIでの推論中にエラーが発生しました: {e}")
                    sys.exit(1)
            except Exception as e:
                print(f"予期せぬエラーが発生しました: {e}")
                sys.exit(1)
    print("推論が完了しました。")
    return results

# --- 1. モデルロード/アンロード系関数（ダミー） --------------------------------------------------
# OpenAI互換APIではクライアント側での明示的なロード/アンロードは不要なため、ダミー関数を定義します。

def inst_model_load(settings: Any) -> str:
    model_name = _get_model_name(settings, "Instruct")
    print(f"OpenAI互換モデル '{model_name}' を使用します。")
    return model_name

def base_model_load(settings: Any) -> str:
    model_name = _get_model_name(settings, "base")
    print(f"OpenAI互換モデル '{model_name}' を使用します。")
    return model_name

def think_model_load(settings: Any) -> str:
    model_name = _get_model_name(settings, "think")
    print(f"OpenAI互換モデル '{model_name}' を使用します。")
    return model_name

def unload_model(model_path_for_log: str):
    print(f"OpenAI互換モデル '{model_path_for_log}' のセッションを終了しました。")
    pass

# --- 2. 推論系関数（5種類） ------------------------------------------------------

def inst_model_inference(llm: str, prompts: List[str], settings: Any) -> List[str]:
    """
    Instructモデル（チャット形式）を用いてバッチ推論を行う。
    """
    model_name = llm
    options = _get_sampling_params(settings, "Instruct")
    return _execute_inference(model_name, prompts, options, is_chat=True, settings=settings)

def base_model_inference(llm: str, prompts: List[str], settings: Any) -> List[str]:
    """
    ベースモデル（生成形式）を用いてバッチ推論を行う。
    """
    model_name = llm
    options = _get_sampling_params(settings, "base")
    options["stop"] = ['<stop>']
    return _execute_inference(model_name, prompts, options, is_chat=False, settings=settings)

def think_model_inference(llm: str, prompts: List[str], settings: Any) -> List[str]:
    """
    長考モデル（チャット形式）を用いてバッチ推論を行う。
    """
    # settingsにthink_endpointがあるならそれを使用
    endpoint = getattr(settings, "think_endpoint", None)
    api_key = getattr(settings, "think_api_key", None)
    model_name = llm
    options = _get_sampling_params(settings, "think")
    return _execute_inference(model_name, prompts, options, is_chat=True, settings=settings, endpoint=endpoint, api_key=api_key)

def curation_model_inference(llm: str, prompts: List[str], settings: Any) -> List[str]:
    """
    キュレーション用の推論を行う。ベースモデルを使用するが、サンプリングパラメータが異なる。
    """
    model_name = llm
    options = _get_sampling_params(settings, "base")
    options["temperature"] = 0.01
    options["top_p"] = 1.0
    options["stop"] = ['<stop>']
    return _execute_inference(model_name, prompts, options, is_chat=False, settings=settings)

def evolution_model_inference(llm: str, prompts: List[str], settings: Any) -> List[str]:
    """
    思考を進化させるための推論を行う。ベースモデルを使用するが、サンプリングパラメータが異なる。
    """
    model_name = llm
    options = _get_sampling_params(settings, "base")
    options["temperature"] = 0.6
    options["top_p"] = 0.9
    options["stop"] = ['<stop>']
    return _execute_inference(model_name, prompts, options, is_chat=False, settings=settings)

# --- 埋め込み関数 -----------------------------------------------------------------

def get_embeddings(sentences: List[str], settings: Any) -> np.ndarray:
    if hasattr(settings, 'E5_path'):
        from sentence_transformers import SentenceTransformer
        model_path = settings.E5_path
        print(f"モデル '{model_path}' を読み込んでいます...")
        try:
            model = SentenceTransformer(model_path)
            return model.encode(['passage: ' + s for s in sentences], show_progress_bar=False)
        except Exception as e:
            print(f"モデル '{model_path}' の読み込みに失敗しました。パスが正しいか確認してください。エラー: {e}")
            sys.exit(1)
    model_name = getattr(settings, 'openai_E5_model_name', 'text-embedding-ada-002')
    print(f"OpenAI互換埋め込みモデル '{model_name}' を使用してベクトルを生成します。")
    try:
        client = OpenAI(
            base_url=getattr(settings, "openai_comp_endpoint", "https://api.openai.com/v1"),
            api_key=getattr(settings, "openai_comp_api_key"),
        )
        response = client.embeddings.create(
            model=model_name,
            input=sentences
        )
        if response and response.data:
            embeddings = [d.embedding for d in response.data]
            # for i, emb in enumerate(embeddings):
            #     print(f"Embedding {i} length: {len(emb)}")
            return np.array(embeddings)
        else:
            print(f"OpenAI互換APIからの予期せぬレスポンス形式です: {response}")
            sys.exit(1)
    except APIError as e:
        print(f"OpenAI互換APIでの埋め込み生成中にエラーが発生しました: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
        sys.exit(1)
