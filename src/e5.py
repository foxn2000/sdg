import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Union, Optional, Any
import ollama
import sys
from ollama import ResponseError
import openai # openaiライブラリをインポート

def reduce_data_by_diversity(
    data: Union[List[str], List[Dict]],
    data_format: str,
    cut_rate: float,
    settings: Any,
    data_key: Optional[str] = None,
) -> Union[List[str], List[Dict]]:
    """
    データの多様性に基づいて、類似度の高いデータから指定された割合を削減する関数。
    推論バックエンドの設定に応じて、OllamaまたはローカルのSentenceTransformerを使い分ける。
    """
    # --- 1. 入力値の検証 ---
    if not (0 <= cut_rate <= 100):
        raise ValueError("カット率（cut_rate）は0から100の間で指定してください。")
    if data_format not in ['list', 'dict']:
        raise ValueError("データ形式（data_format）は 'list' または 'dict' を指定してください。")
    if data_format == 'dict' and not data_key:
        raise ValueError("データ形式が 'dict' の場合、データキー（data_key）の指定は必須です。")
    if not data or len(data) < 2:
        return data

    # --- 2. 処理対象の文章リストを抽出 ---
    if data_format == 'list':
        sentences = data
    else: # 'dict'
        try:
            sentences = [item[data_key] for item in data]
        except KeyError:
            raise KeyError(f"指定されたキー '{data_key}' が一部の辞書に存在しません。")
        except TypeError:
            raise TypeError("データ形式が 'dict' の場合、入力データは辞書のリストである必要があります。")

    # --- 3. 削減するデータ数を計算 ---
    num_to_cut = int(len(sentences) * (cut_rate / 100))
    if num_to_cut == 0:
        return data
    if num_to_cut >= len(sentences):
        return []

    # --- 4. バックエンドに応じてエンコード処理を分岐 ---
    backend = getattr(settings, 'inference_backend', 'vllm')
    embeddings = None
    ollama_url = getattr(settings, "ollama_url", "http://localhost:11434")

    if backend == 'ollama':
        model_name = getattr(settings, 'E5_model_name', 'mxbai-embed-large')
        print(f"埋め込みモデル '{model_name}' を使用してベクトルを生成します (URL: {ollama_url})。")
        try:
            if "/v1" in ollama_url: # LM StudioなどのOpenAI互換APIを検出
                client = openai.OpenAI(base_url=ollama_url, api_key="lm-studio") # api_keyはダミーでOK
                response = client.embeddings.create(
                    model=model_name,
                    input=sentences
                )
                embedding_list = [d.embedding for d in response.data]
                embeddings = np.array(embedding_list)
            else: # OllamaネイティブAPI
                response = ollama.embed(model=model_name, input=sentences)
                
                # responseオブジェクトからembeddings属性を取得しようと試みる
                embedding_list = getattr(response, 'embeddings', None)

                # 古いライブラリとの後方互換性のため、辞書形式もチェック
                if embedding_list is None and isinstance(response, dict) and 'embeddings' in response:
                    embedding_list = response['embeddings']

                if embedding_list:
                    embeddings = np.array(embedding_list)
                else:
                    print(f"Ollamaからの予期せぬレスポンス形式です: {response}")
                    sys.exit(1)
        except ResponseError as e:
            print(f"Ollamaでの埋め込み生成中にエラーが発生しました: {e.error}")
            if e.status_code == 404:
                print(f"エラー: 埋め込みモデル '{model_name}' が見つかりません。")
                print(f"解決策: 'ollama pull {model_name}' を実行してモデルをダウンロードしてください。")
                sys.exit(1)
            sys.exit(1)
        except openai.APIStatusError as e:
            print(f"OpenAI互換APIでの埋め込み生成中にエラーが発生しました: {e.status_code} - {e.response}")
            sys.exit(1)
        except Exception as e:
            print(f"予期せぬエラーが発生しました: {e}")
            sys.exit(1)
    else: # vLLM or default
        model_path = settings.E5_path
        print(f"モデル '{model_path}' を読み込んでいます...")
        try:
            model = SentenceTransformer(model_path)
            # 参考コードと同様に、検索タスク用のプレフィックスを付与
            embeddings = model.encode(['passage: ' + s for s in sentences], show_progress_bar=False)
        except Exception as e:
            print(f"モデル '{model_path}' の読み込みに失敗しました。パスが正しいか確認してください。エラー: {e}")
            # エラーが発生した場合、多様性フィルタをスキップして元のデータを返す
            return data

    # --- 5. 類似度に基づいて削減対象を選定 ---
    if embeddings is not None:
        print("類似度を計算し、削減対象を選定中...")
        similarity_matrix = cosine_similarity(embeddings)
        np.fill_diagonal(similarity_matrix, -1)
        redundancy_scores = np.max(similarity_matrix, axis=1)
        indices_to_cut = np.argsort(-redundancy_scores)[:num_to_cut]
        indices_to_cut_set = set(indices_to_cut)

        # --- 6. 削減後のデータを生成 ---
        reduced_data = [item for i, item in enumerate(data) if i not in indices_to_cut_set]
        
        print(f"処理が完了しました。{len(data)}件のデータから{len(indices_to_cut)}件を削減し、{len(reduced_data)}件になりました。")
        return reduced_data
    else:
        # エンべディングが生成されなかった場合は元のデータを返す
        print("警告: エンべディングが生成されなかったため、多様性フィルタリングをスキップしました。")
        return data

