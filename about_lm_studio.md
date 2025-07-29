# LM Studio を使用した推論ガイド

このドキュメントでは、`sdg` プロジェクトで推論バックエンドとして LM Studio を利用する方法について詳しく説明します。

LM Studio は、ローカルマシン上で大規模言語モデル（LLM）を簡単に検索、ダウンロード、実行できるデスクトップアプリケーションです。GUIを通じてモデルを管理し、数クリックでOpenAI互換のAPIサーバーを起動できるため、GPU環境の構築に慣れていない方や、手軽に様々なモデルを試したい場合に非常に便利です。

## 1. セットアップ

### 1.1. LM Studioのインストール

1.  **LM Studio公式サイト** にアクセスします。
2.  お使いのOS（Windows, macOS, Linux）用のインストーラーをダウンロードし、アプリケーションをインストールします。

### 1.2. モデルのダウンロードとサーバーの起動

1.  **LM Studioを起動**します。
2.  画面上部の検索バー（🔍 Search）で、利用したいモデルを検索します。
    *   **GGUF形式**のモデルが推奨されます。
    *   例: `google/gemma-2-9b-it-gguf`, `microsoft/Phi-3-mini-4k-instruct-gguf`
3.  検索結果からモデルを選択し、右側のファイルリストから適切な量子化バージョン（例: `Q4_K_M`）を選んで `Download` ボタンをクリックします。
4.  ダウンロードが完了したら、左側のメニューから **🖥️ Local Server** タブに移動します。
5.  `Select a model to load` ドロップダウンから、先ほどダウンロードしたモデルを選択します。
6.  **`Start Server`** ボタンをクリックして、ローカルAPIサーバーを起動します。サーバーが起動すると、ログに `URL: http://localhost:1234/v1` のようなエンドポイントが表示されます。

## 2. プロジェクトの設定

LM Studioを推論バックエンドとして使用するには、設定ファイルを編集する必要があります。

### 2.1. メイン設定ファイルの変更

まず、プロジェクトのルートにある `settings.yaml` を開き、推論バックエンドを `lmstudio` に指定します。

```yaml
# settings.yaml

# --- 推論バックエンドの指定 ---
# "vllm" から "lmstudio" に変更します
inference_backend: "lmstudio"

# ... その他の設定 ...
```

### 2.2. LM Studio専用設定ファイルの編集

次に、`settings_lmstudio.yaml` を開き、ご自身の環境に合わせて設定を編集します。

#### `settings_lmstudio.yaml`の設定例
```yaml
# --- LM Studioの設定 ---
# LM StudioサーバーのエンドポイントURLを指定します。
# 通常はデフォルトのままで問題ありません。
ollama_url: "http://localhost:1234/v1"
inference_backend: "lmstudio"

# --- 使用するモデル名の設定 ---
# LM Studioでロードしたモデルの「リポジトリ名/モデル名」を指定します。
# この名前は、LM Studioのサーバーログやモデル選択ドロップダウンで確認できます。
Instruct_model_name: "google/gemma-3-1b"
base_model_name: "qwen/qwen3-1.7b"
think_model_name: "deepseek-r1-distill-qwen-1.5b"

# --- サンプリングパラメータ ---
# モデルごとの生成パラメータを調整できます。
Instruct_temperature: 0.7
Instruct_top_p: 0.6
# ...

# --- バッチサイズ ---
# LM StudioはvLLMより推論速度が遅い場合があるため、バッチサイズは小さめ（例: 1〜8）を推奨します。
batch_size: 8
```

**重要:** `*_model_name` には、LM Studioが認識するモデルの識別子を指定します。これは通常、Hugging Faceのリポジトリ名に似ていますが、LM Studio上で表示される名前を正確に入力してください。

## 3. パイプラインの実行

上記の設定が完了したら、以下のコマンドでデータ生成パイプラインを実行できます。

1.  **LM Studioでサーバーが起動していることを確認**してください。
2.  ターミナルで以下のコマンドを実行します。

```bash
python main.py --config settings_lmstudio.yaml
```

これにより、`settings_lmstudio.yaml` の設定に基づいて、LM Studioサーバーにリクエストが送られ、質問生成や回答生成が実行されます。

## 4. トラブルシューティング

-   **接続エラーが発生する:**
    -   LM Studioでサーバーが正しく起動しているか確認してください。
    -   `settings_lmstudio.yaml` の `ollama_url` が、LM Studioのサーバーログに表示されているURLと一致しているか確認してください。
-   **モデルが見つからない (404 Not Found) エラーが発生する:**
    -   LM Studioのサーバータブで、モデルが正しくロードされているか確認してください。
    -   `settings_lmstudio.yaml` の `*_model_name` が、LM Studioでロードしたモデルの識別子と完全に一致しているか確認してください。
-   **推論が非常に遅い:**
    -   `settings_lmstudio.yaml` の `batch_size` を小さい値（例: `1`）に設定してみてください。
    -   LM Studioのサーバー設定で、GPUオフロードが有効になっているか確認してください（対応GPUがある場合）。