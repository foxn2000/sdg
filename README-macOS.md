# 環境構築ガイド (macOS)

このガイドは、macOS (Apple Silicon / Intel) 環境で `sdg` をセットアップし、実行するための手順を解説します。

## macOSでの実行環境

macOS環境では、NVIDIA GPU (CUDA) を必要とする `vLLM` バックエンドはサポートされていません。そのため、CPUやApple SiliconのGPU (Metal) を利用できる **Ollamaバックエンドを使用します。**

**Ollamaバックエンドの特徴**:
*   セットアップが比較的簡単です。
*   Apple Silicon (M1/M2/M3) のパフォーマンスを活かすことができます。
*   **要件**: [Ollama](https://ollama.com/) がインストールされ、実行中であること。

---

## セットアップ手順

### 1. OllamaとMinicondaのインストール

**Ollama:**
[Ollamaの公式サイト](https://ollama.com/)からmacOS版をダウンロードし、インストールします。アプリケーションを起動すると、メニューバーにOllamaのアイコンが表示されます。

**Miniconda:**
Python環境の管理にはMinicondaを使用します。お使いのMacのCPUに合わせてインストーラをダウンロードしてください。
```bash
# Apple Silicon (M1/M2/M3) の場合
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh

# Intel CPU の場合
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh

# シェルを再起動するか、source ~/.zshrc (または .bashrc) を実行してcondaコマンドを有効化
source ~/.zshrc
```

### 2. プロジェクトのクローンと環境構築

```bash
git clone https://github.com/foxn2000/sdg.git
cd sdg

# 'sdg'という名前でPython 3.11のconda環境を作成
conda create -n sdg python=3.11 -y
# 作成した環境を有効化
conda activate sdg
```

### 3. 依存ライブラリのインストール

`requirements.txt` にはmacOSでは不要なライブラリも含まれているため、Ollamaに必要なものだけを個別にインストールします。

```bash
pip install ollama sentence_transformers wordfreq fugashi
```
*`sentence_transformers`のインストール中に`tokenizers`のビルドで時間がかかる場合があります。*

### 4. モデルの準備

`settings_ollama.yaml` で指定されているモデルを `pull` します。ターミナルで以下のコマンドを実行してください。

```bash
# 例: gemma:2b-instruct の場合
ollama pull gemma:2b-instruct
```
`settings_ollama.yaml` に記載されている他のモデル（`base_model_name`, `think_model_name`）も同様に `pull` してください。

また、多様性フィルタで使用する `E5` モデルは、別途ダウンロードが必要です。

```bash
# git-lfsのインストールが必要な場合があります (brew install git-lfs)
git lfs install
git clone https://huggingface.co/intfloat/multilingual-e5-large ./data/model/multilingual-e5-large
```

### 5. 実行

Ollama用の設定ファイル `settings_ollama.yaml` を使って実行します。

```bash
python main.py --config settings_ollama.yaml
```

処理が完了すると、`output/final_dataset.jsonl` にデータセットが生成されます。
