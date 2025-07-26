# 環境構築ガイド (Linux)

このガイドは、Linux (Ubuntuを想定) 環境で `sdg` をセットアップし、実行するための手順を解説します。

## 2つの実行環境

`sdg` は、使用するAIの「頭脳」である推論バックエンドを2種類から選択できます。

1.  **vLLM バックエンド (NVIDIA GPU必須)**
    *   **特徴**: NVIDIA製のグラフィックボード（GPU）のパワーを最大限に活用し、高速に大量のデータを生成します。
    *   **要件**: NVIDIAドライバ、CUDA Toolkitがインストールされた環境。

2.  **Ollama バックエンド**
    *   **特徴**: セットアップが比較的簡単で、NVIDIA製GPUがない環境（CPUのみ、またはAMD製GPUなど）でも動作します。
    *   **要件**: [Ollama](https://ollama.com/) がインストールされ、実行中であること。

---

## A. vLLM環境のセットアップ (NVIDIA GPUをお持ちの方向け)

### 1. 前提条件

*   NVIDIAドライバがインストール済みであること。
*   CUDA Toolkit 12.1 以上がインストール済みであること。

以下のコマンドで確認できます。
```bash
nvidia-smi
nvcc --version
```

### 2. Minicondaのインストール

Python環境の管理にはMinicondaを使用します。

```bash
# インストーラのダウンロード
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# インストールの実行
bash Miniconda3-latest-Linux-x86_64.sh
# シェルを再起動するか、source ~/.bashrc を実行してcondaコマンドを有効化
source ~/.bashrc
```

### 3. プロジェクトのクローンと環境構築

```bash
git clone https://github.com/foxn2000/sdg.git
cd sdg

# 'sdg'という名前でPython 3.11のconda環境を作成
conda create -n sdg python=3.11 -y
# 作成した環境を有効化
conda activate sdg
```

### 4. 依存ライブラリのインストール

`vllm` はPyTorchに依存しているため、まずはお使いのCUDAバージョンに合ったPyTorchをインストールします。

```bash
# CUDA 12.1 の場合 (公式: https://pytorch.org/get-started/locally/)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

次に、`requirements.txt` を使って残りのライブラリをインストールします。

```bash
pip install -r requirements.txt
```

### 5. モデルの準備

`settings.yaml` で指定されているモデルをダウンロードし、`data/model/` ディレクトリに配置します。Hugging Face Hubから直接ダウンロードするのが一般的です。

```bash
# 例: TinySwallow-1.5B-Instruct の場合
# git-lfsのインストールが必要な場合があります (sudo apt install git-lfs)
git lfs install
git clone https://huggingface.co/tokyotech-llm/TinySwallow-1.5B-Instruct ./data/model/TinySwallow-1.5B-Instruct
```
`settings.yaml` に記載されている他のモデル（`base_model_name`, `think_model_name`, `E5_path`）も同様に準備してください。

### 6. 実行

vLLM用の設定ファイル `settings.yaml` を使って実行します。

```bash
python main.py --config settings.yaml
```

---

## B. Ollama環境のセットアップ

### 1. OllamaとMinicondaのインストール

**Ollama:**
公式サイトの指示に従い、Ollamaをインストールします。
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Miniconda:**
Python環境の管理にはMinicondaを使用します。
```bash
# インストーラのダウンロード
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# インストールの実行
bash Miniconda3-latest-Linux-x86_64.sh
# シェルを再起動するか、source ~/.bashrc を実行してcondaコマンドを有効化
source ~/.bashrc
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

```bash
pip install -r requirements.txt
```
*(注意: `requirements.txt` には `vllm` も含まれていますが、Ollama利用時は実際には使用されません。気になる場合は `vllm` を除いたファイルを作成してインストールしても構いません。)*

### 4. Ollamaサーバーの起動とモデルの準備

まず、Ollamaサーバーを起動します。（通常はインストール後に自動でサービスが起動します）

次に、`settings_ollama.yaml` で指定されているモデルを `pull` します。

```bash
# 例: gemma:2b-instruct の場合
ollama pull gemma:2b-instruct
```
`settings_ollama.yaml` に記載されている他のモデル（`base_model_name`, `think_model_name`）も同様に `pull` してください。

### 5. 実行

Ollama用の設定ファイル `settings_ollama.yaml` を使って実行します。

```bash
python main.py --config settings_ollama.yaml
```
