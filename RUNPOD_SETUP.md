# RunPodでのSDGプロジェクト実行手順

## 1. RunPodインスタンスの起動

1. [RunPod Console](https://console.runpod.io/deploy?gpu=RTX%20A5000&count=1&template=runpod-torch-v280)にアクセス
2. 以下の設定でインスタンスを作成：
   - **GPU**: RTX A5000 (1台)
   - **テンプレート**: vLLM公式テンプレート
   - **ストレージ**: 最低50GB（モデルサイズを考慮）

## 2. SSH接続とプロジェクトセットアップ

```bash
# SSH接続（RunPodのコンソールからSSH情報を取得）
ssh root@<runpod-ip> -p <port>

# プロジェクトのクローン
git clone <your-repository-url>
cd sdg

# 依存関係のインストール
pip install -r requirements.txt
```

## 3. 必要なモデルのダウンロード

```bash
# モデル保存ディレクトリの作成
mkdir -p data/model

# 各モデルのダウンロード
cd data/model

# 1. TinySwallow-1.5B-Instruct
git clone https://huggingface.co/tokyotech-llm/Swallow-1.5b-instruct-hf TinySwallow-1.5B-Instruct

# 2. sarashina2.2-3b  
git clone https://huggingface.co/sbintuitions/sarashina2-3b sarashina2.2-3b

# 3. DeepSeek-R1-Distill-Qwen-1.5B
git clone https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B DeepSeek-R1-Distill-Qwen-1.5B

# 4. multilingual-e5-large
git clone https://huggingface.co/intfloat/multilingual-e5-large multilingual-e5-large

cd ../..
```

## 4. 設定ファイルの調整（必要に応じて）

`settings.yaml`でGPUメモリ使用率を調整：

```yaml
# RTX A5000 (24GB VRAM)の場合
Instruct_gpu_memory_utilization: 0.8
base_gpu_memory_utilization: 0.6
think_gpu_memory_utilization: 0.8
```

## 5. プロジェクトの実行

```bash
# メインスクリプトの実行
python main.py

# または特定の設定ファイルを使用
python main.py --config settings.yaml
```

## 6. 動作確認

```bash
# テストスクリプトの実行
python test_run.py
```

## トラブルシューティング

### モデルダウンロードでエラーが出る場合
```bash
# Git LFS が必要な場合
git lfs install
```

### GPU メモリ不足の場合
- `settings.yaml`の`gpu_memory_utilization`値を下げる（0.5-0.7程度）
- `tensor_parallel_size`を調整

### 依存関係エラーの場合
```bash
# 最新版の依存関係をインストール
pip install --upgrade vllm openai google-generativeai
```

## 注意事項

- RTX A5000は24GB VRAMなので、全モデルを同時実行する場合はメモリ使用量に注意
- モデルダウンロードには時間がかかる場合があります（数GB〜数十GB）
- SSH接続が切れないよう、`screen`や`tmux`の使用を推奨

## 参考

- [vLLM公式ドキュメント](https://docs.vllm.ai/)
- [RunPod公式ドキュメント](https://docs.runpod.io/)