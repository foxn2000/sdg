# sdg(Scalable Data Generator)

## 概要

本リポジトリは、vLLM ベースの日本語 LLM 推論環境と、プロンプト進化／データ生成パイプライン（質問生成・回答生成・キュレーション）を手軽に実行できる CLI スクリプト群を提供します。  
`settings.yaml` でモデル名・量子化・TP サイズなどを一元管理し、`test_run.py` で簡易動作確認が可能です。

---

## 主な構成ファイル

- `settings.yaml` … すべての推論・生成パラメータを集中管理
- `requirements.txt` … vllm 0.9.1 ほか必要ライブラリ
- `test_run.py` … src/ 以下の vllm_inf.py, util.py を用いた最小動作例
- `prompts/` … 質問生成／キュレーション用プロンプトテンプレート
- `data/` … 生成された Q&A や中間ファイルの出力先
- `memo.md` … Hugging Face からモデルをダウンロードする例
- `INSPIRATIONS.md` … インスピレーション(現状空)
- `LICENSE` … MIT License

---

## インストール

```bash
git clone https://github.com/foxn2000/sdg.git
cd sdg
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## モデル取得

`settings.yaml` で指定しているパスにモデルを配置します。例:

```bash
huggingface-cli download SakanaAI/TinySwallow-1.5B-Instruct --local-dir ./data/model/TinySwallow-1.5B-Instruct
huggingface-cli download sbintuitions/sarashina2.2-3b --local-dir ./data/model/sarashina2.2-3b
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --local-dir ./data/model/DeepSeek-R1-Distill-Qwen-1.5B
```

各モデルの GPU メモリ要件は `*_gpu_memory_utilization` を参照してください。

---

## 使い方

### 動作確認

```bash
python test_run.py
```

2 つのサンプルプロンプトに対する推論結果が標準出力に表示されます。

```bash
python main.py
```

### データ生成パイプライン

(※ src パッケージのスクリプトがこの機能を提供。README には
`python -m src.pipeline.generate_questions` 等の想定コマンド例を記載予定)

---

## 設定

`settings.yaml` 内の主なキー

| 区分             | キー例                       | 説明                                 |
|------------------|-----------------------------|--------------------------------------|
| Instruct モデル  | Instruct_model_name          | 推論時に使用する指示追従モデル       |
| Base モデル      | base_model_name              | 質問生成で使用する基盤モデル         |
| Think モデル     | think_model_name             | 長考回答用モデル                     |
| バッチ設定       | batch_size, max_tokens       | vLLM 推論バッチ・トークン長          |
| 生成フロー       | Seed_generation_method など  | プロンプト進化・回答進化の ON/OFF    |

---

## 参考

- [vLLM](https://github.com/vllm-project/vllm)
- SentenceTransformers multilingual-e5
- Cogito / ELYZA Llama-3-JP 8B など

---

## ライセンス

MIT

---

## 貢献

Issues/Pull Requests は歓迎です。モデルパスやパイプライン改善提案もお待ちしております。
