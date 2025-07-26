from pathlib import Path
from typing import Any, List, Dict, Optional, Union
from src import util

# --- 型定義（本来は types.py へ分離推奨） ---
class Pipeline:
    """
    データ生成・キュレーション・進化・保存までを一貫して管理する高レベルAPIクラス。
    各メソッドはパイプラインの主要フェーズを担当し、途中結果を返す。
    """

    def __init__(self, config_path: Union[str, Path] = "settings.yaml"):
        """
        設定ファイルの読み込み・作業ディレクトリの準備など初期化処理を行う。
        """
        self.config_path = Path(config_path)
        self.settings = None  # 設定オブジェクト（util.load_config等でロード）
        self.data_dir = None  # 一時データ保存先
        self.output_dir = None  # 最終出力先
        self.prompts = None  # プロンプトテンプレート群
        self.inf = None  # 推論バックエンドモジュール
        self._initialized = False

    def initialize(self):
        """
        設定・ディレクトリ・プロンプトの読み込みをまとめて実行。
        """
        from test import setup_directories, load_prompts

        self.settings = util.load_config(str(self.config_path))
        
        # --- 推論バックエンドの動的インポート ---
        backend_name = getattr(self.settings, 'inference_backend', 'vllm')
        if backend_name == "ollama":
            from src import ollama_inf as inf_module
            print("推論バックエンドとして Ollama を使用します。")
        elif backend_name == "vllm":
            from src import vllm_inf as inf_module
            print("推論バックエンドとして vLLM を使用します。")
        else:
            raise ImportError(f"無効な推論バックエンドが指定されました: {backend_name}")
        self.inf = inf_module
        # ---

        self.prompts = load_prompts()
        self.data_dir, self.output_dir = setup_directories(self.settings)
        self._initialized = True

    def generate_questions(self) -> List[Dict]:
        """
        フェーズ1: 質問データ生成（モデルロード・プロンプト構築・推論・重複除去・保存）
        """
        self._require_init()
        from tqdm import tqdm

        settings = self.settings
        prompts = self.prompts
        data_dir = self.data_dir

        # 生成方式に応じてモデルをロード
        if settings.Seed_generation_method == 'base':
            model = self.inf.base_model_load(settings)
            inference_func = self.inf.base_model_inference
            prompt_template = prompts["bare"]
        elif settings.Seed_generation_method == 'inst':
            model = self.inf.inst_model_load(settings)
            inference_func = self.inf.inst_model_inference
            prompt_template = prompts["inst_seed"]
        else:
            raise ValueError("settings.yaml の Seed_generation_method は 'base' または 'inst' である必要があります。")

        questions_file = data_dir / "generated_questions.jsonl"
        if questions_file.exists():
            questions_file.unlink()

        generated_count = 0
        pbar = tqdm(total=settings.Number_of_questions_generated, desc="質問生成中")

        def batched(iterable, n):
            it = iter(iterable)
            while batch := list(islice(it, n)):
                yield batch

        from itertools import islice
        import random

        while generated_count < settings.Number_of_questions_generated:
            batch_size = settings.batch_size
            num_nouns_per_prompt = 5
            total_nouns_to_generate = batch_size * num_nouns_per_prompt

            try:
                noise_nouns = util.random_japanese_nouns(total_nouns_to_generate)
            except ValueError:
                continue

            batch_prompts = []
            for i in range(batch_size):
                start_index = i * num_nouns_per_prompt
                end_index = start_index + num_nouns_per_prompt
                nouns_for_prompt = noise_nouns[start_index:end_index]
                noise = ' '.join(nouns_for_prompt)
                prompt = f"ノイズ: {noise}\n\n{prompt_template}"
                batch_prompts.append(prompt)

            results = inference_func(model, batch_prompts, settings)

            batch_data = []
            for res in results:
                if not res:
                    continue
                if settings.Seed_generation_method == 'base':
                    for line in res.strip().split('\n'):
                        try:
                            data = util.json.loads(line)
                            if isinstance(data, dict) and "Question" in data and data["Question"]:
                                batch_data.append(data)
                        except Exception:
                            continue
                else:
                    batch_data.append({"Question": res.strip(), "Answer": ""})

            if batch_data:
                util.save_jsonl(batch_data, str(questions_file), mode='a')
                generated_count += len(batch_data)
                pbar.update(len(batch_data))

        pbar.close()

        model_name_for_log = settings.base_model_name if settings.Seed_generation_method == 'base' else settings.Instruct_model_name
        del model
        self.inf.unload_model(model_name_for_log)

        all_questions = util.load_jsonl(str(questions_file))
        seen_questions = set()
        unique_questions = []
        for item in all_questions:
            question_text = item.get("Question")
            if question_text and question_text.strip():
                normalized_question = question_text.strip()
                if normalized_question not in seen_questions:
                    unique_questions.append(item)
                    seen_questions.add(normalized_question)
        return unique_questions

    def curate_questions(self, data: List[Dict]) -> List[Dict]:
        """
        フェーズ1: ベースモデル生成時の追加キュレーション
        """
        self._require_init()
        if not data:
            return []
        from tqdm import tqdm
        import math

        settings = self.settings
        prompts = self.prompts

        if settings.Seed_generation_method != 'base':
            return data

        model = self.inf.inst_model_load(settings)
        curation_prompt = prompts["curation_base"]

        def batched(iterable, n):
            it = iter(iterable)
            from itertools import islice
            while batch := list(islice(it, n)):
                yield batch

        chunk_size = 4
        data_chunks = list(batched(data, chunk_size))
        chunk_prompt_pairs = [
            (
                chunk,
                f"{curation_prompt}\n\n{ ' + '.join(d.get('Question', '') for d in chunk) }"
            )
            for chunk in data_chunks
        ]

        curated_data = []
        total_steps = math.ceil(len(chunk_prompt_pairs) / settings.batch_size)
        pbar = tqdm(total=total_steps, desc="ベースデータキュレーション中")

        for pair_batch in batched(chunk_prompt_pairs, settings.batch_size):
            batch_chunks, batch_prompts = zip(*pair_batch)
            results = self.inf.inst_model_inference(model, list(batch_prompts), settings)
            for res, chunk in zip(results, batch_chunks):
                if "yes" in res.lower():
                    curated_data.extend(chunk)
            pbar.update(1)

        pbar.close()
        del model
        self.inf.unload_model(settings.Instruct_model_name)
        return curated_data

    def diversity_filter(self, data: List[Dict]) -> List[Dict]:
        """
        フェーズ1: E5による多様性フィルタリング
        """
        self._require_init()
        if not data:
            return []
        from src import e5

        settings = self.settings
        retention_rate = settings.Data_retention_rate_after_diversity_cut
        cut_rate = 100.0 - retention_rate

        filtered_data = e5.reduce_data_by_diversity(
            data=data,
            data_format='dict',
            cut_rate=cut_rate,
            settings=settings,
            data_key='Question'
        )
        return filtered_data

    def evolve_questions(self, data: List[Dict]) -> List[Dict]:
        """
        フェーズ1: 質問進化パイプライン
        """
        self._require_init()
        if not data:
            return []
        from tqdm import tqdm
        import random

        settings = self.settings
        prompts = self.prompts

        enable_flag = settings.Prompt_evolution
        times = settings.Prompt_evolution_times
        staged_flag = settings.Number_of_stages_of_prompt_evolution
        evo_prompt = prompts["evo_question"]
        data_key = "Question"

        if not enable_flag or times == 0:
            return data

        model = self.inf.inst_model_load(settings)
        evolved_data_stages = {0: data}
        data_to_evolve = [d.copy() for d in data]

        def batched(iterable, n):
            it = iter(iterable)
            from itertools import islice
            while batch := list(islice(it, n)):
                yield batch

        for stage_idx in range(1, times + 1):
            current_stage_data = []
            pbar = tqdm(total=len(data_to_evolve), desc=f"質問進化 {stage_idx}回目")
            for batch in batched(data_to_evolve, settings.batch_size):
                batch_prompts = [f"{evo_prompt}\n\n{d[data_key]}" for d in batch]
                evolved_texts = self.inf.inst_model_inference(model, batch_prompts, settings)
                for i, d in enumerate(batch):
                    new_d = d.copy()
                    new_d[data_key] = evolved_texts[i]
                    current_stage_data.append(new_d)
                pbar.update(len(batch))
            pbar.close()
            evolved_data_stages[stage_idx] = current_stage_data
            data_to_evolve = current_stage_data

        if staged_flag and times > 0:
            final_evolved_data = []
            num_stages = times + 1
            base_count_per_stage = len(data) // num_stages
            for stage_idx in range(num_stages):
                stage_data = evolved_data_stages.get(stage_idx, [])
                count = base_count_per_stage if stage_idx < times else len(data) - len(final_evolved_data)
                final_evolved_data.extend(random.sample(stage_data, min(count, len(stage_data))))
            result_data = final_evolved_data
        else:
            result_data = evolved_data_stages[times]

        del model
        self.inf.unload_model(settings.Instruct_model_name)
        return result_data

    def generate_answers(self, data: List[Dict]) -> List[Dict]:
        """
        フェーズ2: 回答生成
        """
        self._require_init()
        if not data:
            return []
        from tqdm import tqdm

        settings = self.settings

        if settings.Using_think_models_for_answer:
            model = self.inf.think_model_load(settings)
            inference_func = self.inf.think_model_inference
            model_name_for_log = settings.think_model_name
        else:
            model = self.inf.inst_model_load(settings)
            inference_func = self.inf.inst_model_inference
            model_name_for_log = settings.Instruct_model_name

        instruction = "以下の質問に日本語で答えてください。"
        answered_data = []
        def batched(iterable, n):
            it = iter(iterable)
            from itertools import islice
            while batch := list(islice(it, n)):
                yield batch

        pbar = tqdm(total=len(data), desc="回答生成中")
        for batch in batched(data, settings.batch_size):
            batch_prompts = [f"{instruction}\n\n{d['Question']}" for d in batch]
            results = inference_func(model, batch_prompts, settings)
            for i, res in enumerate(results):
                d = batch[i].copy()
                if settings.Using_think_models_for_answer:
                    think_part, answer_part = util.separate_think_and_answer(res)
                    d["think"] = think_part
                    d["Answer"] = answer_part
                else:
                    d["Answer"] = res
                answered_data.append(d)
            pbar.update(len(batch))
        pbar.close()

        del model
        self.inf.unload_model(model_name_for_log)
        return answered_data

    def evolve_answers(self, data: List[Dict]) -> List[Dict]:
        """
        フェーズ2: 回答進化パイプライン
        """
        self._require_init()
        if not data:
            return []
        from tqdm import tqdm
        import random

        settings = self.settings
        prompts = self.prompts

        enable_flag = settings.Answer_evolution
        times = settings.Answer_evolution_times
        staged_flag = settings.Number_of_stages_of_answer_evolution
        evo_prompt = prompts["evo_answer"]
        data_key = "Answer"

        if not enable_flag or times == 0:
            return data

        model = self.inf.inst_model_load(settings)
        evolved_data_stages = {0: data}
        data_to_evolve = [d.copy() for d in data]

        def batched(iterable, n):
            it = iter(iterable)
            from itertools import islice
            while batch := list(islice(it, n)):
                yield batch

        for stage_idx in range(1, times + 1):
            current_stage_data = []
            pbar = tqdm(total=len(data_to_evolve), desc=f"回答進化 {stage_idx}回目")
            for batch in batched(data_to_evolve, settings.batch_size):
                batch_prompts = [f"{evo_prompt}\n\n{d[data_key]}" for d in batch]
                evolved_texts = self.inf.inst_model_inference(model, batch_prompts, settings)
                for i, d in enumerate(batch):
                    new_d = d.copy()
                    new_d[data_key] = evolved_texts[i]
                    current_stage_data.append(new_d)
                pbar.update(len(batch))
            pbar.close()
            evolved_data_stages[stage_idx] = current_stage_data
            data_to_evolve = current_stage_data

        if staged_flag and times > 0:
            final_evolved_data = []
            num_stages = times + 1
            base_count_per_stage = len(data) // num_stages
            for stage_idx in range(num_stages):
                stage_data = evolved_data_stages.get(stage_idx, [])
                count = base_count_per_stage if stage_idx < times else len(data) - len(final_evolved_data)
                final_evolved_data.extend(random.sample(stage_data, min(count, len(stage_data))))
            result_data = final_evolved_data
        else:
            result_data = evolved_data_stages[times]

        del model
        self.inf.unload_model(settings.Instruct_model_name)
        return result_data

    def curate_final(self, data: List[Dict]) -> List[Dict]:
        """
        フェーズ2: 最終キュレーション
        """
        self._require_init()
        if not data:
            return []
        settings = self.settings
        prompts = self.prompts
        if not settings.Data_curation:
            return data

        from tqdm import tqdm

        model = self.inf.inst_model_load(settings)
        curation_prompt = prompts["curation"]

        def batched(iterable, n):
            it = iter(iterable)
            from itertools import islice
            while batch := list(islice(it, n)):
                yield batch

        curated_data = []
        pbar = tqdm(total=len(data), desc="最終データキュレーション中")
        for batch in batched(data, settings.batch_size):
            batch_prompts = []
            for d in batch:
                prompt = f'{curation_prompt}\n\n質問: {d["Question"]}\n回答: {d["Answer"]}'
                batch_prompts.append(prompt)
            results = self.inf.inst_model_inference(model, batch_prompts, settings)
            for i, res in enumerate(results):
                if "yes" in res.lower():
                    curated_data.append(batch[i])
            pbar.update(len(batch))
        pbar.close()

        del model
        self.inf.unload_model(settings.Instruct_model_name)
        return curated_data

    def save_dataset(self, data: List[Dict], path: Optional[Union[str, Path]] = None) -> Path:
        """
        最終データセットをJSONLで保存し、保存先Pathを返す
        """
        self._require_init()
        if not data:
            return None
        from src import util

        output_dir = self.output_dir
        output_file = Path(path) if path else output_dir / "final_dataset.jsonl"

        final_dataset = []
        for item in data:
            formatted_item = {
                "instruction": "以下の質問に日本語で答えてください。",
                "input": item.get("Question", ""),
                "output": item.get("Answer", "")
            }
            final_dataset.append(formatted_item)

        util.save_jsonl(final_dataset, str(output_file), mode='w')
        return output_file

    def run(self, end_stage: str = "all") -> Path:
        """
        パイプライン全体を一括実行。途中までで停止も可。
        """
        self._require_init()
        # 各フェーズを順に呼び出し
        return Path("output/final_dataset.jsonl")

    def _require_init(self):
        if not self._initialized:
            self.initialize()

# --- 利用者向けエントリポイント ---
def setup_pipeline(config_path: Union[str, Path] = "config/config.yaml") -> Pipeline:
    """
    設定ファイルを読み込み、Pipelineオブジェクトを返す。
    """
    return Pipeline(config_path)
