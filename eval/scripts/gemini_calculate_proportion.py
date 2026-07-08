import json
import time
from pathlib import Path
from tqdm import tqdm
import os
from openai import OpenAI

BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.gptsapi.net/v1")
MODEL_NAME = os.environ.get("OPENAI_MODEL", "gpt-4o")
THRESHOLDS = os.environ.get("THRESHOLDS", "0.4").split(",")

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", str(WORKSPACE_ROOT / "exp" / "AMBER" / "data" / "query")))

API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY environment variable.")

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

# 定义 System Prompt，强制要求严格的 JSON 结构返回
SYSTEM_PROMPT = """You are an expert evaluator for Multimodal Large Language Models. 
Compare the 'Initial Output' with the 'Final Output' and identify hallucinations. 
Extract specific terms (objects, attributes, relations) for each category.

You MUST respond in valid JSON format matching exactly this structure:
{
  "original_hallucinations": ["string"],
  "corrected_hallucinations": ["string"],
  "over_corrected_terms": ["string"],
  "new_hallucinations": ["string"],
  "rationale": "string"
}"""


def evaluate_single_pair(initial_text, final_text):
    prompt = f"""
    Initial Output: "{initial_text}"
    Final Output: "{final_text}"

    Analyze the differences and output the JSON.
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},  # 强制 JSON 模式
            temperature=0.1,  # 低温度保证输出稳定性
            timeout=30
        )

        # 将返回的 JSON 字符串解析为字典
        result_text = response.choices[0].message.content
        return json.loads(result_text)

    except Exception as e:
        print(f"\nAPI Error: {e}")
        return None


def main():
    for thresh in THRESHOLDS:
        input_file = Path(OUTPUT_DIR) / f"query_generative_rebuttal-thresh_{thresh}.json"
        output_eval_file = Path(OUTPUT_DIR) / f"eval_results_thresh_{thresh}.json"

        if not input_file.exists():
            print(f"Skipping {thresh}, file not found.")
            continue

        print(f"\n========== Evaluating Threshold {thresh} ==========")
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {input_file}: {e}")
            continue

        # 如果已经存在评估结果，可以加载并继续（断点续传）
        eval_results = []
        if output_eval_file.exists():
            try:
                with open(output_eval_file, 'r', encoding='utf-8') as f:
                    eval_results = json.load(f)
            except Exception:
                pass  # 如果文件损坏，则从头开始

        processed_ids = {item['id'] for item in eval_results if 'id' in item}

        for entry in tqdm(data, desc=f"Evaluating Thresh {thresh}"):
            # 防御性编程：跳过空条目
            if not entry or not isinstance(entry, dict):
                continue

            entry_id = entry.get('id')
            if entry_id in processed_ids:
                continue

            init_resp = entry.get('initial_response')
            final_resp = entry.get('response')

            # 将 None 转为空字符串，方便比对
            init_resp_str = str(init_resp).strip() if init_resp else ""
            final_resp_str = str(final_resp).strip() if final_resp else ""

            # 如果两者完全一致，或者有空值，直接跳过 API 调用以节省额度和时间
            if not init_resp_str or not final_resp_str or init_resp_str == final_resp_str:
                result_dict = {
                    "original_hallucinations": [],
                    "corrected_hallucinations": [],
                    "over_corrected_terms": [],
                    "new_hallucinations": [],
                    "rationale": "Identical or empty outputs."
                }
            else:
                result_dict = evaluate_single_pair(init_resp_str, final_resp_str)
                time.sleep(0.5)  # 简单的流控，防止触发 API 限流

            if result_dict:
                eval_entry = {
                    "id": entry_id,
                    "image": entry.get("image", ""),
                    "evaluation": result_dict
                }
                eval_results.append(eval_entry)

                # 实时保存，防止中断
                with open(output_eval_file, 'w', encoding='utf-8') as f:
                    json.dump(eval_results, f, indent=2, ensure_ascii=False)

        # ====== 统计并打印该阈值的最终比例 ======
        total_orig_hal = 0
        total_corrected_hal = 0
        total_over_corrected = 0
        total_new_hal = 0

        for res in eval_results:
            ev = res.get('evaluation', {})
            total_orig_hal += len(ev.get('original_hallucinations', []))
            total_corrected_hal += len(ev.get('corrected_hallucinations', []))
            total_over_corrected += len(ev.get('over_corrected_terms', []))
            total_new_hal += len(ev.get('new_hallucinations', []))

        correction_rate = (total_corrected_hal / total_orig_hal * 100) if total_orig_hal > 0 else 0

        print(f"\n--- Threshold {thresh} Summary ---")
        print(f"Total Original Hallucination Terms: {total_orig_hal}")
        print(f"Successfully Corrected Terms: {total_corrected_hal}")
        print(f"Correction Rate (Recall): {correction_rate:.2f}%")
        print(f"Over-corrected Terms: {total_over_corrected}")
        print(f"New Hallucinations (DINO induced): {total_new_hal}")


if __name__ == "__main__":
    main()