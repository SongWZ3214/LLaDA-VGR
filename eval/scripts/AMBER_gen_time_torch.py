import os
import time
import json
import copy
import torch
import warnings
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ==================== 1. 公平性对齐：强制全局 bfloat16 ====================
torch.set_default_dtype(torch.bfloat16)
# =====================================================================

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

warnings.filterwarnings("ignore")

# 路径配置
PRETRAINED = "GSAI-ML/LLaDA-V"
IMAGE_DIR = "/data0/swz/exp/AMBER/data/image"
BASE_QUERY_JSON = "/data0/swz/exp/AMBER/data/query/query_generative.json"
# 自动在原文件名基础上加上 _0330_base
OUTPUT_JSON = BASE_QUERY_JSON.replace(".json", "_0330_base.json")

DEVICE = "cuda:0"
START_INDEX = 0
END_INDEX = 50

# 默认Prompt (对齐你之前脚本中的 DEFAULT_PROMPT_TEXT)
DEFAULT_PROMPT_TEXT = "Please describe the image in detail. Use less absolute directional descriptions. Do not repeat information."


def main():
    print(f"Loading model {PRETRAINED} on {DEVICE}...")

    # ==================== 2. 公平性对齐：使用 SDPA 加速计算 ====================
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        model_path=PRETRAINED,
        model_base=None,
        model_name="llava_llada",
        attn_implementation="sdpa",  # 强制使用 SDPA 节省显存
        device_map=DEVICE
    )

    # 👇【加上这一行！】强制把刚加载出来的 float16 权重全部洗成 bfloat16
    model.to(torch.bfloat16)

    model.eval()

    # 确保完全不使用 Fast-dLLM 或 dLLM cache
    print("Testing strictly WITHOUT Fast-dLLM and dLLM Cache for fair baseline comparison.")

    # 读取数据
    print(f"Reading queries from: {BASE_QUERY_JSON}")
    with open(BASE_QUERY_JSON, 'r', encoding='utf-8') as f:
        query_data = json.load(f)

    tasks_to_process = query_data[START_INDEX:END_INDEX]
    print(f"Processing {len(tasks_to_process)} samples (Index {START_INDEX} to {END_INDEX})...")

    results = []
    torch_device = torch.device(DEVICE)

    for item in tqdm(tasks_to_process, desc="Benchmarking Base Model"):
        image_filename = item.get('image')
        query_text = item.get('query', '')

        # 处理图片路径
        img_path = Path(IMAGE_DIR) / image_filename
        if not img_path.exists():
            print(f"Warning: Image not found {img_path}")
            continue

        # 加载和处理图像
        image = Image.open(img_path).convert("RGB")
        image_tensor = process_images([image], image_processor, model.config)
        # 对齐 bfloat16
        image_tensor = [_image.to(dtype=torch.bfloat16, device=torch_device) for _image in image_tensor]

        # 准备 Prompt
        prompt_text = query_text if query_text and query_text.strip() else DEFAULT_PROMPT_TEXT
        question = DEFAULT_IMAGE_TOKEN + "\n" + prompt_text
        conv = copy.deepcopy(conv_templates["llava_llada"])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(
            0).to(torch_device)
        image_sizes = [image.size]

        # ==================== 3. 公平性对齐：严格的显存与时间监控 ====================
        torch.cuda.synchronize(torch_device)
        torch.cuda.empty_cache()  # 清理历史缓存
        torch.cuda.reset_peak_memory_stats(torch_device)

        start_time = time.time()

        # ==================== 4. 公平性对齐：关闭梯度计算 (极度重要) =================
        with torch.inference_mode():
            # ==================== 5. 公平性对齐：统一底层生成参数 ====================
            cont = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                steps=128,  # 强制 128 步
                gen_length=128,  # 强制 128 长度
                block_length=128,
                tokenizer=tokenizer,
                stopping_criteria=['<|eot_id|>'],
                prefix_refresh_interval=32,
                threshold=1,
            )

        torch.cuda.synchronize(torch_device)
        end_time = time.time()

        # 记录监控数据
        latency = end_time - start_time
        peak_allocated_mb = torch.cuda.max_memory_allocated(torch_device) / (1024 * 1024)
        peak_reserved_mb = torch.cuda.max_memory_reserved(torch_device) / (1024 * 1024)
        # =========================================================================

        # 解码输出 (忽略特殊 token 以保持文本干净)
        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()

        # 保存当前条目结果
        item['response'] = text_outputs
        item['latency_seconds'] = latency
        item['peak_allocated_mb'] = peak_allocated_mb
        item['peak_reserved_mb'] = peak_reserved_mb
        results.append(item)

    # 保存结果到新文件
    os.makedirs(Path(OUTPUT_JSON).parent, exist_ok=True)
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Benchmark finished! Results saved to: {OUTPUT_JSON}")

    # 打印前几条的平均情况供你快速参考
    avg_latency = sum(r['latency_seconds'] for r in results) / len(results)
    avg_memory = sum(r['peak_allocated_mb'] for r in results) / len(results)
    print(f"📊 Average Latency: {avg_latency:.2f} s/image")
    print(f"📊 Average Peak Allocated Memory: {avg_memory:.2f} MB")


if __name__ == "__main__":
    main()