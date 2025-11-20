import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Literal
from openai import OpenAI
import pandas as pd
from pydantic import BaseModel
import re
import codecs
import base64
from google import genai
import PIL.Image
import io
from google.genai import types
from typing_extensions import TypedDict
BASE64_RE = re.compile(r'^[A-Za-z0-9+/]+={0,2}$')

class Inconsistency(TypedDict):
    """不一致的短语"""
    original_text: str
    corrected_text: str

class FineGrainedPhrase(TypedDict):
    """细粒度短语（正确但描述细粒度视觉细节）"""
    original_text: str
    
class ResultList(TypedDict):
    inconsistencies: List[Inconsistency]
    fine_grained_phrases: List[FineGrainedPhrase]

# 初始化 OpenAI 客户端
# 请确保设置了 OPENAI_API_KEY 环境变量
OPENAI_API_KEY = "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"
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# 初始化 Gemini 客户端（如果可用）
GEMINI_API_KEY = "AIzaSyDhLOePL0qFXgR3eLU6rpQGN-_Dyei_0rY"  # 请在此处填入您的 Gemini API 密钥
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

def load_parquet_data(parquet_path: str) -> pd.DataFrame:
    """加载 parquet 文件数据"""
    df = pd.read_parquet(parquet_path)
    return df

def evaluate_with_openai(image_base64: str, generated_text: str) -> Dict[str, Any]:
    """
    使用 OpenAI API 评估生成文本与图像的一致性
    
    返回包含不一致短语和细粒度短语的字典
    """
    
    prompt = f"""
You are an expert editor with an extremely high attention to detail, specializing in image-text consistency.

## Task Overview:
You will perform TWO tasks simultaneously. Given an image and its caption:
1.  **Task A:** Find all factually *inconsistent* phrases.
2.  **Task B:** Find all factually *correct* phrases that describe *fine-grained visual details*.

You will return a single JSON object containing two separate lists for these findings.

---

## Part 1: Task A - Identifying Inconsistencies

Find every phrase in the caption that is factually *inconsistent* with the image.

**Principle: Semantic Phrase Correction**
- Your goal is to find the *smallest semantically complete phrase* that contains the error. This is often a noun phrase, prepositional phrase, or verb phrase.
- This is *not* limited to a single word, but should *not* be an unnecessarily long sentence.
- **Example 1:** If caption says "a tall red dog" (image: "short red dog"), return:
  {{"original_text": "a tall red dog", "corrected_text": "a short red dog"}}
- **Example 2:** If caption says "on the table" (image: "under the table"), return:
  {{"original_text": "on the table", "corrected_text": "under the table"}}
- **Example 3:** If caption mentions "a blue car" (image: "empty street"), return:
  {{"original_text": "a blue car", "corrected_text": "an empty street"}}

**Correction Rules:**
- For each inconsistency, provide the `original_text` and a `corrected_text`.
- The `corrected_text` MUST be a positive replacement (e.g., "an empty shelf").
- Do NOT use negations or corrective markers (e.g., "no", "not", "instead").
- Maintain the same grammatical role.

---

## Part 2: Task B - Identifying Fine-Grained Phrases

Find all phrases in the caption that are factually *correct* and describe *fine-grained visual details*.

**Definition of "Fine-Grained":**
- These are phrases that go beyond simple object identification (e.g., not just "a cat" or "a toilet").
- Look for descriptions of:
    - **Attributes:** Specific colors, patterns, textures (e.g., "a black and white checkered floor", "tabby fur", "a bright pink flamingo")
    - **Complex Relations:** Non-obvious spatial positions (e.g., "peering over the lid")
    - **Specific States:** (e.g., "lid is open", "stretching its front paw")

**Correction Rules:**
- These phrases MUST be factually correct.
- For these, you ONLY need to provide the `original_text`. No `corrected_text` is needed.

---

## Formatting Rules:
- Return ONLY valid JSON in the following schema (no extra text).
- **If the caption is fully consistent and contains no fine-grained details (rare), return the empty schema.**

```json
{{
  "inconsistencies": [
    {{
      "original_text": "minimal inconsistent semantic phrase",
      "corrected_text": "drop-in positive replacement phrase"
    }}
  ],
  "fine_grained_phrases": [
    {{
      "original_text": "correct and detailed phrase"
    }}
  ]
}}
```
"""
    
    try:
        response = openai_client.responses.parse(
            model="gpt-4o",  # 或使用 "gpt-4-vision-preview"
            input=[
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": generated_text
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    ]
                }
            ],
            text_format=ResultList
        )
        
        result = response.output_parsed
        return result
        
    except Exception as e:
        print(f"调用 OpenAI API 时出错: {e}")
        return None

def evaluate_with_gemini(image_base64: str, generated_text: str) -> Dict[str, Any]:
    """
    使用 Google Gemini API 评估生成文本与图像的一致性
    
    返回包含不一致短语和细粒度短语的字典
    """    
    if not GEMINI_API_KEY:
        print("错误: 未设置 GEMINI_API_KEY")
        return None
    
    prompt = f"""
You are an expert editor with an extremely high attention to detail, specializing in image-text consistency.

## Task Overview:
You will perform TWO tasks simultaneously. Given an image and its caption:
1.  **Task A:** Find all factually *inconsistent* phrases.
2.  **Task B:** Find all factually *correct* phrases that describe *fine-grained visual details*.

You will return a single JSON object containing two separate lists for these findings.

---

## Part 1: Task A - Identifying Inconsistencies

Find every phrase in the caption that is factually *inconsistent* with the image.

**Principle: Semantic Phrase Correction**
- Your goal is to find the *smallest semantically complete phrase* that contains the error. This is often a noun phrase, prepositional phrase, or verb phrase.
- This is *not* limited to a single word, but should *not* be an unnecessarily long sentence.
- **Example 1:** If caption says "a tall red dog" (image: "short red dog"), return:
  {{"original_text": "a tall red dog", "corrected_text": "a short red dog"}}
- **Example 2:** If caption says "on the table" (image: "under the table"), return:
  {{"original_text": "on the table", "corrected_text": "under the table"}}
- **Example 3:** If caption mentions "a blue car" (image: "empty street"), return:
  {{"original_text": "a blue car", "corrected_text": "an empty street"}}

**Correction Rules:**
- For each inconsistency, provide the `original_text` and a `corrected_text`.
- The `corrected_text` MUST be a positive replacement (e.g., "an empty shelf").
- Do NOT use negations or corrective markers (e.g., "no", "not", "instead").
- Maintain the same grammatical role.

---

## Part 2: Task B - Identifying Fine-Grained Phrases

Find all phrases in the caption that are factually *correct* and describe *fine-grained visual details*.

**Definition of "Fine-Grained":**
- These are phrases that go beyond simple object identification (e.g., not just "a cat" or "a toilet").
- Look for descriptions of:
    - **Attributes:** Specific colors, patterns, textures (e.g., "a black and white checkered floor", "tabby fur", "a bright pink flamingo")
    - **Complex Relations:** Non-obvious spatial positions (e.g., "peering over the lid")
    - **Specific States:** (e.g., "lid is open", "stretching its front paw")

**Correction Rules:**
- These phrases MUST be factually correct.
- For these, you ONLY need to provide the `original_text`. No `corrected_text` is needed.

---

## Formatting Rules:
- Return ONLY valid JSON in the following schema (no extra text).
- **If the caption is fully consistent and contains no fine-grained details (rare), return the empty schema.**

```json
{{
  "inconsistencies": [
    {{
      "original_text": "minimal inconsistent semantic phrase",
      "corrected_text": "drop-in positive replacement phrase"
    }}
  ],
  "fine_grained_phrases": [
    {{
      "original_text": "correct and detailed phrase"
    }}
  ]
}}
```

Caption to evaluate:
{generated_text}
"""
    
    try:
        # 解码 base64 图像
        image_data = base64.b64decode(image_base64)
                
        # 准备图像和文本        
        image = PIL.Image.open(io.BytesIO(image_data))
        
        # 调用 Gemini API
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, image],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ResultList,
            )
        )
        # print(response.parsed)
        
        return response.parsed
        
    except Exception as e:
        print(f"调用 Gemini API 时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_all_json_files(directory: str) -> List[Tuple[int, dict, str]]:
    """
    加载目录下所有 JSON 文件。
    
    Returns:
        List of (index, data_dict, filepath) tuples, sorted by index
    """
    directory_path = Path(directory)
    if not directory_path.exists():
        raise FileNotFoundError(f"目录不存在: {directory}")
    
    files_data = []
    json_files = sorted(directory_path.glob("*.json"))
    
    print(f"找到 {len(json_files)} 个 JSON 文件")
    
    for filepath in json_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            index = data.get('index')
            if index is None:
                # 尝试从文件名提取索引
                try:
                    index = int(filepath.stem)
                except ValueError:
                    print(f"警告: 文件 {filepath} 没有有效的索引，跳过")
                    continue
            
            files_data.append((index, data, str(filepath)))
        except Exception as e:
            print(f"警告: 读取文件 {filepath} 时出错: {e}")
            continue
    
    # 按索引排序
    files_data.sort(key=lambda x: x[0])
    print(f"成功加载 {len(files_data)} 个文件")
    
    return files_data

def get_image_base64(df: pd.DataFrame, sample_index: int) -> str:
    """
    从 parquet 文件中获取 base64 编码的图像
    
    Args:
        df: parquet 数据框
        sample_index: 样本索引
    
    Returns:
        base64 编码的图像字符串，如果失败返回 None
    """
    try:
        s = df.iloc[sample_index]['binary']
        image_base64 = None
        looks_hex_escaped = bool(re.search(r'\\x[0-9A-Fa-f]{2}', str(s)))
        s_str = str(s)
        is_base64_like = (
            len(s_str) > 32 and
            len(s_str.replace("\n", "").replace("\r", "").replace(" ", "")) % 4 == 0 and
            bool(BASE64_RE.match(s_str.replace("\n", "").replace("\r", "").replace(" ", "")))
        )

        if looks_hex_escaped and not is_base64_like:
            try:
                # 去掉可能的开头 b'...' 外壳
                if s_str.startswith(("b'", 'b"')) and s_str.endswith(("'", '"')):
                    s_inner = s_str[2:-1]
                else:
                    s_inner = s_str
                # 把 \xHH 转回原字节
                b = codecs.decode(s_inner, 'unicode_escape').encode('latin-1')
                image_base64 = base64.b64encode(b).decode('utf-8')
            except Exception:
                pass
        else:
            # 如果已经是 base64 格式，直接使用
            if isinstance(s, bytes):
                image_base64 = base64.b64encode(s).decode('utf-8')
            else:
                image_base64 = s_str
        
        return image_base64
    except Exception as e:
        print(f"  获取图像数据时出错: {e}")
        return None

def process_evaluation(
    input_directory: str, 
    parquet_file: str, 
    max_samples: int = None, 
    overwrite_existing: bool = False,
    api_provider: Literal["gpt", "gemini"] = "gpt"
):
    """
    处理评估任务，读取目录下的各个文件，评估后将结果写回文件
    
    Args:
        input_directory: 输入目录路径（包含各个 JSON 文件）
        parquet_file: DetailCaps-4870_refined_EN.parquet 文件路径
        max_samples: 最大处理样本数量（用于测试，None 表示处理所有样本）
        overwrite_existing: 如果为True，覆盖已处理的文件；如果为False，跳过已处理的文件
        api_provider: API 提供商，可选 "gpt" 或 "gemini"
    """
    # 加载所有 JSON 文件
    print("正在加载所有 JSON 文件...")
    files_data = load_all_json_files(input_directory)
    
    if not files_data:
        print("错误: 没有找到任何有效的 JSON 文件")
        return
    
    # 限制处理数量（如果指定）
    if max_samples is not None:
        files_data = files_data[:max_samples]
        print(f"测试模式: 只处理前 {max_samples} 个样本")
    
    print("正在加载 parquet 文件...")
    df = load_parquet_data(parquet_file)
    print(f"Parquet 文件包含 {len(df)} 行数据")
    
    # 显示覆盖模式
    if overwrite_existing:
        print("模式: 覆盖已处理的文件")
    else:
        print("模式: 跳过已处理的文件")
    
    # 显示 API 提供商
    if api_provider == "gemini":
        if not GEMINI_API_KEY:
            print("错误: 未设置 GEMINI_API_KEY")
            return
        print(f"API 提供商: Gemini (gemini-1.5-pro)")
    else:
        print(f"API 提供商: OpenAI (gpt-4o)")
    
    # 统计信息
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    # 处理每个文件
    total = len(files_data)
    for idx, (sample_index, data, filepath) in enumerate(files_data):
        print(f"\n[{idx+1}/{total}] 正在处理文件 {Path(filepath).name} (index: {sample_index})...")
        
        # 检查是否已经有评估结果
        # 如果两个字段都存在且都是列表，至少有一个是非空列表，则跳过
        # 如果两个都是空列表，则需要重新处理
        inconsistencies = data.get('inconsistencies', None)
        fine_grained_phrases = data.get('fine_grained_phrases', None)
        
        has_inconsistencies_list = isinstance(inconsistencies, list)
        has_fine_grained_list = isinstance(fine_grained_phrases, list)
        
        # 检查是否至少有一个是非空列表
        has_non_empty = False
        both_empty = False
        if has_inconsistencies_list and has_fine_grained_list:
            inconsistencies_len = len(inconsistencies)
            fine_grained_len = len(fine_grained_phrases)
            # 至少有一个是非空列表
            has_non_empty = inconsistencies_len > 0 or fine_grained_len > 0
            # 如果两个都是空列表，需要重新处理
            both_empty = inconsistencies_len == 0 and fine_grained_len == 0
        
        # 如果两个字段都存在且都是列表，且至少有一个是非空列表，则跳过
        if not overwrite_existing and has_inconsistencies_list and has_fine_grained_list and has_non_empty:
            print(f"  跳过: 文件已包含评估结果（inconsistencies: {len(inconsistencies)} 个, fine_grained_phrases: {len(fine_grained_phrases)} 个）")
            skipped_count += 1
            continue
        elif overwrite_existing and has_inconsistencies_list and has_fine_grained_list and has_non_empty:
            print(f"  覆盖: 文件已包含评估结果，将重新处理")
        
        # 如果两个都是空列表，需要重新处理并移除 error 字段
        if has_inconsistencies_list and has_fine_grained_list and both_empty:
            print(f"  重新处理: 两个字段都是空列表，需要重新评估")
            # 移除 error 字段（如果存在）
            if 'error' in data:
                del data['error']
                # 先保存一次，移除 error 字段
                try:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"  警告: 无法移除 error 字段: {e}")
        
        # 获取生成文本
        generated_text = data.get('generated_text', '')
        if not generated_text:
            print(f"  警告: 文件没有 generated_text，跳过")
            skipped_count += 1
            continue
        
        # 获取 base64 图像
        image_base64 = get_image_base64(df, sample_index)
        if not image_base64:
            print(f"  警告: 无法获取图像数据")
            error_count += 1
            # 添加错误标记到文件
            try:
                data['inconsistencies'] = []
                data['fine_grained_phrases'] = []
                data['error'] = f"Cannot find base64 for image: {sample_index}"
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"  错误: 无法更新文件: {e}")
            continue
        
        # 调用 API 评估
        try:
            if api_provider == "gemini":
                evaluation_result = evaluate_with_gemini(image_base64, generated_text)
            else:
                evaluation_result = evaluate_with_openai(image_base64, generated_text)
            
            if not evaluation_result:
                print(f"  警告: 评估失败")
                error_count += 1
                # 添加错误标记到文件
                data['inconsistencies'] = []
                data['fine_grained_phrases'] = []
                data['error'] = f"Evaluation failed ({api_provider})"
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                continue
            
            # 构建不一致列表
            # inconsistencies = evaluation_result.inconsistencies
            inconsistencies = evaluation_result['inconsistencies']
            inconsistencies_list = []
            
            for inconsistency in inconsistencies:
                inconsistency_entry = {
                    "original_text": inconsistency['original_text'],
                    "corrected_text": inconsistency['corrected_text']
                }
                inconsistencies_list.append(inconsistency_entry)
            
            # 构建细粒度短语列表
            fine_grained_phrases = evaluation_result['fine_grained_phrases']
            fine_grained_list = []
            
            for phrase in fine_grained_phrases:
                phrase_entry = {
                    "original_text": phrase['original_text']
                }
                fine_grained_list.append(phrase_entry)
            
            # 将结果添加到文件数据中
            data['inconsistencies'] = inconsistencies_list
            data['fine_grained_phrases'] = fine_grained_list
            
            # 移除 error 字段（如果存在）
            if 'error' in data:
                del data['error']
            
            # 写回文件
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            processed_count += 1
            print(f"  ✓ 成功处理，发现 {len(inconsistencies_list)} 个不一致短语，{len(fine_grained_list)} 个细粒度短语")
            
        except Exception as e:
            print(f"  ✗ 处理出错: {e}")
            error_count += 1
            # 添加错误标记到文件
            try:
                data['inconsistencies'] = []
                data['fine_grained_phrases'] = []
                data['error'] = f"Evaluation error: {str(e)}"
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except Exception as e2:
                print(f"  错误: 无法更新文件: {e2}")
        
        # 每处理 10 个文件打印一次进度
        if (idx + 1) % 10 == 0:
            print(f"\n进度: 已处理 {idx + 1}/{total} 个文件 [成功: {processed_count}, 跳过: {skipped_count}, 错误: {error_count}]")
    
    # 打印最终统计
    print("\n" + "="*60)
    print("评估完成")
    print("="*60)
    print(f"总文件数: {total}")
    print(f"成功处理: {processed_count}")
    print(f"跳过: {skipped_count}")
    print(f"错误: {error_count}")
    print(f"\n所有结果已写入各自的 JSON 文件")

def main():
    """主函数"""
    input_directory = '/data2/swz/LLaDA-VGR/test/result/detailcaps_outputs'
    parquet_file = '/data2/swz/LLaDA-VGR/test/data/DetailCaps-4870_refined_EN.parquet'
    
    # 控制处理数据的数量，用于测试
    # 设置为 None 表示处理所有数据，设置为数字表示只处理前 N 个样本
    MAX_SAMPLES = None  # 测试时设置为 5，正式运行时设置为 None
    
    # 控制是否覆盖已处理的文件
    # True: 覆盖已处理的文件（重新评估）
    # False: 跳过已处理的文件（默认）
    OVERWRITE_EXISTING = False
    
    # 选择 API 提供商
    # "gpt": 使用 OpenAI GPT-4o
    # "gemini": 使用 Google Gemini 1.5 Pro
    API_PROVIDER = "gemini"  # 或 "gemini"

    try:
        process_evaluation(
            input_directory, 
            parquet_file, 
            max_samples=MAX_SAMPLES,
            overwrite_existing=OVERWRITE_EXISTING,
            api_provider=API_PROVIDER
        )
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

