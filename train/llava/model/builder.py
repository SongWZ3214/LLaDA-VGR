#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings
import shutil
import time
import random

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from llava.model import *
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.utils import rank0_print


def _load_tokenizer_with_retry(model_path_or_name, use_fast=False, max_retries=3, base_delay=2):
    """
    加载 tokenizer，带重试机制和延迟，避免 HuggingFace API 速率限制
    
    Args:
        model_path_or_name: 模型路径或名称
        use_fast: 是否使用快速 tokenizer
        max_retries: 最大重试次数
        base_delay: 基础延迟（秒），每次重试会增加随机延迟
    """
    import os
    from huggingface_hub.errors import HfHubHTTPError
    
    # 添加随机延迟，避免多个进程同时访问
    delay = base_delay + random.uniform(0, 2)
    time.sleep(delay)
    
    for attempt in range(max_retries):
        try:
            # 首先尝试使用本地缓存
            if attempt == 0:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_path_or_name, 
                        use_fast=use_fast,
                        local_files_only=True
                    )
                    return tokenizer
                except (OSError, ValueError):
                    # 如果本地没有缓存，继续尝试在线下载
                    pass
            
            # 尝试在线下载
            tokenizer = AutoTokenizer.from_pretrained(
                model_path_or_name, 
                use_fast=use_fast,
                local_files_only=False
            )
            return tokenizer
            
        except HfHubHTTPError as e:
            if e.response.status_code == 429:  # Too Many Requests
                if attempt < max_retries - 1:
                    # 指数退避 + 随机抖动
                    wait_time = base_delay * (2 ** attempt) + random.uniform(0, 3)
                    rank0_print(f"Rate limited. Waiting {wait_time:.1f}s before retry {attempt + 1}/{max_retries}...")
                    time.sleep(wait_time)
                else:
                    rank0_print(f"Failed to load tokenizer after {max_retries} attempts due to rate limiting.")
                    raise
            else:
                # 其他错误直接抛出
                raise
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = base_delay * (2 ** attempt) + random.uniform(0, 2)
                rank0_print(f"Error loading tokenizer: {e}. Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                raise
    
    raise RuntimeError(f"Failed to load tokenizer after {max_retries} attempts")


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", torch_dtype="float16",attn_implementation="flash_attention_2", customized_config=None, overwrite_config=None, **kwargs):
    kwargs["device_map"] = device_map

    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    elif torch_dtype == "float16":
        kwargs["torch_dtype"] = torch.float16
    elif torch_dtype == "bfloat16":
        kwargs["torch_dtype"] = torch.bfloat16
    else:
        import pdb;pdb.set_trace()

    if customized_config is not None:
        kwargs["config"] = customized_config

    if "multimodal" in kwargs:
        if kwargs["multimodal"] is True:
            is_multimodal = True
            kwargs.pop("multimodal")
    else:
        is_multimodal = False

    if "llava" in model_name.lower() or is_multimodal:
        # Load LLaVA model
        if "lora" in model_name.lower() and model_base is None:
            warnings.warn(
                "There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged."
            )
        if "lora" in model_name.lower() and model_base is not None:
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            tokenizer = _load_tokenizer_with_retry(model_base, use_fast=False)
            rank0_print("Loading LLaVA from base model...")
            if "mixtral" in model_name.lower():
                from llava.model.language_model.llava_mixtral import LlavaMixtralConfig

                lora_cfg_pretrained = LlavaMixtralConfig.from_pretrained(model_path)
                tokenizer = _load_tokenizer_with_retry(model_base, use_fast=False)
                model = LlavaMixtralForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, attn_implementation=attn_implementation, **kwargs)
            elif "mistral" in model_name.lower():
                from llava.model.language_model.llava_mistral import LlavaMistralConfig

                lora_cfg_pretrained = LlavaMistralConfig.from_pretrained(model_path)
                tokenizer = _load_tokenizer_with_retry(model_base, use_fast=False)
                model = LlavaMistralForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, attn_implementation=attn_implementation, **kwargs)
            elif "gemma" in model_name.lower():
                from llava.model.language_model.llava_gemma import LlavaGemmaConfig

                lora_cfg_pretrained = LlavaGemmaConfig.from_pretrained(model_path)
                tokenizer = _load_tokenizer_with_retry(model_base, use_fast=False)
                model = LlavaGemmaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, attn_implementation=attn_implementation, **kwargs)
            elif "llada" in model_name.lower():
                from llava.model.language_model.llava_llada import LlavaLLaDAConfig

                lora_cfg_pretrained = LlavaLLaDAConfig.from_pretrained(model_path, trust_remote_code=True)
                tokenizer = _load_tokenizer_with_retry(model_base, use_fast=False)
                model = LlavaLLaDAModelLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, attn_implementation=attn_implementation, trust_remote_code=True, **kwargs)
            else:
                from llava.model.language_model.llava_llama import LlavaConfig

                lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
                tokenizer = _load_tokenizer_with_retry(model_base, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, attn_implementation=attn_implementation, **kwargs)

            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            rank0_print("Loading additional LLaVA weights...")
            if os.path.exists(os.path.join(model_path, "non_lora_trainables.bin")):
                non_lora_trainables = torch.load(os.path.join(model_path, "non_lora_trainables.bin"), map_location="cpu")
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download

                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(repo_id=repo_id, filename=filename, subfolder=subfolder)
                    return torch.load(cache_file, map_location="cpu")

                # non_lora_trainables = load_from_hf(model_path, "non_lora_trainables.bin")
                print("Notice: 'non_lora_trainables.bin' not found. Assuming weights are merged in adapter_model.bin/safetensors.")
                non_lora_trainables = {}
            # 处理键名：去掉 base_model. 前缀
            non_lora_trainables = {(k[11:] if k.startswith("base_model.") else k): v for k, v in non_lora_trainables.items()}
            # 处理键名：去掉 model. 前缀（如果存在）
            if any(k.startswith("model.model.") for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith("model.") else k): v for k, v in non_lora_trainables.items()}
            # 处理 mm_projector 的键名：将 modules_to_save.default 和 original_module 映射到正确的路径
            processed_non_lora = {}
            for k, v in non_lora_trainables.items():
                if "mm_projector.modules_to_save.default" in k:
                    # 将 mm_projector.modules_to_save.default.X 映射到 mm_projector.X
                    new_k = k.replace("mm_projector.modules_to_save.default", "mm_projector")
                    processed_non_lora[new_k] = v
                elif "mm_projector.original_module" in k:
                    # 将 mm_projector.original_module.X 映射到 mm_projector.X
                    new_k = k.replace("mm_projector.original_module", "mm_projector")
                    processed_non_lora[new_k] = v
                else:
                    processed_non_lora[k] = v
            model.load_state_dict(processed_non_lora, strict=False)

            from peft import PeftModel
            import json
            import shutil

            rank0_print("Loading LoRA weights...")
            # 临时修改适配器配置，移除 modules_to_save 以避免加载时的键名检查错误
            adapter_config_path = os.path.join(model_path, "adapter_config.json")
            backup_config_path = adapter_config_path + ".backup"
            if os.path.exists(adapter_config_path):
                # 备份原始配置
                shutil.copy(adapter_config_path, backup_config_path)
                # 读取并修改配置
                with open(adapter_config_path, 'r') as f:
                    adapter_config = json.load(f)
                original_modules_to_save = adapter_config.get("modules_to_save", [])
                # 临时移除 modules_to_save
                if "mm_projector" in original_modules_to_save:
                    adapter_config["modules_to_save"] = []
                    with open(adapter_config_path, 'w') as f:
                        json.dump(adapter_config, f, indent=2)
            
            try:
                model = PeftModel.from_pretrained(model, model_path, trust_remote_code=True)
            finally:
                # 恢复原始配置
                if os.path.exists(backup_config_path):
                    shutil.move(backup_config_path, adapter_config_path)
            
            rank0_print("Merging LoRA weights...")
            model = model.merge_and_unload()
            rank0_print("Model is loaded...")
        elif model_base is not None:  # this may be mm projector only, loading projector with preset language mdoel
            rank0_print(f"Loading LLaVA from base model {model_base}...")
            if "mixtral" in model_name.lower():
                tokenizer = _load_tokenizer_with_retry(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaMixtralForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, attn_implementation=attn_implementation, **kwargs)
            elif "mistral" in model_name.lower() or "zephyr" in model_name.lower():
                tokenizer = _load_tokenizer_with_retry(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaMistralForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, attn_implementation=attn_implementation, **kwargs)
            elif "gemma" in model_name.lower():
                tokenizer = _load_tokenizer_with_retry(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaGemmaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, attn_implementation=attn_implementation, **kwargs)
            elif (
                "wizardlm-2" in model_name.lower()
                and "vicuna" in model_name.lower()
                or "llama" in model_name.lower()
                or "yi" in model_name.lower()
                or "nous-hermes" in model_name.lower()
                or "llava-v1.6-34b" in model_name.lower()
                or "llava-v1.5" in model_name.lower()
            ):
                from llava.model.language_model.llava_llama import LlavaConfig

                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                if customized_config is None:
                    llava_cfg = LlavaConfig.from_pretrained(model_path)
                    if "v1.5" in model_name.lower():
                        llava_cfg.delay_load = True  # a workaround for correctly loading v1.5 models
                else:
                    llava_cfg = customized_config

                tokenizer = _load_tokenizer_with_retry(model_base, use_fast=False)
                llava_cfg = LlavaConfig.from_pretrained(model_path)
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=llava_cfg, **kwargs)
            elif "llada" in model_name.lower():
                from llava.model.language_model.llava_llada import LlavaLLaDAConfig

                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                if customized_config is None:
                    llada_cfg = LlavaLLaDAConfig.from_pretrained(model_path, trust_remote_code=True)
                else:
                    llada_cfg = customized_config

                if overwrite_config is not None:
                    rank0_print(f"Overwriting config with {overwrite_config}")
                    for k, v in overwrite_config.items():
                        setattr(llada_cfg, k, v)

                tokenizer = _load_tokenizer_with_retry(model_base, use_fast=False)
                llada_cfg = LlavaLLaDAConfig.from_pretrained(model_path, trust_remote_code=True)
                model = LlavaLLaDAModelLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=llada_cfg, trust_remote_code=True, **kwargs)
            else:
                raise ValueError(f"Model {model_name} not supported")

            mm_projector_weights = torch.load(os.path.join(model_path, "mm_projector.bin"), map_location="cpu")
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            rank0_print(f"Loaded LLaVA model: {model_path}")
            if "mixtral" in model_name.lower():
                from llava.model.language_model.llava_mixtral import LlavaMixtralConfig

                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                if customized_config is None:
                    llava_cfg = LlavaMixtralConfig.from_pretrained(model_path)
                else:
                    llava_cfg = customized_config

                if overwrite_config is not None:
                    rank0_print(f"Overwriting config with {overwrite_config}")
                    for k, v in overwrite_config.items():
                        setattr(llava_cfg, k, v)

                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = LlavaMixtralForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, attn_implementation=attn_implementation, config=llava_cfg, **kwargs)

            elif "mistral" in model_name.lower() or "zephyr" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = LlavaMistralForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, attn_implementation=attn_implementation, **kwargs)
            elif (
                "wizardlm-2" in model_name.lower()
                and "vicuna" in model_name.lower()
                or "llama" in model_name.lower()
                or "yi" in model_name.lower()
                or "nous-hermes" in model_name.lower()
                or "llava-v1.6-34b" in model_name.lower()
                or "llava-v1.5" in model_name.lower()
            ):
                from llava.model.language_model.llava_llama import LlavaConfig

                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                if customized_config is None:
                    llava_cfg = LlavaConfig.from_pretrained(model_path)
                    if "v1.5" in model_name.lower():
                        llava_cfg.delay_load = True  # a workaround for correctly loading v1.5 models
                else:
                    llava_cfg = customized_config

                if overwrite_config is not None:
                    rank0_print(f"Overwriting config with {overwrite_config}")
                    for k, v in overwrite_config.items():
                        setattr(llava_cfg, k, v)

                model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, attn_implementation=attn_implementation, config=llava_cfg, **kwargs)

            elif "qwen" in model_name.lower() or "quyen" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                if "moe" in model_name.lower() or "A14B" in model_name.lower():
                    from llava.model.language_model.llava_qwen_moe import LlavaQwenMoeConfig
                    if overwrite_config is not None:
                        llava_cfg = LlavaQwenMoeConfig.from_pretrained(model_path)
                        rank0_print(f"Overwriting config with {overwrite_config}")
                        for k, v in overwrite_config.items():
                            setattr(llava_cfg, k, v)
                        model = LlavaQwenMoeForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, attn_implementation=attn_implementation, config=llava_cfg, **kwargs)
                    else:
                        model = LlavaQwenMoeForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, attn_implementation=attn_implementation, **kwargs)

                else:
                    from llava.model.language_model.llava_qwen import LlavaQwenConfig
                    if overwrite_config is not None:
                        llava_cfg = LlavaQwenConfig.from_pretrained(model_path)
                        rank0_print(f"Overwriting config with {overwrite_config}")
                        for k, v in overwrite_config.items():
                            setattr(llava_cfg, k, v)
                        model = LlavaQwenForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, attn_implementation=attn_implementation, config=llava_cfg, **kwargs)
                    else:
                        model = LlavaQwenForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, attn_implementation=attn_implementation, **kwargs)

            elif "gemma" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaGemmaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, config=cfg_pretrained, attn_implementation=attn_implementation, **kwargs)
            elif "llada" in model_name.lower():
                from llava.model.language_model.llava_llada import LlavaLLaDAConfig

                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                if customized_config is None:
                    llada_cfg = LlavaLLaDAConfig.from_pretrained(model_path, trust_remote_code=True)
                else:
                    llada_cfg = customized_config

                if overwrite_config is not None:
                    rank0_print(f"Overwriting config with {overwrite_config}")
                    for k, v in overwrite_config.items():
                        setattr(llada_cfg, k, v)

                model = LlavaLLaDAModelLM.from_pretrained(model_path, low_cpu_mem_usage=True, attn_implementation=attn_implementation, config=llada_cfg, trust_remote_code=True, **kwargs)

            else:
                try:
                    from llava.model.language_model.llava_llama import LlavaConfig

                    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                    if customized_config is None:
                        llava_cfg = LlavaConfig.from_pretrained(model_path)
                        if "v1.5" in model_path.lower():
                            llava_cfg.delay_load = True  # a workaround for correctly loading v1.5 models
                    else:
                        llava_cfg = customized_config

                    if overwrite_config is not None:
                        rank0_print(f"Overwriting config with {overwrite_config}")
                        for k, v in overwrite_config.items():
                            setattr(llava_cfg, k, v)
                    model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, attn_implementation=attn_implementation, config=llava_cfg, **kwargs)
                except:
                    raise ValueError(f"Model {model_name} not supported")

    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel

            tokenizer = _load_tokenizer_with_retry(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print("Convert to FP16...")
            model.to(torch.float16)
        else:
            use_fast = False
            if "mpt" in model_name.lower().replace("prompt", ""):
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    rank0_print(f"Model Class: {model.__class__.__name__}")
    image_processor = None

    if "llava" in model_name.lower() or is_multimodal:
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=device_map)
        if device_map != "auto":
            vision_tower.to(device="cuda", dtype=torch.float16)
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    elif hasattr(model.config, "max_position_embeddings"):
        context_len = model.config.max_position_embeddings
    elif hasattr(model.config, "tokenizer_model_max_length"):
        context_len = model.config.tokenizer_model_max_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
