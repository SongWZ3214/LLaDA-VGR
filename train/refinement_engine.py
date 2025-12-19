"""
智能描述细化引擎
封装了基于 Jitter 分析和视觉定位的迭代细化流程
"""
import warnings
warnings.filterwarnings("ignore")

import torch
import copy
import time
import os
import sys
import numpy as np
import logging
import gc
from PIL import Image
from typing import Optional, Tuple, Dict, Any

from crop import Config, GroundingAgent, JitterAnalyzer, TextMiner
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.hooks.fast_dllm_hook import register_fast_dllm_hook


class RefinementEngine:
    """智能描述细化引擎"""
    
    def __init__(
        self,
        model_path: str,
        model_base: str,
        model_name: str = "llava_llada_lora",
        vision_tower_path: str = "google/siglip2-so400m-patch14-384",
        device: str = "cuda:0",
        max_steps: int = 10,
        jitter_threshold: float = 0.30,
        span_k: int = 3,
        mask_expansion: int = 2,
        global_suppress_radius: int = 3,
        temp_dir: str = "./cropped_image",
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化细化引擎
        
        Args:
            model_path: 训练好的模型路径
            model_base: 基础模型路径
            model_name: 模型名称
            vision_tower_path: 视觉编码器路径
            device: 设备
            max_steps: 最大迭代次数
            jitter_threshold: Jitter 阈值
            span_k: Span 半径
            mask_expansion: Mask 扩张
            global_suppress_radius: 全局抑制半径
            temp_dir: 临时文件目录
            logger: 日志记录器
        """
        self.device = device
        self.max_steps = max_steps
        self.jitter_threshold = jitter_threshold
        self.span_k = span_k
        self.mask_expansion = mask_expansion
        self.global_suppress_radius = global_suppress_radius
        self.temp_dir = temp_dir
        self.mask_token_id = 126336
        
        # 设置日志
        if logger is None:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
                self.logger.addHandler(handler)
        else:
            self.logger = logger
        
        # 创建临时目录
        os.makedirs(temp_dir, exist_ok=True)
        
        # 加载模型
        self._load_models(model_path, model_base, model_name, vision_tower_path)
        
    def _load_models(self, model_path, model_base, model_name, vision_tower_path):
        """加载模型"""
        self.logger.info(">>> Loading LLaDA Model...")
        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(
            model_path, model_base, model_name,
            attn_implementation="sdpa",
            device_map=None,
            overwrite_config={"mm_vision_tower": vision_tower_path}
        )
        self.tokenizer.padding_side = "left"
        self.model = self.model.to(self.device)
        self.model.eval()
        register_fast_dllm_hook(self.model)
        
        # 加载 GroundingDINO
        self.logger.info(">>> Loading GroundingDINO (Standing by on CPU)...")
        dino_config = Config()
        dino_config.SPAN_K = self.span_k
        self.grounder = GroundingAgent(dino_config, logger=self.logger)
        self.grounder.load()
        if self.grounder.model:
            self.grounder.model.to("cpu")
    
    def _flush(self):
        """强制清理显存"""
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    def _move_model_to_cpu(self, model_obj):
        """将模型移动到 CPU"""
        if model_obj:
            model_obj.to("cpu")
            self._flush()
    
    def _move_model_to_gpu(self, model_obj):
        """将模型移动回 GPU"""
        if model_obj:
            model_obj.to(self.device)
    
    def _expand2square(self, pil_img, background_color):
        """将图片扩展为正方形"""
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result
    
    def _map_char_span_to_token_span(self, token_ids, tokenizer, char_start, char_end):
        """将字符级范围映射回 Token 索引范围"""
        current_char_pos = 0
        token_start = None
        token_end = None

        for i, tid in enumerate(token_ids):
            token_text = tokenizer.decode([tid], skip_special_tokens=False)
            token_len = len(token_text)
            
            t_s = current_char_pos
            t_e = current_char_pos + token_len
            
            if t_e > char_start and t_s < char_end:
                if token_start is None:
                    token_start = i
                token_end = i + 1
            
            current_char_pos += token_len
            
            if current_char_pos >= char_end and token_start is not None:
                break
                
        if token_start is None:
            return None, None
            
        return token_start, token_end
    
    def refine(
        self,
        image_path: str,
        base_instruction: str = "Please describe the image in detail. Use less absolute directional descriptions.",
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        对图像进行智能描述细化
        
        Args:
            image_path: 图像路径
            base_instruction: 基础指令
            **kwargs: 其他参数
            
        Returns:
            (final_response, metadata): 最终响应和元数据
        """
        # 加载图像
        original_image_obj = Image.open(image_path).convert("RGB")
        
        # 状态变量
        last_intermediate_confidences = None
        current_generated_ids = []
        global_covered_indices = set()
        initial_jitter = []
        target_span_range = None
        
        self.logger.info(f"{'=' * 20} Start Intelligent Loop (Threshold: {self.jitter_threshold}) {'=' * 20}")
        
        # 主循环
        for step_idx in range(self.max_steps + 1):
            # 分析与决策阶段
            if step_idx > 0:
                self.logger.info(f"\n>>> [Phase {step_idx}] Analyzing Uncertainty...")
                
                if last_intermediate_confidences:
                    history_data = [{'confidences': [float(c) for c in step.get('confidences', [])]}
                                    for step in last_intermediate_confidences]
                    
                    del last_intermediate_confidences
                    last_intermediate_confidences = None
                    self._flush()
                    
                    conf_matrix, _ = JitterAnalyzer.extract_confidence_data(history_data)
                    jitter_values = JitterAnalyzer.calculate_jitter_numpy(conf_matrix)
                    del conf_matrix, history_data
                    
                if step_idx == 1:
                    initial_jitter = jitter_values
                else:
                    if len(initial_jitter) > 0 and len(jitter_values) == len(initial_jitter) and target_span_range:
                        s, e = target_span_range
                        initial_jitter[s:e] = jitter_values[s:e]
                        jitter_values = initial_jitter.copy()
                    else:
                        self.logger.warning("  -> Initial jitter and current jitter have different lengths. Skipping.")
                        continue
                
                if len(jitter_values) == 0:
                    break
                
                # 应用全局 Mask
                if global_covered_indices:
                    suppressed_mask = np.zeros_like(jitter_values, dtype=bool)
                    for idx in global_covered_indices:
                        if idx < len(jitter_values):
                            suppressed_mask[idx] = True
                    jitter_values[suppressed_mask] = 0.0
                    self.logger.info(f"  -> Suppressed {sum(suppressed_mask)} tokens based on history.")
                
                # 智能选点逻辑
                sorted_indices = np.argsort(jitter_values)[::-1]
                selected_idx = None
                target_phrase = None
                target_span_range = None
                
                full_text = self.tokenizer.decode(current_generated_ids, skip_special_tokens=True)
                nlp = TextMiner.get_nlp()
                doc = nlp(full_text)
                
                self.logger.info(f"  -> Searching for valid refinement target (NOUN/ADJ)...")
                
                for idx in sorted_indices:
                    if jitter_values[idx] < self.jitter_threshold:
                        break
                    
                    prefix_ids = current_generated_ids[:idx]
                    prefix_text = self.tokenizer.decode(prefix_ids, skip_special_tokens=True)
                    char_offset = len(prefix_text)
                    
                    chunk_text, char_span = TextMiner.analyze_token_for_refinement(doc, char_offset)
                    
                    if chunk_text:
                        selected_idx = idx
                        target_phrase = chunk_text
                        max_val = jitter_values[idx]
                        
                        self.logger.info(f"  -> Valid Target Found: '{self.tokenizer.decode([current_generated_ids[idx]])}' (POS Valid). Expanding to chunk: '{target_phrase}'")
                        
                        start_char, end_char = char_span
                        s_start, s_end = self._map_char_span_to_token_span(
                            current_generated_ids, self.tokenizer, start_char, end_char
                        )
                        
                        if s_start is None:
                            self.logger.warning(f"     [Error] Failed to map char span to tokens. Skipping.")
                            continue
                        
                        target_span_range = [s_start, s_end]
                        break
                    else:
                        self.logger.debug(f"     [Skip] Token '{self.tokenizer.decode([current_generated_ids[idx]])}' is not a valid target.")
                
                if selected_idx is None:
                    self.logger.info(f"  -> No valid refinement target found. Stability reached.")
                    break
                
                self.logger.info(f"  -> Selected Focus: '{target_phrase}' (Jitter: {max_val:.4f})")
                
                # Grounding
                self._move_model_to_gpu(self.grounder.model)
                crop_path, dino_conf = self.grounder.run_grounding(target_phrase, self.temp_dir, image_path)
                self._move_model_to_cpu(self.grounder.model)
                
                # 加载 Crop
                if crop_path:
                    crop_img = Image.open(crop_path).convert("RGB")
                    if hasattr(self.image_processor, 'image_mean') and self.image_processor.image_mean is not None:
                        mean = self.image_processor.image_mean
                        bg_color = tuple(int(x * 255) for x in mean)
                    else:
                        bg_color = (122, 116, 104)
                    crop_img = self._expand2square(crop_img, bg_color)
                    active_images = [crop_img]
                else:
                    if dino_conf:
                        self.logger.info(f"  -> [REJECT] Visual evidence weak (Conf: {dino_conf:.2f} < 0.50). Skipping.")
                    else:
                        self.logger.info("  -> No crop found. SKIPPING refinement for this step.")
                    if target_span_range:
                        for i in range(target_span_range[0], target_span_range[1]):
                            global_covered_indices.add(i)
                    continue
            else:
                self.logger.info(f"\n>>> [Phase {step_idx}] Initial Generation...")
                active_images = [original_image_obj]
            
            # 构建 Prompt
            self._flush()
            
            image_tensor = process_images(active_images, self.image_processor, self.model.config)
            if isinstance(image_tensor, list):
                image_tensor = [_img.to(dtype=torch.float16, device=self.device) for _img in image_tensor]
            else:
                image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)
            
            img_tokens_str = " ".join([DEFAULT_IMAGE_TOKEN] * len(active_images))
            kwargs_text_ids = {}
            
            if step_idx == 0:
                prompt_text = img_tokens_str + "\n" + base_instruction
            else:
                masked_ids = current_generated_ids.copy()
                
                if target_span_range:
                    s = target_span_range[0]
                    e = target_span_range[1]
                    for _ in range(self.mask_expansion):
                        if s > 0:
                            prev_token_id = current_generated_ids[s - 1]
                            prev_token = self.tokenizer.decode([prev_token_id])
                            if '.' in prev_token or '!' in prev_token or '?' in prev_token or ',' in prev_token:
                                break
                            s -= 1
                    for _ in range(self.mask_expansion):
                        if e < len(current_generated_ids):
                            next_token_id = current_generated_ids[e]
                            next_token = self.tokenizer.decode([next_token_id])
                            if '.' in next_token or '!' in next_token or '?' in next_token or ',' in next_token:
                                break
                            e += 1
                    
                    original_text_segment = self.tokenizer.decode(masked_ids[s:e], skip_special_tokens=False)
                    self.logger.info(f"  -> Applied Masking on: '{original_text_segment}' (Indices: {s}-{e})")
                    
                    for i in range(s, e):
                        masked_ids[i] = self.mask_token_id
                        global_covered_indices.add(i)
                
                text_token_ids_tensor = torch.tensor([masked_ids], device=self.device)
                kwargs_text_ids = {"text_token_ids": text_token_ids_tensor}
                
                refine_instruction = (
                    "\nYou are presented with a zoomed-in visual detail and a text description with missing parts. "
                    "Task: Analyze the specific visual attributes (such as texture, pattern, color, and object shape) "
                    "in the provided image crop. Instruction: The masked part of the text describes this specific visual evidence. "
                    "Do not guess based on common language patterns. Instead, look closely at the image crop to accurately "
                    "restore the missing text. What you see in the crop is the ground truth."
                )
                prompt_text = img_tokens_str + refine_instruction
            
            # 对话模版处理
            conv = copy.deepcopy(conv_templates["llava_llada"])
            conv.append_message(conv.roles[0], prompt_text)
            conv.append_message(conv.roles[1], None)
            prompt_ids = tokenizer_image_token(conv.get_prompt(), self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
            
            # 模型推理
            self._flush()
            
            try:
                with torch.inference_mode():
                    result = self.model.generate(
                        prompt_ids,
                        images=image_tensor,
                        image_sizes=[img.size for img in active_images],
                        steps=128,
                        gen_length=128,
                        block_length=128,
                        tokenizer=self.tokenizer,
                        stopping_criteria=['<|eot_id|>'],
                        return_confidences=True,
                        save_confidence_interval=1,
                        is_initial_generation=(step_idx == 0),
                        repetition_penalty=1.15,
                        **kwargs_text_ids
                    )
            except RuntimeError as e:
                if "out of memory" in str(e):
                    self.logger.error(f"!!! CUDA OOM at Step {step_idx}. Exiting.")
                    if 'last_intermediate_confidences' in locals():
                        del last_intermediate_confidences
                    del image_tensor, prompt_ids
                    self._flush()
                    break
                else:
                    raise e
            
            # 结果处理
            if isinstance(result, tuple):
                if len(result) == 3:
                    cont, _, intermediate_confidences = result
                elif len(result) == 2:
                    cont, _ = result
                    intermediate_confidences = None
            else:
                cont = result
                intermediate_confidences = None
            
            last_intermediate_confidences = intermediate_confidences
            
            if hasattr(cont, 'cpu'):
                current_generated_ids = cont[0].cpu().numpy().tolist()
            elif isinstance(cont, list):
                current_generated_ids = cont[0] if len(cont) > 0 else []
            
            if current_generated_ids:
                text_res = self.tokenizer.decode(current_generated_ids, skip_special_tokens=True)
                self.logger.info(f"--> Result: {text_res.replace(chr(10), ' ')}")
            
            # 清理
            del result, cont, image_tensor, prompt_ids
            if 'text_token_ids_tensor' in locals():
                del text_token_ids_tensor
            if 'crop_img' in locals():
                del crop_img
            self._flush()
        
        # 生成最终响应
        if current_generated_ids:
            final_response = self.tokenizer.decode(current_generated_ids, skip_special_tokens=True)
        else:
            final_response = ""
        
        metadata = {
            "total_steps": step_idx + 1,
            "final_step": step_idx
        }
        
        self.logger.info(f"\n{'=' * 20} Loop Completed {'=' * 20}")
        
        return final_response, metadata

