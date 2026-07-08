#!/usr/bin/env python3
"""
Simple test script for one sample.
"""
import logging
import sys
import os
from pathlib import Path

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from refinement_engine import RefinementEngine

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('refinement_test.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)


def main():
    repo_root = Path(__file__).resolve().parents[1]
    workspace_root = repo_root.parent

    model_path = os.environ.get("PRETRAINED_MODEL", str(repo_root / "train" / "exp" / "llada_v_lora_rank64_1227"))
    model_base = "GSAI-ML/LLaDA-V"
    model_name = "llava_llada_lora"
    vision_tower_path = "google/siglip2-so400m-patch14-384"
    device = "cuda:0"
    IMAGE_INPUT_MODE = "both"
    MASK_MODE = "span"
    TOKEN_SELECTION_MODE = "jitter"
    LOCAL_REFINEMENT_MODE = "crop"

    image_path = os.environ.get(
        "TEST_IMAGE",
        str(workspace_root / "exp" / "MMHal-Bench" / "images" / "16189396430_4dce91a9d7_o.jpg"),
    )
    base_instruction = "Please describe the image in detail. Use less absolute directional descriptions. Do not repeat information."
    
    logger.info("=" * 60)
    logger.info("Starting Refinement Test")
    logger.info(f"Image: {image_path}")
    logger.info(f"Instruction: {base_instruction}")
    logger.info("=" * 60)
    
    try:
        # 初始化引擎
        engine = RefinementEngine(
            model_path=model_path,
            model_base=model_base,
            model_name=model_name,
            vision_tower_path=vision_tower_path,
            device=device,
            max_steps=6,
            jitter_threshold=0.35,
            mask_expansion=2,
            temp_dir="./cropped_image",
            logger=logger,
            image_input_mode=IMAGE_INPUT_MODE,
            mask_mode=MASK_MODE,
            token_selection_mode=TOKEN_SELECTION_MODE,
            local_refinement_mode=LOCAL_REFINEMENT_MODE
        )
        
        # 执行细化
        final_response, metadata = engine.refine(
            image_path=image_path,
            base_instruction=base_instruction
        )
        
        # 输出结果
        logger.info("\n" + "=" * 60)
        logger.info("FINAL RESULT")
        logger.info("=" * 60)
        logger.info(f"Response:\n{final_response}")
        logger.info(f"\nMetadata: {metadata}")
        logger.info("=" * 60)
        
        print("\n" + "=" * 60)
        print("FINAL RESULT")
        print("=" * 60)
        print(f"Response:\n{final_response}")
        print(f"\nMetadata: {metadata}")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Error during refinement: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

