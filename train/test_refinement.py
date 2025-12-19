#!/usr/bin/env python3
"""
简单的测试脚本 - 处理单条数据
"""
import logging
import sys
import os

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
    """主函数"""
    # 配置参数
    model_path = "/data0/swz/LLaDA-VGR/train/exp/llada_vgr_lora_rank64"
    model_base = "jiyatai/ReDiff"
    model_name = "llava_llada_lora"
    vision_tower_path = "google/siglip2-so400m-patch14-384"
    device = "cuda:0"
    
    # 测试数据
    image_path = "/data0/swz/LLaDA-VGR/train/test2.jpg"
    base_instruction = "Please describe the image in detail. Use less absolute directional descriptions."
    
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
            max_steps=10,
            jitter_threshold=0.30,
            span_k=3,
            mask_expansion=2,
            global_suppress_radius=3,
            temp_dir="./cropped_image",
            logger=logger
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

