#!/usr/bin/env python3
"""
自动化执行 CapArena 和 AMBER 评估流程
按顺序执行：
1. CapArena_gen.py
2. caparena_auto_eval.py
3. AMBER_gen.py
4. inference.py
"""

import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 脚本路径和工作目录
SCRIPT_DIR = Path(__file__).parent.absolute()
CAPARENA_SCRIPTS_DIR = Path("/data0/swz/LLaDA-VGR/eval/scripts")
CAPARENA_DIR = Path("/data0/swz/exp/CapArena")
AMBER_DIR = Path("/data0/swz/exp/AMBER")

CAPARENA_GEN_SCRIPT = CAPARENA_SCRIPTS_DIR / "CapArena_gen.py"
AMBER_GEN_SCRIPT = CAPARENA_SCRIPTS_DIR / "AMBER_gen.py"
CAPARENA_EVAL_SCRIPT = CAPARENA_DIR / "caparena_auto_eval.py"
INFERENCE_SCRIPT = AMBER_DIR / "inference.py"

# 执行配置
CAPARENA_EVAL_ARGS = [
    "--test_model", "LLaDA-VGR-original-image",
    "--result_path", "/data0/swz/exp/CapArena/data/caparena_auto_test_original_image.json",
    "--imgs_dir", "/data0/swz/exp/CapArena/data/caparena_auto_docci_600"
]

INFERENCE_ARGS = [
    "--inference_data", "/data0/swz/exp/AMBER/data/query/llada-vgr-original-image.json",
    "--evaluation_type", "g"
]


def run_command(cmd, description, cwd=None):
    """
    运行命令并处理错误
    
    Args:
        cmd: 命令列表或字符串
        description: 命令描述
        cwd: 工作目录
    
    Returns:
        bool: 是否成功
    """
    logger.info(f"{'='*50}")
    logger.info(f"{description}")
    logger.info(f"{'='*50}")
    
    if isinstance(cmd, str):
        cmd = cmd.split()
    
    logger.info(f"执行命令: {' '.join(cmd)}")
    if cwd:
        logger.info(f"工作目录: {cwd}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd or SCRIPT_DIR,
            check=True,
            capture_output=False,  # 实时输出
            text=True
        )
        logger.info(f"✓ {description} 执行成功")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} 执行失败 (退出码: {e.returncode})")
        return False
    except FileNotFoundError:
        logger.error(f"✗ 找不到命令或脚本: {cmd[0]}")
        return False
    except Exception as e:
        logger.error(f"✗ {description} 执行出错: {e}")
        return False


def check_file_exists(filepath, description):
    """检查文件是否存在"""
    if not filepath.exists():
        logger.error(f"✗ 找不到 {description}: {filepath}")
        return False
    logger.info(f"✓ 找到 {description}: {filepath}")
    return True


def main():
    """主函数"""
    logger.info("="*60)
    logger.info("开始执行自动化评估流程")
    logger.info("="*60)
    
    start_time = datetime.now()
    
    # 检查必要文件
    logger.info("\n检查必要文件...")
    if not check_file_exists(CAPARENA_GEN_SCRIPT, "CapArena_gen.py"):
        sys.exit(1)
    if not check_file_exists(AMBER_GEN_SCRIPT, "AMBER_gen.py"):
        sys.exit(1)
    if not check_file_exists(CAPARENA_EVAL_SCRIPT, "caparena_auto_eval.py"):
        sys.exit(1)
    if not check_file_exists(INFERENCE_SCRIPT, "inference.py"):
        sys.exit(1)
    
    # 步骤1: 运行 CapArena_gen.py (在 /data0/swz/LLaDA-VGR/eval/scripts)
    logger.info("\n" + "="*60)
    logger.info("步骤 1/4: 运行 CapArena_gen.py")
    logger.info("="*60)
    if not run_command([sys.executable, str(CAPARENA_GEN_SCRIPT)], "CapArena_gen.py", cwd=CAPARENA_SCRIPTS_DIR):
        logger.error("流程中断：CapArena_gen.py 执行失败")
        sys.exit(1)
    
    # 步骤2: 运行 caparena_auto_eval.py (在 /data0/swz/exp/CapArena)
    logger.info("\n" + "="*60)
    logger.info("步骤 2/4: 运行 caparena_auto_eval.py")
    logger.info("="*60)
    cmd = [sys.executable, "caparena_auto_eval.py"] + CAPARENA_EVAL_ARGS
    if not run_command(cmd, "caparena_auto_eval.py", cwd=CAPARENA_DIR):
        logger.error("流程中断：caparena_auto_eval.py 执行失败")
        sys.exit(1)
    
    # 步骤3: 运行 AMBER_gen.py (在 /data0/swz/LLaDA-VGR/eval/scripts)
    logger.info("\n" + "="*60)
    logger.info("步骤 3/4: 运行 AMBER_gen.py")
    logger.info("="*60)
    if not run_command([sys.executable, str(AMBER_GEN_SCRIPT)], "AMBER_gen.py", cwd=CAPARENA_SCRIPTS_DIR):
        logger.error("流程中断：AMBER_gen.py 执行失败")
        sys.exit(1)
    
    # 步骤4: 运行 inference.py (在 /data0/swz/exp/AMBER)
    logger.info("\n" + "="*60)
    logger.info("步骤 4/4: 运行 inference.py")
    logger.info("="*60)
    cmd = [sys.executable, "inference.py"] + INFERENCE_ARGS
    if not run_command(cmd, "inference.py", cwd=AMBER_DIR):
        logger.error("流程中断：inference.py 执行失败")
        sys.exit(1)
    
    # 完成
    end_time = datetime.now()
    duration = end_time - start_time
    
    logger.info("\n" + "="*60)
    logger.info("所有步骤执行完成！")
    logger.info(f"总耗时: {duration}")
    logger.info("="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\n用户中断执行")
        sys.exit(1)
    except Exception as e:
        logger.error(f"未预期的错误: {e}", exc_info=True)
        sys.exit(1)

