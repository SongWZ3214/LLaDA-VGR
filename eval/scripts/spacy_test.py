import spacy
import time
import torch


def profile_spacy_gpu(model_name="en_core_web_trf", text_multiplier=2):

    print("\n" + "=" * 50)
    print(f"准备测试 spaCy 模型: {model_name} (GPU 模式)")

    # 1. 强制启用 GPU
    try:
        spacy.require_gpu()
        print("✅ 成功激活 spaCy GPU 模式。")
    except Exception as e:
        print("❌ 无法激活 GPU，请确保安装了正确的 CUDA 版本以及 spacy[cuda] 库。")
        print(f"错误信息: {e}")
        return

    # 构造测试文本（模拟长文本）
    base_text = "Apple is looking at buying U.K. startup for $1 billion. hahaha. "
    text = base_text * text_multiplier
    print(f"📄 测试文本总长度: {len(text)} 字符")

    # ==================== 监控模型加载 ====================
    print("\n[阶段 1] 模型加载中...")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    start_load_time = time.perf_counter()

    try:
        nlp = spacy.load(model_name)
    except OSError:
        print(f"❌ 找不到模型 '{model_name}'。")
        print(f"请先运行: python -m spacy download {model_name}")
        return

    torch.cuda.synchronize()  # 确保 GPU 操作完成
    end_load_time = time.perf_counter()

    load_latency = end_load_time - start_load_time
    peak_load_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)

    print(f"⏱️ 模型加载耗时: {load_latency:.4f} 秒")
    print(f"📈 模型加载占用显存: {peak_load_mem:.2f} MB")

    # ==================== 监控推理阶段 ====================
    print("\n[阶段 2] 开始处理文本...")

    # 清理缓存并重置显存峰值记录，以便准确测量推理阶段的显存增量
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    start_infer_time = time.perf_counter()

    # 运行 spaCy 管道
    doc = nlp(text)

    torch.cuda.synchronize()  # 强制等待 GPU 推理计算完毕
    end_infer_time = time.perf_counter()

    infer_latency = end_infer_time - start_infer_time
    peak_infer_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)

    print(f"⏱️ 文本推理耗时: {infer_latency:.4f} 秒")
    print(f"📈 推理过程峰值显存 (模型+激活值): {peak_infer_mem:.2f} MB")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    # 默认测试英文 Transformer 模型，文本放大 100 倍。
    # 如果你用的是中文，可以改为 "zh_core_web_trf"
    profile_spacy_gpu(model_name="en_core_web_trf", text_multiplier=2)