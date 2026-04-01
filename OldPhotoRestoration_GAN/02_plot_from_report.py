import matplotlib.pyplot as plt
import numpy as np
import os
import re


def plot_robustness_from_file():
    # --- 1. 路径配置 ---
    base_dir = r'E:\OldPhotoRestoration_GAN\data\outputs'
    report_path = os.path.join(base_dir, 'robustness_evaluation_report.txt')
    output_image = os.path.join(base_dir, 'robustness_chart.png')

    if not os.path.exists(report_path):
        print(f"❌ 错误：找不到文件 {report_path}")
        return

    # --- 2. 解析文本文件 ---
    levels = []
    psnr_vals = []
    ssim_vals = []

    print(f"📖 正在读取报告: {report_path}")
    with open(report_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 使用正则表达式匹配包含等级和数字的行
            # 匹配格式示例: Light | 28.36 | 0.6396
            match = re.search(r'(\w+)\s*\|\s*([\d\.]+)\s*\|\s*([\d\.]+)', line)
            if match:
                level_name = match.group(1)
                if level_name in ['Light', 'Medium', 'Heavy']:
                    levels.append(level_name)
                    psnr_vals.append(float(match.group(2)))
                    ssim_vals.append(float(match.group(3)))

    if not levels:
        print("❌ 错误：未能在文件中解析到有效数据，请检查文件格式。")
        return

    # --- 3. 绘图设置 ---
    plt.rcParams.update({'font.size': 11, 'font.family': 'sans-serif'})
    # 采用学术论文常用的淡雅配色
    colors = ['#6699CC', '#88CCEE', '#CC6677']  # 蓝色系到红色的过渡

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- 左图: PSNR ---
    bars1 = ax1.bar(levels, psnr_vals, color=colors, edgecolor='black', alpha=0.8, width=0.5)
    ax1.set_title('Average PSNR (Higher is better)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('PSNR (dB)')
    ax1.set_ylim(0, max(psnr_vals) * 1.2)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)

    # 添加数值标注
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                 f'{height:.2f}', ha='center', va='bottom', fontweight='bold')

    # --- 右图: SSIM ---
    bars2 = ax2.bar(levels, ssim_vals, color=colors, edgecolor='black', alpha=0.8, width=0.5)
    ax2.set_title('Average SSIM (Higher is better)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('SSIM Value')
    ax2.set_ylim(0, 1.0)  # SSIM 范围固定为 0-1
    ax2.grid(axis='y', linestyle='--', alpha=0.3)

    # 添加数值标注
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                 f'{height:.4f}', ha='center', va='bottom', fontweight='bold')

    # --- 4. 保存结果 ---
    plt.tight_layout()
    plt.savefig(output_image, dpi=300)  # 300DPI 满足大部分期刊打印要求
    print(f"✅ 图表已生成：{output_image}")
    # plt.show() # 如果需要直接弹出查看，取消此行注释


if __name__ == "__main__":
    plot_robustness_from_file()