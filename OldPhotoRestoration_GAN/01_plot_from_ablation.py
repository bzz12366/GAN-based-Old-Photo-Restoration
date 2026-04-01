import matplotlib.pyplot as plt
import os
import re

# 1. 路径设置
report_path = r'E:\OldPhotoRestoration_GAN\data\outputs\ablation_study_report.txt'
save_path = r'E:\OldPhotoRestoration_GAN\ablation_comparison.png'


def generate_plot():
    if not os.path.exists(report_path):
        print(f"❌ 错误：找不到文件 {report_path}")
        return

    # 准备存储数据的列表
    labels = []
    psnr_list = []
    ssim_list = []

    # 2. 读取并解析文件内容 (针对新格式优化)
    with open(report_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 修改后的正则表达式：匹配 字母/下划线标签 | 数字 | 数字
            # 例如: GFPGAN_Only | 39.10 | 0.9719
            match = re.search(r'([\w_-]+)\s*\|\s*(\d+\.\d+)\s*\|\s*(\d+\.\d+)', line)
            if match:
                tag = match.group(1).strip()
                # 排除表头行
                if tag in ["实验组别", "Average"]:
                    continue
                labels.append(tag)
                psnr_list.append(float(match.group(2)))
                ssim_list.append(float(match.group(3)))

    if not labels:
        print("❌ 错误：解析失败，请确认 TXT 文件内容是否符合格式。")
        return

    # 3. 绘图配置
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示
    plt.rcParams['axes.unicode_minus'] = False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    colors = ['#8ecfc9', '#ffbe7a', '#fa7e6f']  # 学术配色

    # --- 左图：PSNR ---
    bars1 = ax1.bar(labels, psnr_list, color=colors, width=0.5)
    ax1.set_title('PSNR Comparison (dB) ↑', fontsize=12, fontweight='bold')
    # 动态设置 Y 轴范围，留出顶部空间显示标签
    ax1.set_ylim(min(psnr_list) * 0.8, max(psnr_list) * 1.15)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    ax1.tick_params(axis='x', rotation=15)  # 标签稍微旋转，防止重叠

    # --- 右图：SSIM ---
    bars2 = ax2.bar(labels, ssim_list, color=colors, width=0.5)
    ax2.set_title('SSIM Comparison ↑', fontsize=12, fontweight='bold')
    ax2.set_ylim(min(ssim_list) * 0.95, 1.0)
    ax2.grid(axis='y', linestyle='--', alpha=0.4)
    ax2.tick_params(axis='x', rotation=15)

    # 4. 自动标注数值函数
    def add_tags(bars, ax, fmt):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h, f'{h:{fmt}}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    add_tags(bars1, ax1, '.2f')
    add_tags(bars2, ax2, '.4f')

    # 5. 保存结果
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✅ 图表已根据新数据格式生成并保存至: {save_path}")
    plt.show()


if __name__ == "__main__":
    generate_plot()