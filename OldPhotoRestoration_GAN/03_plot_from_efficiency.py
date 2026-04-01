import matplotlib.pyplot as plt
import numpy as np
import os

# 设置绘图风格
plt.style.use('seaborn-v0_8-muted')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_resolution_analysis(report_path, output_image):
    resolutions = []
    avg_times = []

    # 1. 读取 TXT 文件数据
    if not os.path.exists(report_path):
        print(f"❌ 找不到报告文件: {report_path}")
        return

    with open(report_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        # 跳过前两行表头
        for line in lines[2:]:
            parts = line.split('|')
            if len(parts) >= 5:
                res = int(parts[0].strip())
                avg_t = float(parts[-1].strip())
                resolutions.append(res)
                avg_times.append(avg_t)

    # 2. 开始绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # --- 图表 1: 分辨率 vs 推理耗时 (折线图) ---
    ax1.plot(resolutions, avg_times, marker='o', linestyle='-', color='royalblue', linewidth=2, markersize=8)
    ax1.fill_between(resolutions, avg_times, color='royalblue', alpha=0.1)

    ax1.set_title('分辨率与推理耗时增长趋势', fontsize=14, pad=15)
    ax1.set_xlabel('输入分辨率 (px)', fontsize=12)
    ax1.set_ylabel('平均耗时 (秒)', fontsize=12)
    ax1.set_xticks(resolutions)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # 在点上标注具体数值
    for i, txt in enumerate(avg_times):
        ax1.annotate(f'{txt:.2f}s', (resolutions[i], avg_times[i]),
                     textcoords="offset points", xytext=(0, 10), ha='center', fontweight='bold')

    # --- 图表 2: 单位像素处理压力 (柱状图) ---
    # 计算每 100w 像素需要的处理时间，观察算法在高压下的表现
    megapixels = [(r * r) / 1e6 for r in resolutions]
    time_per_mp = [t / mp for t, mp in zip(avg_times, megapixels)]

    bars = ax2.bar([str(r) for r in resolutions], time_per_mp, color='mediumseagreen', alpha=0.8, width=0.5)
    ax2.set_title('单位像素处理开销 (秒/百万像素)', fontsize=14, pad=15)
    ax2.set_ylabel('耗时/MP', fontsize=12)

    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"✅ 可视化图表已保存至: {output_image}")
    plt.show()


if __name__ == "__main__":
    report = r'E:\OldPhotoRestoration_GAN\data\outputs\res_efficiency_report.txt'
    save_path = r'E:\OldPhotoRestoration_GAN\data\outputs\res_analysis_charts.png'
    plot_resolution_analysis(report, save_path)