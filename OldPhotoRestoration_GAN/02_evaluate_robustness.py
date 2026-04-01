import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def evaluate_robustness(base_dir, report_path):
    # 1. 路径设置
    input_dir = os.path.join(base_dir, r'data\inputs')
    # 修复后的图片根目录
    restored_root = os.path.join(base_dir, r'data\outputs\Robustness\Restored_Images')

    # 定义需要评估的三个等级
    levels = ["Light", "Medium", "Heavy"]
    all_metrics = {level: [] for level in levels}

    # 获取原始图片列表
    valid_exts = ('.jpg', '.png', '.jpeg')
    img_list = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)]

    print(f"📊 开始鲁棒性评估（共 {len(img_list)} 张原始参考图）...")

    for level in levels:
        current_level_dir = os.path.join(restored_root, level)
        if not os.path.exists(current_level_dir):
            print(f"⚠️ 跳过等级 {level}：找不到文件夹 {current_level_dir}")
            continue

        print(f"🔎 正在评估等级: {level}...")

        for img_name in img_list:
            # 读取原图 (Ground Truth)
            gt_img = cv2.imread(os.path.join(input_dir, img_name))
            # 读取对应等级下的修复图
            res_path = os.path.join(current_level_dir, img_name)

            if gt_img is None or not os.path.exists(res_path):
                continue

            res_img = cv2.imread(res_path)
            if res_img is None: continue

            # 尺寸对齐 (以原图为准)
            if gt_img.shape != res_img.shape:
                res_img = cv2.resize(res_img, (gt_img.shape[1], gt_img.shape[0]))

            # 转为 Y 通道计算
            gt_y = cv2.cvtColor(gt_img, cv2.COLOR_BGR2YCrCb)[:, :, 0]
            res_y = cv2.cvtColor(res_img, cv2.COLOR_BGR2YCrCb)[:, :, 0]

            # 计算指标
            p_val = psnr(gt_y, res_y)
            s_val = ssim(gt_y, res_y)
            all_metrics[level].append((p_val, s_val))

    # 2. 生成评估报告
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("图像修复鲁棒性实验评估报告\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'干扰等级':<15} | {'平均 PSNR':<12} | {'平均 SSIM':<10}\n")
        f.write("-" * 60 + "\n")

        for level in levels:
            metrics = all_metrics[level]
            if metrics:
                avg_p = np.mean([m[0] for m in metrics])
                avg_s = np.mean([m[1] for m in metrics])
                f.write(f"{level:<15} | {avg_p:<12.2f} | {avg_s:<10.4f}\n")
            else:
                f.write(f"{level:<15} | {'N/A':<12} | {'N/A':<10}\n")

    print(f"✅ 评估完成！鲁棒性报告已保存至: {report_path}")


if __name__ == '__main__':
    base_path = r'E:\OldPhotoRestoration_GAN'
    report = os.path.join(base_path, r'data\outputs\robustness_evaluation_report.txt')

    evaluate_robustness(base_path, report)