import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def evaluate_ablation_study(base_dir, report_path):
    # 定义子文件夹路径
    input_dir = os.path.join(base_dir, r'data\inputs')
    output_root = os.path.join(base_dir, r'data\outputs\ablation')

    folders = {
        "GFPGAN_Only": os.path.join(output_root, 'GFPGAN'),
        "RealESRGAN_Only": os.path.join(output_root, 'Real-ESRGAN'),
        "Full_Pipeline": os.path.join(output_root, 'Full')
    }

    all_metrics = {key: [] for key in folders.keys()}

    # 以输入文件夹的图片列表为基准
    valid_exts = ('.jpg', '.png', '.jpeg')
    img_list = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)]

    print(f"📊 正在从三个子目录读取图片并计算指标...")

    for img_name in img_list:
        # 1. 读取原图 (作为 Ground Truth)
        gt_img = cv2.imread(os.path.join(input_dir, img_name))
        if gt_img is None: continue
        gt_y = cv2.cvtColor(gt_img, cv2.COLOR_BGR2YCrCb)[:, :, 0]

        # 2. 遍历三个实验组文件夹
        for exp_name, folder_path in folders.items():
            res_path = os.path.join(folder_path, img_name)
            if not os.path.exists(res_path):
                continue

            res_img = cv2.imread(res_path)
            if res_img is None: continue

            # 将修复后的图缩放到与原图一致，以便计算指标
            res_resized = cv2.resize(res_img, (gt_img.shape[1], gt_img.shape[0]))
            res_y = cv2.cvtColor(res_resized, cv2.COLOR_BGR2YCrCb)[:, :, 0]

            # 计算 PSNR 和 SSIM
            p_val = psnr(gt_y, res_y)
            s_val = ssim(gt_y, res_y)
            all_metrics[exp_name].append((p_val, s_val))

    # 生成报告
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("老照片修复消融实验评估报告 (分文件夹读取版)\n")
        f.write("=" * 65 + "\n")
        f.write(f"{'实验组别':<20} | {'平均 PSNR':<12} | {'平均 SSIM':<10}\n")
        f.write("-" * 65 + "\n")

        for exp_name in folders.keys():
            metrics = all_metrics[exp_name]
            if metrics:
                avg_p = np.mean([m[0] for m in metrics])
                avg_s = np.mean([m[1] for m in metrics])
            else:
                avg_p, avg_s = 0.0, 0.0

            f.write(f"{exp_name:<20} | {avg_p:<12.2f} | {avg_s:<10.4f}\n")

    print(f"✅ 评估完成！报告已保存至: {report_path}")


if __name__ == '__main__':
    # 根目录
    base_path = r'E:\OldPhotoRestoration_GAN'
    # 报告保存位置
    report = os.path.join(base_path, r'data\outputs\ablation_study_report.txt')

    evaluate_ablation_study(base_path, report)