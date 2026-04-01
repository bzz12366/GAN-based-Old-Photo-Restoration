import os
import cv2
import torch
import time
import numpy as np
from gfpgan import GFPGANer
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact


class ResolutionEfficiencyExperiment:
    def __init__(self, base_dir=r'E:\OldPhotoRestoration_GAN'):
        self.base_dir = base_dir
        self.raw_input = os.path.join(base_dir, 'data', 'inputs')
        # 修改后的输出根目录
        self.output_root = os.path.join(base_dir, 'data', 'outputs')
        self.report_path = os.path.join(self.output_root, 'res_efficiency_report.txt')
        self.weight_dir = os.path.join(base_dir, 'weights')

        self.resolutions = [256, 512, 1024]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        os.makedirs(self.output_root, exist_ok=True)
        self._init_models()

    def _init_models(self):
        print("📦 正在初始化模型链...")
        model_v3 = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        self.upsampler = RealESRGANer(
            scale=4,
            model_path=os.path.join(self.weight_dir, 'realesr-general-x4v3.pth'),
            model=model_v3, tile=400, half=True if self.device.type == 'cuda' else False
        )
        self.full_restorer = GFPGANer(
            model_path=os.path.join(self.weight_dir, 'GFPGANv1.4.pth'),
            upscale=2, arch='clean', channel_multiplier=2, bg_upsampler=self.upsampler
        )

    def prepare_multires_data(self):
        """步骤1：生成不同分辨率的测试集，保存在 outputs 目录下"""
        print("🛠️ 正在生成多分辨率测试图...")
        if not os.path.exists(self.raw_input):
            print(f"❌ 错误：找不到原图路径 {self.raw_input}")
            return

        files = [f for f in os.listdir(self.raw_input) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        for res in self.resolutions:
            # 直接在 outputs 下创建子文件夹
            res_dir = os.path.join(self.output_root, f"input_{res}")
            os.makedirs(res_dir, exist_ok=True)

            for fname in files:
                img = cv2.imread(os.path.join(self.raw_input, fname))
                if img is None: continue
                # 统一缩放为正方形尺寸进行对比实验
                resized = cv2.resize(img, (res, res), interpolation=cv2.INTER_LANCZOS4)
                cv2.imwrite(os.path.join(res_dir, fname), resized)
        print("✅ 不同分辨率的图片已准备就绪。")

    def run_benchmark(self, num_rounds=3):
        """步骤2：多轮效率实验"""
        results = {}

        for res in self.resolutions:
            res_dir = os.path.join(self.output_root, f"input_{res}")
            if not os.path.exists(res_dir): continue

            files = os.listdir(res_dir)
            results[res] = []

            print(f"\n🚀 开始测试分辨率: {res}x{res}")

            for r in range(1, num_rounds + 1):
                round_times = []
                for fname in files:
                    img = cv2.imread(os.path.join(res_dir, fname))
                    if img is None: continue

                    if self.device.type == 'cuda': torch.cuda.synchronize()
                    start_t = time.time()

                    # 执行全流程推理
                    self.full_restorer.enhance(img, paste_back=True)

                    if self.device.type == 'cuda': torch.cuda.synchronize()
                    round_times.append(time.time() - start_t)

                avg_round_t = np.mean(round_times) if round_times else 0
                results[res].append(avg_round_t)
                print(f"  第 {r} 轮平均耗时: {avg_round_t:.4f}s")

        self._save_report(results, num_rounds)

    def _save_report(self, data, num_rounds):
        with open(self.report_path, 'w', encoding='utf-8') as f:
            header = f"{'Resolution':<12} | " + " | ".join([f"R{i + 1}(s)" for i in range(num_rounds)]) + " | Avg(s)\n"
            f.write(header)
            f.write("-" * len(header) + "\n")

            for res, times in data.items():
                line = f"{res:<12} | "
                line += " | ".join([f"{t:<7.4f}" for t in times])
                line += f" | {np.mean(times):<7.4f}\n"
                f.write(line)
        print(f"\n📊 实验对比报告已保存至: {self.report_path}")


if __name__ == "__main__":
    exp = ResolutionEfficiencyExperiment()
    # 1. 制作不同分辨率的测试素材
    exp.prepare_multires_data()
    # 2. 进行 3 轮对比实验
    exp.run_benchmark(num_rounds=3)