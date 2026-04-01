import os
import cv2
import torch
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact


class RealESRGANOnlySystem:
    def __init__(self, base_dir=r'E:\OldPhotoRestoration_GAN'):
        self.base_dir = base_dir
        self.input_dir = os.path.join(base_dir, 'data', 'inputs')
        # 输出路径：data/outputs/ablation/Real-ESRGAN
        self.output_dir = os.path.join(base_dir, 'data', 'outputs', 'ablation', 'Real-ESRGAN')
        os.makedirs(self.output_dir, exist_ok=True)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 Real-ESRGAN 单模型超分启动 | 设备: {self.device}")
        self._init_models()

    def _init_models(self):
        print("📦 正在加载 Real-ESRGAN 权重 (general-x4v3)...")
        weight_dir = os.path.join(self.base_dir, 'weights')
        model_path = os.path.join(weight_dir, 'realesr-general-x4v3.pth')

        # 显式定义模型架构以防止 'NoneType' 错误
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')

        # 初始化 RealESRGANer
        self.upsampler = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=model,
            tile=400,
            half=True if self.device == 'cuda' else False
        )

    def run(self):
        valid_extensions = ('.jpg', '.png', '.jpeg')
        img_list = sorted([f for f in os.listdir(self.input_dir) if f.lower().endswith(valid_extensions)])

        if not img_list:
            print("❌ 未在输入目录发现图片。")
            return

        for img_name in img_list:
            print(f"📸 正在处理 (仅背景增强): {img_name}")
            img_path = os.path.join(self.input_dir, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            try:
                # 仅进行背景增强/超分辨率重建
                # outscale=4 表示输出分辨率提升4倍
                restored_img, _ = self.upsampler.enhance(img, outscale=4)

                # 保存结果，不修改文件名
                save_path = os.path.join(self.output_dir, img_name)
                cv2.imwrite(save_path, restored_img)

            except Exception as e:
                print(f"❌ 处理 {img_name} 时出错: {e}")

        print(f"✅ 处理完成！结果存放在: {self.output_dir}")


if __name__ == '__main__':
    system = RealESRGANOnlySystem()
    system.run()