import os
import cv2
import torch
from gfpgan import GFPGANer


class GFPGANOnlySystem:
    def __init__(self, base_dir=r'E:\OldPhotoRestoration_GAN'):
        self.base_dir = base_dir
        self.input_dir = os.path.join(base_dir, 'data', 'inputs')
        # 输出路径：data/outputs/ablation/GFPGAN
        self.output_dir = os.path.join(base_dir, 'data', 'outputs', 'ablation', 'GFPGAN')
        os.makedirs(self.output_dir, exist_ok=True)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 GFPGAN 单模型修复启动 | 设备: {self.device}")
        self._init_models()

    def _init_models(self):
        print("📦 正在加载 GFPGAN 权重...")
        weight_dir = os.path.join(self.base_dir, 'weights')
        model_path = os.path.join(weight_dir, 'GFPGANv1.4.pth')

        # 初始化 GFPGANer
        # 注意：bg_upsampler 设为 None，表示不进行背景增强
        self.restorer = GFPGANer(
            model_path=model_path,
            upscale=2,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None
        )

    def run(self):
        valid_extensions = ('.jpg', '.png', '.jpeg')
        img_list = sorted([f for f in os.listdir(self.input_dir) if f.lower().endswith(valid_extensions)])

        if not img_list:
            print("❌ 未在输入目录发现图片。")
            return

        for img_name in img_list:
            print(f"📸 正在处理 (仅人脸修复): {img_name}")
            img_path = os.path.join(self.input_dir, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            try:
                # 仅进行人脸修复
                # paste_back=True 表示将修复后的人脸粘贴回原图
                _, _, restored_img = self.restorer.enhance(
                    img,
                    has_aligned=False,
                    only_center_face=False,
                    paste_back=True
                )

                # 保存修复后的图片，不修改文件名
                save_path = os.path.join(self.output_dir, img_name)
                cv2.imwrite(save_path, restored_img)

            except Exception as e:
                print(f"❌ 处理 {img_name} 时出错: {e}")

        print(f"✅ 处理完成！结果存放在: {self.output_dir}")


if __name__ == '__main__':
    system = GFPGANOnlySystem()
    system.run()