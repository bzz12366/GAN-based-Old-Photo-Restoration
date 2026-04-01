import os
import cv2
import torch
import numpy as np
from gfpgan import GFPGANer
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

class FinalRestorationSystem:
    def __init__(self, base_dir=r'E:\OldPhotoRestoration_GAN'):
        self.base_dir = base_dir
        self.input_dir = os.path.join(base_dir, 'data', 'inputs')
        # 输出路径：data/outputs/ablation/Full
        self.output_dir = os.path.join(base_dir, 'data', 'outputs', 'ablation', 'Full')
        os.makedirs(self.output_dir, exist_ok=True)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 最终修复系统启动 | 设备: {self.device}")
        self._init_models()

    def _init_models(self):
        print("📦 正在加载修复模型组...")
        weight_dir = os.path.join(self.base_dir, 'weights')

        # 1. 背景增强 (Real-ESRGAN)
        model_v3 = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        self.upsampler = RealESRGANer(
            scale=4,
            model_path=os.path.join(weight_dir, 'realesr-general-x4v3.pth'),
            model=model_v3,
            tile=400,
            half=True if self.device == 'cuda' else False
        )

        # 2. 全功能修复器 (GFPGAN 级联 Real-ESRGAN)
        self.face_with_bg = GFPGANer(
            model_path=os.path.join(weight_dir, 'GFPGANv1.4.pth'),
            upscale=2, arch='clean', channel_multiplier=2, bg_upsampler=self.upsampler
        )

        # 3. 上色引擎 (Caffe)
        proto = os.path.join(weight_dir, 'colorization_deploy_v2.prototxt')
        model = os.path.join(weight_dir, 'colorization_release_v2.caffemodel')
        pts_path = os.path.join(weight_dir, 'pts_in_hull.npy')

        self.net_color = cv2.dnn.readNetFromCaffe(proto, model)
        pts = np.load(pts_path).transpose().reshape(2, 313, 1, 1)
        self.net_color.getLayer(self.net_color.getLayerId("class8_ab")).blobs = [pts.astype("float32")]
        self.net_color.getLayer(self.net_color.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    def process_colorization(self, img):
        """核心上色逻辑"""
        h, w = img.shape[:2]
        lab = cv2.cvtColor(img.astype("float32") / 255.0, cv2.COLOR_BGR2LAB)
        L_full = lab[:, :, 0]
        L_resized = cv2.resize(L_full, (224, 224)) - 50
        self.net_color.setInput(cv2.dnn.blobFromImage(L_resized))
        ab = self.net_color.forward()[0, :, :, :].transpose((1, 2, 0))
        ab = cv2.resize(ab, (w, h))
        colorized = cv2.merge([L_full, ab])
        colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
        return (255 * np.clip(colorized, 0, 1)).astype("uint8")

    def run(self):
        valid_extensions = ('.jpg', '.png', '.jpeg')
        img_list = sorted([f for f in os.listdir(self.input_dir) if f.lower().endswith(valid_extensions)])

        if not img_list:
            print("❌ 未发现可处理的图片。")
            return

        for img_name in img_list:
            print(f"📸 正在修复: {img_name}")
            img_path = os.path.join(self.input_dir, img_name)
            original = cv2.imread(img_path)
            if original is None: continue

            try:
                # 步骤 1: 自动上色
                color_img = self.process_colorization(original)

                # 步骤 2: 人脸修复 + 背景增强 (同步进行)
                # 使用级联了 upsampler 的模型进行推理
                _, _, restored_img = self.face_with_bg.enhance(
                    color_img,
                    has_aligned=False,
                    only_center_face=False,
                    paste_back=True
                )

                # 步骤 3: 只保存最终修复后的单张图片
                # 不修改文件名，直接存入指定的 Full 文件夹
                save_path = os.path.join(self.output_dir, img_name)
                cv2.imwrite(save_path, restored_img)

            except Exception as e:
                print(f"❌ 处理 {img_name} 出错: {e}")

        print(f"✅ 处理完成！修复后的文件存放在: {self.output_dir}")

if __name__ == '__main__':
    FinalRestorationSystem().run()