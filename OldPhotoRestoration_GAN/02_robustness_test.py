import os
import cv2
import torch
import numpy as np
from gfpgan import GFPGANer
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact


class RobustnessExperiment:
    def __init__(self, base_dir=r'E:\OldPhotoRestoration_GAN'):
        self.base_dir = base_dir
        self.input_dir = os.path.join(base_dir, 'data', 'inputs')
        self.output_root = os.path.join(base_dir, 'data', 'outputs', 'Robustness')
        self.weight_dir = os.path.join(base_dir, 'weights')

        # 定义三个主要的输出文件夹
        self.dir_degraded = os.path.join(self.output_root, 'Degraded_Images')
        self.dir_restored = os.path.join(self.output_root, 'Restored_Images')
        self.dir_canvas = os.path.join(self.output_root, 'Comparison_Canvas')

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._init_models()

    def _init_models(self):
        # 1. 背景修复模型 (Real-ESRGAN)
        model_v3 = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        self.upsampler = RealESRGANer(
            scale=4,
            model_path=os.path.join(self.weight_dir, 'realesr-general-x4v3.pth'),
            model=model_v3,
            tile=400
        )

        # 2. 人脸修复模型 (GFPGAN 级联背景修复)
        self.restorer = GFPGANer(
            model_path=os.path.join(self.weight_dir, 'GFPGANv1.4.pth'),
            upscale=2,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=self.upsampler
        )

        # 3. 添加颜色模型 (Caffe)
        proto = os.path.join(self.weight_dir, 'colorization_deploy_v2.prototxt')
        model = os.path.join(self.weight_dir, 'colorization_release_v2.caffemodel')
        pts_path = os.path.join(self.weight_dir, 'pts_in_hull.npy')

        self.net_color = cv2.dnn.readNetFromCaffe(proto, model)
        pts = np.load(pts_path).transpose().reshape(2, 313, 1, 1)
        self.net_color.getLayer(self.net_color.getLayerId("class8_ab")).blobs = [pts.astype("float32")]
        self.net_color.getLayer(self.net_color.getLayerId("conv8_313_rh")).blobs = [
            np.full([1, 313], 2.606, dtype="float32")]

    def process_colorization(self, img):
        h, w = img.shape[:2]
        scaled = img.astype("float32") / 255.0
        lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
        L = cv2.resize(lab[:, :, 0], (224, 224)) - 50
        self.net_color.setInput(cv2.dnn.blobFromImage(L))
        ab = self.net_color.forward()[0, :, :, :].transpose((1, 2, 0))
        ab = cv2.resize(ab, (w, h))
        colorized = cv2.merge([lab[:, :, 0], ab])
        colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
        return (255 * np.clip(colorized, 0, 1)).astype("uint8")

    def add_stress(self, img, blur_k, noise_sigma):
        stressed = cv2.GaussianBlur(img, (blur_k, blur_k), 0)
        noise = np.random.normal(0, noise_sigma, stressed.shape).astype(np.float32)
        stressed = np.clip(stressed.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return stressed

    def run(self):
        levels = {
            "Light": {"blur": 3, "noise": 15},
            "Medium": {"blur": 7, "noise": 45},
            "Heavy": {"blur": 13, "noise": 85}
        }

        img_list = sorted([f for f in os.listdir(self.input_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

        for img_name in img_list:
            img_path = os.path.join(self.input_dir, img_name)
            original = cv2.imread(img_path)
            if original is None: continue
            h, w = original.shape[:2]

            for level_name, p in levels.items():
                print(f"🔄 处理中: {img_name} | 等级: {level_name}")

                # 1. 制造破坏并保存
                stressed = self.add_stress(original, p['blur'], p['noise'])
                deg_level_dir = os.path.join(self.dir_degraded, level_name)
                os.makedirs(deg_level_dir, exist_ok=True)
                cv2.imwrite(os.path.join(deg_level_dir, img_name), stressed)

                # 2. 全模型修复并保存
                color_img = self.process_colorization(stressed)
                _, _, restored = self.restorer.enhance(color_img, paste_back=True)
                res_level_dir = os.path.join(self.dir_restored, level_name)
                os.makedirs(res_level_dir, exist_ok=True)
                cv2.imwrite(os.path.join(res_level_dir, img_name), restored)

                # 3. 创建三合一对比图 [原始 | 破坏 | 修复]
                str_res = cv2.resize(stressed, (w, h))
                final_res = cv2.resize(restored, (w, h))
                canvas = np.hstack([original, str_res, final_res])

                # 存入对比图文件夹 (按等级分子目录)
                canvas_level_dir = os.path.join(self.dir_canvas, level_name)
                os.makedirs(canvas_level_dir, exist_ok=True)
                cv2.imwrite(os.path.join(canvas_level_dir, f"compare_{img_name}"), canvas)

        print(f"✅ 实验完成！结果已分类存入: {self.output_root}")


if __name__ == '__main__':
    exp = RobustnessExperiment()
    exp.run()