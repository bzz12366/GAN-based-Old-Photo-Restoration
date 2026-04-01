import cv2
import os
import shutil


def preprocess_and_rename(input_path, output_path, target_size=(512, 512)):
    # 1. 清理并创建输出目录（确保每次运行都是从001开始）
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    # 2. 获取所有图片文件
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    files = [f for f in os.listdir(input_path) if f.lower().endswith(valid_extensions)]

    # 对原文件名进行排序，保证编号的逻辑顺序
    files.sort()

    print(f"检测到 {len(files)} 张图片，开始处理并按序号重命名...")

    for i, filename in enumerate(files, 1):
        img_path = os.path.join(input_path, filename)
        img = cv2.imread(img_path)

        if img is None:
            continue

        # 3. 居中裁剪 (Center Crop) 逻辑
        h, w = img.shape[:2]
        side = min(h, w)
        top = (h - side) // 2
        left = (w - side) // 2
        crop_img = img[top:top + side, left:left + side]

        # 4. 统一缩放
        resized_img = cv2.resize(crop_img, target_size, interpolation=cv2.INTER_LANCZOS4)

        # 5. 【核心修改】使用三位数字编号命名
        # 例如：001.png, 002.png, 010.png
        new_name = f"{i:03d}.png"

        save_path = os.path.join(output_path, new_name)
        cv2.imwrite(save_path, resized_img)

        print(f"处理完成: {filename} ----> {new_name}")

    print(f"\n✅ 预处理成功！输出路径: {output_path}")


if __name__ == "__main__":
    raw_dir = r'E:\OldPhotoRestoration_GAN\data\inputs'
    out_dir = r'E:\OldPhotoRestoration_GAN\data\inputs_processed'

    preprocess_and_rename(raw_dir, out_dir, target_size=(512, 512))