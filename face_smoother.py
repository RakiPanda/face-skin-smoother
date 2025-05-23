import cv2
import numpy as np
import os
from pathlib import Path
import urllib.request

def download_cascade():
    """顔検出用のカスケードファイルをダウンロード"""
    cascade_path = Path("haarcascade_frontalface_default.xml")
    if not cascade_path.exists():
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        urllib.request.urlretrieve(url, cascade_path)
    return str(cascade_path)

def create_output_dirs():
    """出力ディレクトリを作成"""
    output_dir = Path("output_images")
    output_dir.mkdir(exist_ok=True)

def detect_face(image):
    """顔検出を行う"""
    cascade_path = download_cascade()
    face_cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        return None
    
    # 最大の顔領域を返す
    max_face = max(faces, key=lambda x: x[2] * x[3])
    return max_face

def visualize_face_region(image, face_rect):
    """顔検出領域を可視化"""
    x, y, w, h = face_rect
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 0), -1)  # 緑色の矩形
    alpha = 0.3  # 透明度
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    return image

def apply_smoothing(image, face_rect, level):
    """バイラテラルフィルターで美肌処理を適用"""
    x, y, w, h = face_rect
    face_roi = image[y:y+h, x:x+w]
    
    # フィルターの強度をレベルに応じて調整
    d = 5  # フィルターの直径
    sigma_color = 10 + (level * 5)  # 色空間の標準偏差
    sigma_space = 10 + (level * 5)  # 座標空間の標準偏差
    
    smoothed_face = cv2.bilateralFilter(face_roi, d, sigma_color, sigma_space)
    
    # 元の画像に処理した顔を戻す
    result = image.copy()
    result[y:y+h, x:x+w] = smoothed_face
    
    return result

def process_image(input_path, output_dir):
    """画像を処理して6段階の美肌効果を適用"""
    image = cv2.imread(str(input_path))
    if image is None:
        print(f"画像の読み込みに失敗: {input_path}")
        return
    
    face_rect = detect_face(image)
    if face_rect is None:
        print(f"顔が検出できませんでした: {input_path}")
        return
    
    # 元画像を保存
    base_name = input_path.stem
    cv2.imwrite(str(output_dir / f"{base_name}_original{input_path.suffix}"), image)
    
    # 顔検出領域を可視化した画像を保存
    vis_image = image.copy()
    vis_image = visualize_face_region(vis_image, face_rect)
    cv2.imwrite(str(output_dir / f"{base_name}_detected{input_path.suffix}"), vis_image)
    
    # 6段階の美肌効果を適用
    for level in range(1, 6):
        smoothed = apply_smoothing(image, face_rect, level)
        output_path = output_dir / f"{base_name}_smooth_level{level}{input_path.suffix}"
        cv2.imwrite(str(output_path), smoothed)

def main():
    input_dir = Path("input_images")
    output_dir = Path("output_images")
    
    # 入力ディレクトリが存在しない場合は作成
    input_dir.mkdir(exist_ok=True)
    create_output_dirs()
    
    # 入力画像の処理
    image_extensions = ('.jpg', '.jpeg', '.png')
    for image_path in input_dir.glob('*'):
        if image_path.suffix.lower() in image_extensions:
            print(f"処理中: {image_path}")
            process_image(image_path, output_dir)

if __name__ == "__main__":
    main() 