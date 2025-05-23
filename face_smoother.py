import cv2
import numpy as np
import os
from pathlib import Path
import mediapipe as mp

class FaceSkinDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        # 顔の主要な輪郭ポイント（目、鼻、口周りを除く）
        self.face_oval = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
        ]

    def detect_skin_hsv(self, image):
        """HSV色空間を使用した肌色検出"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 肌色の範囲を定義（複数の範囲を使用）
        lower_skin1 = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin1 = np.array([20, 255, 255], dtype=np.uint8)
        
        lower_skin2 = np.array([0, 0, 0], dtype=np.uint8)
        upper_skin2 = np.array([180, 255, 230], dtype=np.uint8)
        
        # 肌色マスクを作成
        mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
        mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
        
        # YCrCb色空間も併用
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        lower_ycrcb = np.array([0, 135, 85], dtype=np.uint8)
        upper_ycrcb = np.array([255, 180, 135], dtype=np.uint8)
        mask3 = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
        
        # マスクを結合
        mask = cv2.bitwise_or(mask1, mask2)
        mask = cv2.bitwise_and(mask, mask3)
        
        # ノイズ除去
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask

    def get_face_mask(self, image):
        """MediaPipeを使用して顔の領域マスクを作成"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return None, None, None, None, None
        
        h, w = image.shape[:2]
        face_landmarks = results.multi_face_landmarks[0]
        
        # 顔の輪郭ポイントを取得
        face_points = []
        for idx in self.face_oval:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            face_points.append([x, y])
        
        # 顔の輪郭マスクを作成
        face_mask = np.zeros((h, w), dtype=np.uint8)
        face_points = np.array(face_points, dtype=np.int32)
        cv2.fillPoly(face_mask, [face_points], 255)
        
        # 肌色マスクを取得
        skin_mask = self.detect_skin_hsv(image)
        
        # 顔の輪郭と肌色の両方の条件を満たす領域
        combined_mask = cv2.bitwise_and(face_mask, skin_mask)
        
        # さらにノイズ除去
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # デバッグ用に全てのランドマークを描画した画像を作成
        landmarks_image = image.copy()
        for i, landmark in enumerate(face_landmarks.landmark):
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(landmarks_image, (x, y), 1, (0, 255, 0), -1)
            # おでこ周辺のポイントを赤色で強調
            if i in [9, 10, 151, 337, 299, 333, 298, 301]:
                cv2.circle(landmarks_image, (x, y), 3, (0, 0, 255), -1)
        
        return combined_mask, face_points, face_mask, skin_mask, landmarks_image

def create_output_dirs():
    """出力ディレクトリを作成"""
    output_dir = Path("output_images")
    output_dir.mkdir(exist_ok=True)

def visualize_skin_region(image, mask):
    """肌検出領域を可視化"""
    overlay = image.copy()
    # マスク領域に緑色を適用
    overlay[mask > 0] = overlay[mask > 0] * 0.7 + np.array([0, 255, 0]) * 0.3
    return overlay

def visualize_skin_color_mask(image, skin_mask):
    """肌色マスクのみを可視化"""
    overlay = image.copy()
    # 肌色マスク領域に青色を適用
    overlay[skin_mask > 0] = overlay[skin_mask > 0] * 0.7 + np.array([255, 0, 0]) * 0.3
    return overlay

def apply_skin_smoothing(image, mask, level):
    """マスクされた肌領域にバイラテラルフィルターを適用"""
    if mask is None or np.sum(mask) == 0:
        return image
    
    # フィルターの強度をレベルに応じて調整
    d = 9  # フィルターの直径
    sigma_color = 20 + (level * 10)  # 色空間の標準偏差
    sigma_space = 20 + (level * 10)  # 座標空間の標準偏差
    
    # 画像全体にフィルターを適用
    smoothed = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    # マスクを使って肌領域のみ置き換え
    result = image.copy()
    result[mask > 0] = smoothed[mask > 0]
    
    return result

def process_image(input_path, output_dir, detector):
    """画像を処理して6段階の美肌効果を適用"""
    image = cv2.imread(str(input_path))
    if image is None:
        print(f"画像の読み込みに失敗: {input_path}")
        return
    
    # 肌領域マスクを取得
    skin_mask, face_points, face_mask, skin_mask_debug, landmarks_image = detector.get_face_mask(image)
    if skin_mask is None:
        print(f"顔が検出できませんでした: {input_path}")
        return
    
    base_name = input_path.stem
    
    # 元画像を保存
    cv2.imwrite(str(output_dir / f"{base_name}_original{input_path.suffix}"), image)
    
    # 肌検出領域を可視化した画像を保存
    vis_image = visualize_skin_region(image, skin_mask)
    cv2.imwrite(str(output_dir / f"{base_name}_detected{input_path.suffix}"), vis_image)
    
    # 肌色マスクのみを可視化した画像を保存
    skin_color_vis = visualize_skin_color_mask(image, skin_mask_debug)
    cv2.imwrite(str(output_dir / f"{base_name}_skin_color_detected{input_path.suffix}"), skin_color_vis)
    
    # デバッグ用画像を保存
    cv2.imwrite(str(output_dir / f"{base_name}_landmarks{input_path.suffix}"), landmarks_image)
    cv2.imwrite(str(output_dir / f"{base_name}_face_mask{input_path.suffix}"), face_mask)
    cv2.imwrite(str(output_dir / f"{base_name}_skin_mask{input_path.suffix}"), skin_mask_debug)
    cv2.imwrite(str(output_dir / f"{base_name}_final_mask{input_path.suffix}"), skin_mask)
    
    print(f"  デバッグ画像保存完了")
    
    # 6段階の美肌効果を適用
    for level in range(1, 6):
        smoothed = apply_skin_smoothing(image, skin_mask, level)
        output_path = output_dir / f"{base_name}_smooth_level{level}{input_path.suffix}"
        cv2.imwrite(str(output_path), smoothed)
        print(f"  レベル{level}完了")

def main():
    input_dir = Path("input_images")
    output_dir = Path("output_images")
    
    # 入力ディレクトリが存在しない場合は作成
    input_dir.mkdir(exist_ok=True)
    create_output_dirs()
    
    # 肌検出器を初期化
    detector = FaceSkinDetector()
    
    # 入力画像の処理
    image_extensions = ('.jpg', '.jpeg', '.png')
    for image_path in input_dir.glob('*'):
        if image_path.suffix.lower() in image_extensions:
            print(f"処理中: {image_path}")
            process_image(image_path, output_dir, detector)

if __name__ == "__main__":
    main() 