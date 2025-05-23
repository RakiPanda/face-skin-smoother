import cv2
import numpy as np
import os
from pathlib import Path
import mediapipe as mp
import argparse

class FaceSkinSmoother:
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
        
        # 唇の輪郭ポイント
        self.lips_points = [
            61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318,
            402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 267, 272, 271, 272
        ]
        
        # 目の輪郭ポイント
        # 左目
        self.left_eye_points = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        # 右目
        self.right_eye_points = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # 眉毛の輪郭ポイント
        # 左眉毛
        self.left_eyebrow_points = [46, 53, 52, 51, 48, 115, 131, 134, 102, 49, 220, 305]
        # 右眉毛
        self.right_eyebrow_points = [285, 336, 296, 334, 293, 300, 276, 283, 282, 295, 285, 336]

    def guided_filter(self, I, p, r, eps):
        """Guided Filter implementation based on Kaiming He et al. paper"""
        mean_I = cv2.boxFilter(I, cv2.CV_64F, (r, r))
        mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
        mean_Ip = cv2.boxFilter(I * p, cv2.CV_64F, (r, r))
        cov_Ip = mean_Ip - mean_I * mean_p
        
        mean_II = cv2.boxFilter(I * I, cv2.CV_64F, (r, r))
        var_I = mean_II - mean_I * mean_I
        
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I
        
        mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
        mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))
        
        q = mean_a * I + mean_b
        return q
    
    def advanced_skin_enhancement(self, image, mask, level):
        """Guided Filter based skin enhancement"""
        if image is None or mask is None:
            return image
        
        # Convert to float for precision
        img_float = image.astype(np.float32) / 255.0
        
        # Fixed parameters for optimal quality
        r = 10  # Filter radius (fixed)
        eps = 0.02  # Regularization parameter (fixed)
        
        # Only blend ratio varies by level
        blend_ratios = {
            1: 0.12,  # Very subtle
            2: 0.24,  # Subtle
            3: 0.36,  # Moderate
            4: 0.48,  # Strong
            5: 0.60   # Very strong
        }
        
        blend = blend_ratios[level]
        
        # Use luminance as guidance image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        
        # Apply guided filter to each channel
        enhanced_channels = []
        for i in range(3):
            channel = img_float[:, :, i]
            filtered = self.guided_filter(gray, channel, r=r, eps=eps)
            enhanced_channels.append(filtered)
        
        enhanced_result = np.stack(enhanced_channels, axis=2)
        
        # Create 3D mask for blending
        mask_3d = np.stack([mask/255.0] * 3, axis=2)
        
        # Smooth mask edges for natural transition
        mask_smooth = cv2.GaussianBlur(mask_3d, (15, 15), 0)
        
        # Blend with original
        result = img_float * (1 - mask_smooth * blend) + enhanced_result * (mask_smooth * blend)
        
        # Convert back to uint8
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        
        return result

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
        # 顔の輪郭ポイント
        face_oval = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
        ]
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return None, None, None, None, None, None, None, None
        
        h, w = image.shape[:2]
        face_landmarks = results.multi_face_landmarks[0]
        
        # 顔の輪郭ポイントを取得
        face_points = []
        for idx in face_oval:
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
        
        # 唇マスクを作成
        lips_mask = self.create_lips_mask(image, face_landmarks)
        
        # 目マスクを作成
        eyes_mask = self.create_eyes_mask(image, face_landmarks)
        
        # 眉毛マスクを作成
        eyebrows_mask = self.create_eyebrows_mask(image, face_landmarks)
        
        # 肌色マスクから唇、目、眉毛を除外
        skin_mask_clean = cv2.bitwise_and(skin_mask, cv2.bitwise_not(lips_mask))
        skin_mask_clean = cv2.bitwise_and(skin_mask_clean, cv2.bitwise_not(eyes_mask))
        skin_mask_clean = cv2.bitwise_and(skin_mask_clean, cv2.bitwise_not(eyebrows_mask))
        
        # 肌色マスクから全ての不要部位を除外したものを最終マスクとする
        combined_mask = skin_mask_clean
        
        # デバッグ用に全てのランドマークを描画した画像を作成
        landmarks_image = image.copy()
        for i, landmark in enumerate(face_landmarks.landmark):
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(landmarks_image, (x, y), 1, (0, 255, 0), -1)
            # おでこ周辺のポイントを赤色で強調
            if i in [9, 10, 151, 337, 299, 333, 298, 301]:
                cv2.circle(landmarks_image, (x, y), 3, (0, 0, 255), -1)
            # 唇のポイントを青色で強調
            if i in self.lips_points:
                cv2.circle(landmarks_image, (x, y), 2, (255, 0, 0), -1)
            # 目のポイントを黄色で強調
            if i in self.left_eye_points or i in self.right_eye_points:
                cv2.circle(landmarks_image, (x, y), 2, (0, 255, 255), -1)
            # 眉毛のポイントを紫色で強調
            if i in self.left_eyebrow_points or i in self.right_eyebrow_points:
                cv2.circle(landmarks_image, (x, y), 2, (255, 0, 255), -1)
        
        return combined_mask, face_points, face_mask, skin_mask, landmarks_image, lips_mask, eyes_mask, eyebrows_mask

    def create_lips_mask(self, image, face_landmarks):
        """唇の領域マスクを作成"""
        h, w = image.shape[:2]
        lips_mask = np.zeros((h, w), dtype=np.uint8)
        
        # 唇の輪郭ポイントを取得
        lips_points = []
        for idx in self.lips_points:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            lips_points.append([x, y])
        
        # 唇の輪郭マスクを作成
        if len(lips_points) > 0:
            lips_points = np.array(lips_points, dtype=np.int32)
            cv2.fillPoly(lips_mask, [lips_points], 255)
            
            # 唇マスクを少し膨張させて確実に除外
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            lips_mask = cv2.dilate(lips_mask, kernel, iterations=2)
        
        return lips_mask

    def create_eyes_mask(self, image, face_landmarks):
        """目の領域マスクを作成"""
        h, w = image.shape[:2]
        eyes_mask = np.zeros((h, w), dtype=np.uint8)
        
        # 左目の輪郭ポイントを取得
        left_eye_points = []
        for idx in self.left_eye_points:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            left_eye_points.append([x, y])
        
        # 右目の輪郭ポイントを取得
        right_eye_points = []
        for idx in self.right_eye_points:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            right_eye_points.append([x, y])
        
        # 目の輪郭マスクを作成
        if len(left_eye_points) > 0:
            left_eye_points = np.array(left_eye_points, dtype=np.int32)
            cv2.fillPoly(eyes_mask, [left_eye_points], 255)
            
        if len(right_eye_points) > 0:
            right_eye_points = np.array(right_eye_points, dtype=np.int32)
            cv2.fillPoly(eyes_mask, [right_eye_points], 255)
            
        # 目マスクを少し膨張させて確実に除外
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        eyes_mask = cv2.dilate(eyes_mask, kernel, iterations=2)
        
        return eyes_mask

    def create_eyebrows_mask(self, image, face_landmarks):
        """眉毛の領域マスクを作成"""
        h, w = image.shape[:2]
        eyebrows_mask = np.zeros((h, w), dtype=np.uint8)
        
        # 左眉毛の輪郭ポイントを取得
        left_eyebrow_points = []
        for idx in self.left_eyebrow_points:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            left_eyebrow_points.append([x, y])
        
        # 右眉毛の輪郭ポイントを取得
        right_eyebrow_points = []
        for idx in self.right_eyebrow_points:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            right_eyebrow_points.append([x, y])
        
        # 眉毛の輪郭マスクを作成
        if len(left_eyebrow_points) > 0:
            left_eyebrow_points = np.array(left_eyebrow_points, dtype=np.int32)
            cv2.fillPoly(eyebrows_mask, [left_eyebrow_points], 255)
            
        if len(right_eyebrow_points) > 0:
            right_eyebrow_points = np.array(right_eyebrow_points, dtype=np.int32)
            cv2.fillPoly(eyebrows_mask, [right_eyebrow_points], 255)
            
        # 眉毛マスクを少し膨張させて確実に除外
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        eyebrows_mask = cv2.dilate(eyebrows_mask, kernel, iterations=3)
        
        return eyebrows_mask

    def create_landmarks_image(self, image, face_landmarks, base_name, output_dir):
        """ランドマーク可視化画像を作成"""
        landmarks_image = image.copy()
        h, w = image.shape[:2]
        
        # 一般的なランドマークを緑色で描画
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(landmarks_image, (x, y), 1, (0, 255, 0), -1)
        
        # おでこのランドマークを赤色で描画
        for idx in self.forehead_points:
            if idx < len(face_landmarks.landmark):
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(landmarks_image, (x, y), 2, (0, 0, 255), -1)
        
        # 唇のランドマークを青色で描画
        for idx in self.lips_points:
            if idx < len(face_landmarks.landmark):
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(landmarks_image, (x, y), 2, (255, 0, 0), -1)
        
        # 目のランドマークを黄色で描画
        for idx in self.left_eye_points + self.right_eye_points:
            if idx < len(face_landmarks.landmark):
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(landmarks_image, (x, y), 2, (0, 255, 255), -1)
        
        # 眉毛のランドマークを紫色で描画
        for idx in self.left_eyebrow_points + self.right_eyebrow_points:
            if idx < len(face_landmarks.landmark):
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(landmarks_image, (x, y), 2, (255, 0, 255), -1)
        
        return landmarks_image

    def visualize_skin_color_mask(self, image, skin_mask, base_name, output_dir):
        """肌色検出結果を可視化"""
        overlay = image.copy()
        # 肌色マスク領域に青色を適用
        overlay[skin_mask > 0] = overlay[skin_mask > 0] * 0.7 + np.array([255, 0, 0]) * 0.3
        
        # 結果を保存
        output_path = output_dir / f"{base_name}_skin_color_detected.jpg"
        cv2.imwrite(str(output_path), overlay)
        
        return overlay

    def process_image(self, input_path, output_dir):
        """画像を処理して6段階の美肌効果を適用"""
        image = cv2.imread(str(input_path))
        if image is None:
            print(f"画像の読み込みに失敗: {input_path}")
            return
        
        # 肌領域マスクを取得
        skin_mask, face_points, face_mask, skin_mask_debug, landmarks_image, lips_mask, eyes_mask, eyebrows_mask = self.get_face_mask(image)
        if skin_mask is None:
            print(f"顔が検出できませんでした: {input_path}")
            return
        
        base_name = input_path.stem
        
        # 元画像を保存
        cv2.imwrite(str(output_dir / f"{base_name}_original{input_path.suffix}"), image)
        
        # 肌検出領域を可視化した画像を保存
        vis_image = self.visualize_skin_color_mask(image, skin_mask, base_name, output_dir)
        
        # デバッグ用画像を保存
        cv2.imwrite(str(output_dir / f"{base_name}_landmarks{input_path.suffix}"), landmarks_image)
        cv2.imwrite(str(output_dir / f"{base_name}_face_mask{input_path.suffix}"), face_mask)
        cv2.imwrite(str(output_dir / f"{base_name}_skin_mask{input_path.suffix}"), skin_mask_debug)
        cv2.imwrite(str(output_dir / f"{base_name}_lips_mask{input_path.suffix}"), lips_mask)
        cv2.imwrite(str(output_dir / f"{base_name}_eyes_mask{input_path.suffix}"), eyes_mask)
        cv2.imwrite(str(output_dir / f"{base_name}_eyebrows_mask{input_path.suffix}"), eyebrows_mask)
        cv2.imwrite(str(output_dir / f"{base_name}_final_mask{input_path.suffix}"), skin_mask)
        
        print(f"  デバッグ画像保存完了")
        
        # 6段階の美肌効果を適用
        for level in range(1, 6):
            smoothed = self.advanced_skin_enhancement(image, skin_mask, level)
            output_path = output_dir / f"{base_name}_smooth_level{level}{input_path.suffix}"
            cv2.imwrite(str(output_path), smoothed)
            print(f"  レベル{level}完了")

def main():
    input_dir = Path("input_images")
    output_dir = Path("output_images")
    
    # 入力ディレクトリが存在しない場合は作成
    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    
    # 肌検出器を初期化
    detector = FaceSkinSmoother()
    
    # 入力画像の処理
    image_extensions = ('.jpg', '.jpeg', '.png')
    for image_path in input_dir.glob('*'):
        if image_path.suffix.lower() in image_extensions:
            print(f"処理中: {image_path}")
            detector.process_image(image_path, output_dir)

if __name__ == "__main__":
    main() 