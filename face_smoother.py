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
            402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 272, 271
        ]
        
        # 目の輪郭ポイント
        self.left_eye_points = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.right_eye_points = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # 眉毛の輪郭ポイント
        self.left_eyebrow_points = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
        self.right_eyebrow_points = [296, 334, 293, 300, 276, 283, 282, 295, 285, 336]

    def guided_filter(self, I, p, r, eps):
        """Guided Filter implementation"""
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
    
    def advanced_skin_enhancement(self, image, mask, blend: float):
        """Guided filter based skin enhancement; blend is a float in [0.0,1.0]"""
        if image is None or mask is None:
            return image
        
        img_float = image.astype(np.float32) / 255.0
        r, eps = 10, 0.02
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        enhanced_channels = []
        for i in range(3):
            channel = img_float[:, :, i]
            filtered = self.guided_filter(gray, channel, r=r, eps=eps)
            enhanced_channels.append(filtered)
        enhanced_result = np.stack(enhanced_channels, axis=2)
        
        mask_3d = np.stack([mask/255.0]*3, axis=2)
        mask_smooth = cv2.GaussianBlur(mask_3d, (15,15), 0)
        
        result = img_float * (1 - mask_smooth * blend) + enhanced_result * (mask_smooth * blend)
        return (np.clip(result * 255, 0, 255)).astype(np.uint8)

    def detect_skin_hsv(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_skin1 = np.array([0,20,70], dtype=np.uint8)
        upper_skin1 = np.array([20,255,255], dtype=np.uint8)
        lower_skin2 = np.array([0,0,0], dtype=np.uint8)
        upper_skin2 = np.array([180,255,230], dtype=np.uint8)
        mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
        mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        lower_ycrcb = np.array([0,135,85], dtype=np.uint8)
        upper_ycrcb = np.array([255,180,135], dtype=np.uint8)
        mask3 = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
        mask = cv2.bitwise_or(mask1, mask2)
        mask = cv2.bitwise_and(mask, mask3)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def get_face_mask(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return (None,)*8
        h,w = image.shape[:2]
        lm = results.multi_face_landmarks[0]
        pts = [[int(lm.landmark[i].x*w), int(lm.landmark[i].y*h)] for i in self.face_oval]
        face_mask = np.zeros((h,w), dtype=np.uint8)
        cv2.fillPoly(face_mask, [np.array(pts, np.int32)], 255)
        skin = self.detect_skin_hsv(image)
        lips = self.create_lips_mask(image, lm)
        eyes = self.create_eyes_mask(image, lm)
        brows = self.create_eyebrows_mask(image, lm)
        skin_clean = cv2.bitwise_and(skin, cv2.bitwise_not(lips))
        skin_clean = cv2.bitwise_and(skin_clean, cv2.bitwise_not(eyes))
        skin_clean = cv2.bitwise_and(skin_clean, cv2.bitwise_not(brows))
        # debug landmarks img
        debug = image.copy()
        for i_land in lm.landmark:
            x,y = int(i_land.x*w), int(i_land.y*h)
            cv2.circle(debug,(x,y),1,(0,255,0),-1)
        return skin_clean, np.array(pts), face_mask, skin, debug, lips, eyes, brows

    def create_lips_mask(self, image, lm):
        h,w = image.shape[:2]
        mask = np.zeros((h,w), dtype=np.uint8)
        pts = [[int(lm.landmark[i].x*w), int(lm.landmark[i].y*h)] for i in self.lips_points]
        if pts:
            cv2.fillPoly(mask, [np.array(pts,np.int32)], 255)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
            mask = cv2.dilate(mask,k,iterations=2)
        return mask

    def create_eyes_mask(self, image, lm):
        h,w = image.shape[:2]
        mask = np.zeros((h,w), dtype=np.uint8)
        for group in (self.left_eye_points, self.right_eye_points):
            pts = [[int(lm.landmark[i].x*w), int(lm.landmark[i].y*h)] for i in group]
            if pts:
                cv2.fillPoly(mask, [np.array(pts,np.int32)], 255)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        return cv2.dilate(mask,k,iterations=2)

    def create_eyebrows_mask(self, image, lm):
        h,w = image.shape[:2]
        mask = np.zeros((h,w), dtype=np.uint8)
        for grp in (self.left_eyebrow_points, self.right_eyebrow_points):
            pts = [[int(lm.landmark[i].x*w), int(lm.landmark[i].y*h)] for i in grp]
            if pts:
                cv2.fillPoly(mask,[np.array(pts,np.int32)],255)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        return cv2.dilate(mask,k,iterations=3)

    def visualize_skin_color_mask(self, image, skin_mask, base, out_dir):
        overlay = image.copy()
        overlay[skin_mask>0] = overlay[skin_mask>0]*0.7 + np.array([255,0,0])*0.3
        p = out_dir / f"{base}_skin_detected.jpg"
        cv2.imwrite(str(p), overlay)
        return overlay

    def process_image(self, input_path, output_dir):
        image = cv2.imread(str(input_path))
        if image is None:
            print(f"画像の読み込みに失敗: {input_path}")
            return
        skin_mask, pts, face_mask, skin_dbg, debug_img, lips_m, eyes_m, brows_m = self.get_face_mask(image)
        if skin_mask is None:
            print(f"顔が検出できませんでした: {input_path}")
            return
        base = input_path.stem
        # 保存
        cv2.imwrite(str(output_dir/f"{base}_original{input_path.suffix}"), image)
        cv2.imwrite(str(output_dir/f"{base}_landmarks{input_path.suffix}"), debug_img)
        cv2.imwrite(str(output_dir/f"{base}_face_mask{input_path.suffix}"), face_mask)
        cv2.imwrite(str(output_dir/f"{base}_skin_mask{input_path.suffix}"), skin_dbg)
        cv2.imwrite(str(output_dir/f"{base}_lips_mask{input_path.suffix}"), lips_m)
        cv2.imwrite(str(output_dir/f"{base}_eyes_mask{input_path.suffix}"), eyes_m)
        cv2.imwrite(str(output_dir/f"{base}_eyebrows_mask{input_path.suffix}"), brows_m)
        cv2.imwrite(str(output_dir/f"{base}_final_mask{input_path.suffix}"), skin_mask)
        print("  デバッグ画像保存完了")
        # 顔だけPNG
        face_rgba = cv2.cvtColor(image,cv2.COLOR_BGR2BGRA)
        face_rgba[:,:,3] = face_mask
        cv2.imwrite(str(output_dir/f"{base}_face_only.png"), face_rgba)
        # 0..100 のインデックスで blend
        for i, blend in enumerate(np.linspace(0.0,1.0,101)):
            sm = self.advanced_skin_enhancement(image, skin_mask, blend)
            cv2.imwrite(str(output_dir/f"blend_ratio_{i:03d}.png"), sm)
            sm_rgba = cv2.cvtColor(sm,cv2.COLOR_BGR2BGRA)
            sm_rgba[:,:,3] = face_mask
            cv2.imwrite(str(output_dir/f"blend_ratio_{i:03d}_face.png"), sm_rgba)
            if i % 10 == 0 or i == 100:
                print(f"  blend_ratio_{i:03d}.png 完了")

def main():
    input_dir = Path("input_images")
    output_dir = Path("output_images")
    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    detector = FaceSkinSmoother()
    for img_path in input_dir.glob('*'):
        if img_path.suffix.lower() in ('.jpg','.jpeg','.png'):
            print(f"処理中: {img_path.name}")
            detector.process_image(img_path, output_dir)

if __name__ == "__main__":
    main()
