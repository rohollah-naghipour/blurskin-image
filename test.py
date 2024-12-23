import cv2
import mediapipe as mp
import numpy as np


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
def blur_skin(image):
    with mp_holistic.Holistic(static_image_mode=True) as holistic:
        # تبدیل تصویر به RGB برای Mediapipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)
        # اگر بدنی شناسایی نشد
        if not results.pose_landmarks:
            print("هیچ انسانی شناسایی نشد!")
            return image
        # تشخیص نواحی بدن
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for landmark in results.pose_landmarks.landmark:
            x = int(landmark.x * image.shape[1])
            y = int(landmark.y * image.shape[0])
            cv2.circle(mask, (x, y), 5, 255, -1)
        # ایجاد ماسک برای پوست
        skin_mask = cv2.inRange(image, (0, 30, 60), (20, 150, 255))
        # ترکیب ماسک Mediapipe با ماسک پوست
        combined_mask = cv2.bitwise_and(mask, skin_mask)
        # اعمال بلور روی مناطق پوست
        blurred_image = image.copy()
        blurred_image[combined_mask > 0] = cv2.GaussianBlur(
            image, (15, 15), 0)[combined_mask > 0]
        return blurred_image
    

with open('input.jpg', 'rb') as f:
    image = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_UNCHANGED)

output_image = blur_skin(image)

