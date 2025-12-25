import sys
import cv2
import mediapipe as mp

model_path = "models/mediapipe/pose_landmarker_heavy.task"

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
)

if len(sys.argv) < 2:
    raise ValueError("Usage: python image_pose.py image_path")

image_path = sys.argv[1]

image = cv2.imread(image_path)

roi_area = cv2.selectROI(
    "Pick ROI (ENTER confirm, ESC cancel)", image, fromCenter=False, showCrosshair=True
)
cv2.destroyAllWindows()

x, y, w, h = roi_area

roi_image = image[y : y + h, x : x + w]

mp_image = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=mp_image)

landmarker = PoseLandmarker.create_from_options(options)

pose_landmarker_result = landmarker.detect(mp_image)

for landmark_detection in pose_landmarker_result.pose_landmarks:
    for landmark in landmark_detection:
        if landmark.visibility > 0.0:
            landmark_x = int(x + landmark.x * w)
            landmark_y = int(y + landmark.y * h)

            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 255), -1)

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
