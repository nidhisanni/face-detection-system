import cv2
from detector import FaceDetector

detector = FaceDetector(
    model_path="models/res10_300x300_ssd_iter_140000.caffemodel",
    config_path="models/deploy.prototxt.txt"
)

img = cv2.imread("test.jpg.png")
faces = detector.detect_faces(img)

print(f"Faces detected: {len(faces)}")

for face in faces:
    print(face["confidence"])
