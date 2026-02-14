import cv2
import pytesseract
from ultralytics import YOLO
import numpy as np
import os

pytesseract.pytesseract.tesseract_cmd = r"D:\Softwares\Tesseract-OCR\tesseract.exe"

model_path = "best.pt"
if not os.path.exists(model_path):
    print("Error: YOLO model not found. Train it first!")
    exit()

model = YOLO(model_path)

cap = cv2.VideoCapture(0)  
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    results = model(frame)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            margin_x = int((x2 - x1) * 0.1)  
            margin_y = int((y2 - y1) * 0.2)  
            x1m = max(x1 - margin_x, 0)
            y1m = max(y1 - margin_y, 0)
            x2m = min(x2 + margin_x, frame.shape[1])
            y2m = min(y2 + margin_y, frame.shape[0])

            plate_img = frame[y1m:y2m, x1m:x2m]

            h, w = plate_img.shape[:2]
            new_w = 600
            new_h = int(h * (new_w / w))
            plate_img = cv2.resize(plate_img, (new_w, new_h))

            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

            kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
            gray = cv2.filter2D(gray, -1, kernel)

            gray = cv2.bilateralFilter(gray, 9, 75, 75)

            text = pytesseract.image_to_string(
                gray,
                config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            ).strip()

            if text:
                cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                print("Detected License Plate Number:", text)
            else:
                cv2.putText(frame, "OCR Failed", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("License Plate Detection - Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
