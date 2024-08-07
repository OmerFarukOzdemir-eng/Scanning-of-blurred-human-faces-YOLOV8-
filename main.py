import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for result in results:
        if hasattr(result, 'boxes'):
            boxes = result.boxes 

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())

                if cls == 0:
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                    roi_color = frame[y1:y2, x1:x2]
                    roi_color = cv2.GaussianBlur(roi_color, (25, 25), 0)
                    frame[y1:y2, x1:x2] = roi_color

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
