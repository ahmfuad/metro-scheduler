from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load YOLOv8 pretrained model (or use YOLOv5, both work)
model = YOLO("yolov8n.pt")  # 'n' = nano version (fastest), you can try yolov8s.pt or yolov8m.pt too

# Load image
image_path = "image_path"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Run detection
results = model(image_rgb)

# Filter only 'person' class (class id 0 in COCO)
person_detections = [det for det in results[0].boxes.data if int(det[-1]) == 0]

# Draw bounding boxes and count
for det in person_detections:
    x1, y1, x2, y2, conf, cls = det
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.putText(image, f'Person {conf:.2f}', (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

count = len(person_detections)
cv2.putText(image, f'Total People: {count}', (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

# Show result
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title(f"People Detected: {count}")
plt.axis("off")
plt.show()

cv2.imshow(f"People Detected: {count}", image)
cv2.waitKey(0)
cv2.destroyAllWindows()