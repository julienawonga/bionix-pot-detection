from cvlib.object_detection import YOLO
import cv2
import time

cap = cv2.VideoCapture(0)
weights = "yolov4-tiny-custom_best.weights"
config = "yolov4-tiny-custom.cfg"
labels = "obj.names"
count = 0

# Initialize variables for FPS calculation
start_time = time.time()
frame_count = 0

while True:
    ret, img = cap.read()

    img = cv2.resize(img, (680, 460))

    yolo = YOLO(weights, config, labels)
    bbox, label, conf = yolo.detect_objects(img)
    img1 = yolo.draw_bbox(img, bbox, label, conf)
    
    # Calculate FPS
    frame_count += 1
    if frame_count >= 10:
        end_time = time.time()
        elapsed_time = end_time - start_time
        fps = frame_count / elapsed_time
        print(f"FPS: {fps:.2f}")
        frame_count = 0
        start_time = end_time
    
    cv2.imshow("img1", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
