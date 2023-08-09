from cvlib.object_detection import YOLO
import cv2

cap=cv2.VideoCapture(0)
weights="yolov4-tiny-custom_best.weights"
config="yolov4-tiny-custom.cfg"
labels="obj.names"

#while True:
#ret,img=cap.read()

img = cv2.imread("images.jpg")
img=cv2.resize(img,(680,460))

yolo = YOLO(weights, config, labels)
bbox, label, conf = yolo.detect_objects(img)
img1=yolo.draw_bbox(img, bbox, label, conf)

cv2.imshow("img1",img)

cv2.waitKey(0)
#if cv2.waitKey(1)&0xFF==27:
    #break