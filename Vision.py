import numpy as np
import cv2 as cv
from ultralytics import YOLO



capture = cv.VideoCapture(0)#uses webcam
capture.set(cv.CAP_PROP_FRAME_WIDTH, 640)#resolution set
capture.set(cv.CAP_PROP_FRAME_HEIGHT, 480)#resolution set
capture.set(cv.CAP_PROP_EXPOSURE,-7)#exposure set


model = YOLO("yolov8m.pt")#example model will be swapped with custom model


if not capture.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = capture.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Our operations on the frame come here
    flipped = cv.flip(frame, 1)

    results = model(flipped, device="mps") #Change for GPU this is for an M1 Mac 
    
    #bounding box stuff
    result = results[0]
    bboxes = np.array(result.boxes.xyxy, dtype="int")
    classes = np.array(result.boxes.cls, dtype="int")


    for cls, bbox in zip(classes, bboxes):
        (x, y, x2, y2) = bbox
        cv.rectangle(flipped, (x, y), (x2, y2), (0, 0, 225), 2)
        if(cls == 0):
            cv.putText(flipped, str("Person"), (x, y - 5), cv.FONT_HERSHEY_PLAIN, 2, (0, 255,0),2)
        cv.putText(flipped, str(cls), (x, y - 5), cv.FONT_HERSHEY_PLAIN, 2, (0, 255,0),2)


    # Display the resulting frame

    cv.imshow('frame', flipped)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture

capture.release()
cv.destroyAllWindows()
