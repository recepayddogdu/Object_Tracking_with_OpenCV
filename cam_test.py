import cv2

cap = cv2.VideoCapture(4)

while True:
    ret, frame = cap.read()
    if ret:
       
        
        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            exit()