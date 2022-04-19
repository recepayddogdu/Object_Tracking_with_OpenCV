import cv2
import numpy as np

#kamera ac
cap = cv2.VideoCapture(4)

# bir tane frame oku
ret, frame = cap.read()

if not ret:
    print("Frame okunamadi.")


# detection
face_cascade = cv2.CascadeClassifier("1_meanshift_camshift/haarcascade_frontalface_default.xml")
face_rects = face_cascade.detectMultiScale(frame)

(face_x, face_y, w, h) = tuple(face_rects[0])

# meanshift algoritma girdisi
track_window = (face_x, face_y, w, h)

# ROI (Region of Interest)
roi = frame[face_y : face_y + h, face_x : face_x + w] # roi = face

hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
# takip icin histogram gerekli
roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Takip icin gerekli durdurma kriterleri
# count = hesaplanacak max oge sayisi
# eps = yinelemeli algoritmanin durdugu parametrelerde istenen
term_crit = (cv2.TermCriteria_EPS | cv2.TermCriteria_COUNT, 5, 1)

while True:
    
    ret, frame = cap.read()
    
    if ret: 
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Back projection
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        # histogrami bir goruntude bulmak icin kullaniyoruz
        # boylece takip gerceklesiyor
        # piksel karsilastirma
        
        # Meanshift algoritmasi
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        
        x, y, w, h = track_window
        
        img2 = cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 5)
        
        cv2.imshow("Takip", img2)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
            