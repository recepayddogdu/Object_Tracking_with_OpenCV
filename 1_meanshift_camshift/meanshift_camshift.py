import cv2
import numpy as np

#kamera ac
cap = cv2.VideoCapture(0)

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