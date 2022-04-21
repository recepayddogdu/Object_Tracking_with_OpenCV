import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

OPENCV_OBJECT_TRACKERS = {"csrt"      : cv2.legacy.TrackerCSRT_create(),
		                  "kcf"       : cv2.legacy.TrackerKCF_create(),
		                  "boosting"  : cv2.legacy.TrackerBoosting_create(),
		                  "mil"       : cv2.legacy.TrackerMIL_create(),
		                  "tld"       : cv2.legacy.TrackerTLD_create(),
		                  "medianflow": cv2.legacy.TrackerMedianFlow_create(),
		                  "mosse"     : cv2.legacy.TrackerMOSSE_create()}

tracker_name = "mosse"
print("Tracker:", tracker_name)

tracker = OPENCV_OBJECT_TRACKERS[tracker_name]

# Ground truth'u yukle;

gt = pd.read_csv("gt_new.txt")

# Videoyu ice aktar;

video_path = "MOT17_13_SDP.mp4"
cap = cv2.VideoCapture(video_path)

# Genel Parametreler

initBB = None # Secilen nesnenin bounding box bilgisi depolanacak
fps = 25
frame_number = []
f = 0
success_frame_track = 0
track_list = []
start_time = time.time()

while True:
    ret, frame = cap.read()
    
    # okunan videoyu yavaslatmak icin;
    # time.sleep(1/fps)
    
    if ret:
        frame = cv2.resize(frame, (960, 540))
        (H, W) = frame.shape[:2]
        
        # ground truth
        car_gt = gt[gt.frame_no == f]
        
        if len(car_gt) != 0:
            
            # 0. frame'de bir nesne sececegiz, secilen nesneyi boosting takip edecek.
            # Secilen nesnenin konum bilgileri;
            x = car_gt.x.values[0]
            y = car_gt.y.values[0]
            h = car_gt.h.values[0]
            w = car_gt.w.values[0]
            center_x = car_gt.center_x.values[0]
            center_y = car_gt.center_y.values[0]
            
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.circle(frame, (center_x, center_y), 2, (0,0,255), -1)
            
        # Gorsellestirme
        key = cv2.waitKey(1) & 0xFF
        
        # Tracking sonucunun uretilmesi;
        if initBB is not None:
            
            (success, box) = tracker.update(frame)
            
            # Basari metrigini belirlemek icin gt'nin frame icinde olmasi gerekiyor.
            # Bunu kontrol edebilmek icin if kosulu;
            if f <= np.max(gt.frame_no):
            
                (x, y, w, h) = [int(i) for i in box]
                
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)
                
                success_frame_track = success_frame_track + 1
                
                track_center_x = int(x+w/2)
                track_center_y = int(y+h/2)
                
                # ground truth ile tracking sonucunu karsilastirabilmek icin;
                track_list.append([f, track_center_x, track_center_y])
        
            # Tracking sonuclarini gorsellestirme;
            info = [("Tracker:", tracker_name),
                    ("Success:", "Yes" if success else "No")]
            
            for (i, (o,p)) in enumerate(info):
                text = "{}: {}".format(o, p)
                cv2.putText(frame, text, (10, H-(i*20)-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                
        cv2.putText(frame, "Frame num: "+str(f), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        cv2.imshow("BoostingAlgorithm", frame)
        
        if key == ord("t"):
            # Takip etmek istenilen bolgenin secilmesi;
            # t tusuna basildiginda secim ekrani acilir, secimden sonra space tusuna basinca program devam eder.
            initBB = cv2.selectROI("SelectROI", frame, fromCenter = False)
            
            tracker.init(frame, initBB)
            
        elif key == ord("q"): break # quit
        
        # frame parametreleri
        frame_number.append(f)
        f = f + 1
        
    else:
        print("Video dosyasi okunamiyor ya da video bitti...")
        break
    
cap.release()
cv2.destroyAllWindows()

# Model degerlendirmesi
stop_time = time.time()
time_diff = stop_time - start_time

track_df = pd.DataFrame(track_list, columns=["frame_no", "center_x", "center_y"])

if len(track_df) != 0:
    print("Tracking Algorithm:", tracker)
    print("Time:", time_diff)
    print("Number of frame to track (gt):", len(gt))
    print("Number of frame to track (track success):", success_frame_track)
    
    track_df_frame = track_df.frame_no
    
    gt_center_x = gt.center_x[track_df_frame].values
    gt_center_y = gt.center_y[track_df_frame].values

    track_df_center_x = track_df.center_x.values
    track_df_center_y = track_df.center_y.values
    
    plt.plot(np.sqrt((gt_center_x-track_df_center_x)**2 + (gt_center_y-track_df_center_y)**2))
    plt.xlabel("frame")
    plt.ylabel("Euclidian Distance btw gt and track")
    error = np.sum((gt_center_x-track_df_center_x)**2 + (gt_center_y-track_df_center_y)**2)
    print("Total error:", error)


















