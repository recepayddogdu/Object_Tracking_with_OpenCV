from importlib.resources import path
import cv2
import os
from os.path import isfile, join
import matplotlib.pyplot as plt
import tqdm

pathIn = r"2_exploratory_data_analysis/img1"
pathOut = "2_exploratory_data_analysis/MOT17_13_SDP.mp4"

files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]

# img = cv2.imread(join(pathIn,files[44]))

# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# plt.figure(), plt.imshow(img_rgb), plt.show()

fps = 25
size = (1920, 1080)
out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*"MP4V"), fps, size, True)

for i in tqdm.tqdm(files):
    print(i)
    
    filename = pathIn + "/" + i
    img = cv2.imread(filename)
    out.write(img)


print("\nend...")