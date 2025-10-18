'''
Copyright (c) 2025, Quan Nguyen, All rights reserved.
'''

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')


img = cv2.imread(DATA_PATH + '/1.png', cv2.IMREAD_GRAYSCALE)
assert img is not None, "Image not found"

# Simple Thresholding
threshold_value = 150
_,thresh1 = cv2.threshold(img,threshold_value,255,cv2.THRESH_BINARY)
_,thresh2 = cv2.threshold(img,threshold_value,255,cv2.THRESH_BINARY_INV)
_,thresh3 = cv2.threshold(img,threshold_value,255,cv2.THRESH_TRUNC)
_,thresh4 = cv2.threshold(img,threshold_value,255,cv2.THRESH_TOZERO)
_,thresh5 = cv2.threshold(img,threshold_value,255,cv2.THRESH_TOZERO_INV)

titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray',vmin=0,vmax=255)
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()

# Adaptive Thresholding
img = cv2.medianBlur(img,5)
_,th1 = cv2.threshold(img,threshold_value,255,cv2.THRESH_TOZERO_INV)
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
titles = ['Original Image', 'Global Thresholding (v = %d)' % threshold_value, \
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]
for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()


# Global Otsu → binary lung candidate mask
_, lung_bin = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Clean mask (open/close)
kernel5 = np.ones((5,5), np.uint8)
kernel9 = np.ones((9,9), np.uint8)
lung_bin = cv2.morphologyEx(lung_bin, cv2.MORPH_OPEN,  kernel5)
lung_bin = cv2.morphologyEx(lung_bin, cv2.MORPH_CLOSE, kernel9)

num, labels, stats, _ = cv2.connectedComponentsWithStats(lung_bin, connectivity=8)
lung_mask = lung_bin
if num > 1:
    areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num)]  
    areas.sort(key=lambda x: x[1], reverse=True)
    keep = {a[0] for a in areas[:2]}
    lung_mask = np.where(np.isin(labels, list(keep)), 255, 0).astype(np.uint8)

overlay = cv2.addWeighted(img, 0.7, lung_mask, 0.3, 0)

plt.figure(figsize=(10,4))
plt.subplot(1,3,1); plt.imshow(img, cmap='gray');      plt.title('Original');       plt.axis('off')
plt.subplot(1,3,2); plt.imshow(lung_mask, cmap='gray');plt.title('Otsu Lung Mask'); plt.axis('off')
plt.subplot(1,3,3); plt.imshow(overlay, cmap='gray');  plt.title('Overlay');        plt.axis('off')
plt.tight_layout(); plt.show()
