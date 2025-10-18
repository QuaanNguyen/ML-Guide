'''
Copyright (c) 2025, Quan Nguyen, All rights reserved.
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label, regionprops
from skimage.segmentation import active_contour

img = cv2.imread('1.png', cv2.IMREAD_GRAYSCALE)
img = img.astype('float32') / 255.0

img_smooth = gaussian(img, sigma=3)
thr = threshold_otsu(img_smooth)
binary = img_smooth < thr

def rect_init(top, bottom, left, right, pts_per_edge=80):
    r = np.concatenate([
        np.linspace(top, bottom, pts_per_edge),
        np.full(pts_per_edge, bottom),
        np.linspace(bottom, top, pts_per_edge),
        np.full(pts_per_edge, top),
    ])
    c = np.concatenate([
        np.full(pts_per_edge, left),
        np.linspace(left, right, pts_per_edge),
        np.full(pts_per_edge, right),
        np.linspace(right, left, pts_per_edge),
    ])
    return np.column_stack([r, c])

init_L = rect_init(top=10, bottom=80, left=30, right=60)
init_R = rect_init(top=10, bottom=70, left=70, right=100)

# Run snakes
snake_L = active_contour(gaussian(binary, sigma=3),
                         init_L, alpha=0.05, beta=10, gamma=0.001, max_num_iter=800)
snake_R = active_contour(gaussian(binary, sigma=3),
                         init_R, alpha=0.05, beta=10, gamma=0.001, max_num_iter=800)


# --- Plot ---
fig, ax = plt.subplots(1,2, figsize=(8,4))
ax[0].imshow(img, cmap='gray')
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(binary, cmap='gray')
ax[1].set_title('Snakes on Otsu Thresholded Image')
for init, snake, col in [(init_L, snake_L, 'r'), (init_R, snake_R, 'y')]:
    ax[1].plot(init[:,1], init[:,0], '--', lw=2, color=col)
    ax[1].plot(snake[:,1], snake[:,0], '-',  lw=2, color=col)
ax[1].axis('off')

plt.savefig('snakes_two_lungs.png', dpi=300, bbox_inches='tight')
plt.show()
