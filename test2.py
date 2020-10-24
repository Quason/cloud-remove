from cv2 import cv2
import numpy as np

fg_img = cv2.imread('./data/fg1.png')
bg_img = cv2.imread('./data/bg1.png')
mask = cv2.imread('./data/mask1-4.png', 0)
center = (int(np.shape(mask)[1]/2), int(np.shape(mask)[0]/2))
output = cv2.seamlessClone(fg_img, bg_img, mask, center, cv2.MIXED_CLONE)

cv2.imwrite('./data/res1-4.png', output)
