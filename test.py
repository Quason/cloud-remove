import os
import sys

from cv2 import cv2
import numpy as np
import gdal

import utils

fn_list = [
    'D:/tmp/pip-test/001-TCI/S2B_MSI_2020_09_01_02_55_49_T49QFE_tci.tif',
    'D:/tmp/pip-test/001-TCI/S2B_MSI_2020_09_11_02_55_49_T49QFE_tci.tif',
    'D:/tmp/pip-test/001-TCI/S2B_MSI_2020_08_22_02_55_49_T49QFE_tci.tif',
    'D:/tmp/pip-test/001-TCI/S2B_MSI_2020_07_23_02_55_49_T49QFE_tci.tif',
    'D:/tmp/pip-test/001-TCI/S2A_MSI_2020_07_28_02_55_51_T49QFE_tci.tif',
]
ds = gdal.Open(fn_list[0])
geo_trans = ds.GetGeoTransform()
proj_ref = ds.GetProjection()
img_width = ds.RasterXSize
img_height = ds.RasterYSize
img_stack = np.zeros((img_height,img_width,len(fn_list)), np.uint8)
for i, item in enumerate(fn_list):
    img = cv2.imread(item)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_stack[:,:,i] = img_gray
min_stack = np.min(img_stack, axis=2)
max_stack = np.max(img_stack, axis=2)
median_stack = np.median(img_stack, axis=2)
imedian_stack = np.zeros((img_height,img_width), np.uint8)

img = cv2.imread(fn_list[0])
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_std = np.std(img.astype(float), axis=2)
key_cloud = (img_gray==max_stack) * (img_gray>150) * (img_std<10) + (img_gray==255) # cloud
cloud = np.zeros(key_cloud.shape, np.uint8)
cloud_cnt = key_cloud.astype(np.uint8) # 用于后期计数
cloud[key_cloud] = 255
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
cloud = cv2.morphologyEx(cloud, cv2.MORPH_OPEN, kernel) # 开操作去除碎斑
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
for i in range(20):
    cloud = cv2.dilate(cloud, kernel)
# dst_fn = fn_list[0].replace('.tif', '_cloud_shadow.tif')
# utils.raster2tif(cloud.astype(np.uint8), geo_trans, proj_ref, dst_fn, type='uint8', mask=True)

for item in fn_list[1:]:
    print('%s...' % item)
    img = cv2.imread(item)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_std = np.std(img.astype(float), axis=2)
    key_cloud = (img_gray==max_stack) * (img_gray>150) * (img_std<10) + (img_gray==255) # cloud
    dst_fn = fn_list[0].replace('.tif', '_cloud_shadow.tif')
    cloud = np.zeros(key_cloud.shape, np.uint8)
    cloud_cnt = key_cloud.astype(np.uint8) # 用于后期计数
    cloud[key_cloud] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cloud = cv2.morphologyEx(cloud, cv2.MORPH_OPEN, kernel) # 开操作去除碎斑
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    for i in range(20):
        cloud = cv2.dilate(cloud, kernel)
    utils.raster2tif(cloud.astype(np.uint8), geo_trans, proj_ref, dst_fn, type='uint8', mask=True)
    sys.exit(0)


# for i in range(len(fn_list)):
#     key = img_stack[:,:,i] == median_stack
#     imedian_stack[key] = i+1
# dst_fn = fn_list[0].replace('.tif', '_imedian.tif')
# utils.raster2tif(imedian_stack, geo_trans, proj_ref, dst_fn, type='uint8', mask=True)

# start = False
# blue_weighted = None
# green_weighted = None
# red_weighted = None
# for item in fn_list:
#     print('%s...' % item)
#     img = cv2.imread(item)
#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     key_cloud = (img_gray > min_stack) * (img_gray > 150) + (img_gray==255) # cloud
#     key_shadow = img_gray == min_stack # cloud shadow
#     img = img.astype(float)
#     if start:
#         weight = np.zeros(key_shadow.shape) + 0.5
#         # weight[key_shadow] = 0.25
#         weight[key_cloud] = 0
#         blue_weighted = (img[:,:,0] * weight) + (blue_weighted*(1-weight))
#         green_weighted = (img[:,:,1] * weight) + (green_weighted*(1-weight))
#         red_weighted = (img[:,:,2] * weight) + (red_weighted*(1-weight))
#     else:
#         weight = np.zeros(key_shadow.shape) + 1
#         # weight[key_shadow] = 0.5
#         weight[key_cloud] = 0
#         blue_weighted = img[:,:,0] * weight
#         green_weighted = img[:,:,1] * weight
#         red_weighted = img[:,:,2] * weight
#     start = True
# rgb = np.copy(img)
# rgb[:, :, 0] = red_weighted
# rgb[:, :, 1] = green_weighted
# rgb[:, :, 2] = blue_weighted
# dst_fn = fn_list[0].replace('.tif', '_res.tif')
# utils.raster2tif(rgb.astype(np.uint8), geo_trans, proj_ref, dst_fn, type='uint8', mask=True)
