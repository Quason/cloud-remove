import os
import sys

from cv2 import cv2
import numpy as np
import gdal

import utils

fn_list = [
    'C:/Users/Admin/Desktop/rgb_mosaic/S2B_MSI_2020_09_01_02_55_49_T49QFE_tci.tif',
    'C:/Users/Admin/Desktop/rgb_mosaic/S2B_MSI_2020_09_11_02_55_49_T49QFE_tci.tif',
    'C:/Users/Admin/Desktop/rgb_mosaic/S2B_MSI_2020_08_22_02_55_49_T49QFE_tci.tif',
    'C:/Users/Admin/Desktop/rgb_mosaic/S2B_MSI_2020_07_23_02_55_49_T49QFE_tci.tif',
    'C:/Users/Admin/Desktop/rgb_mosaic/S2A_MSI_2020_07_28_02_55_51_T49QFE_tci.tif',
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

# 判断基准图像的云分布
print('基准影像云检测...')
img = cv2.imread(fn_list[0])
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_std = np.std(img.astype(float), axis=2)
key_cloud = (img_gray==max_stack) * (img_gray>150) * (img_std<10) + (img_gray==255) # cloud
cloud0 = np.zeros(key_cloud.shape, np.uint8)
cloud0[key_cloud] = 255
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
cloud0 = cv2.morphologyEx(cloud0, cv2.MORPH_OPEN, kernel) # 开操作去除碎斑
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
for i in range(20):
    cloud0 = cv2.dilate(cloud0, kernel)
# dst_fn = fn_list[0].replace('.tif', '_cloud_shadow.tif')
# utils.raster2tif(cloud0.astype(np.uint8), geo_trans, proj_ref, dst_fn, type='uint8', mask=True)

# 判断其他影像的云分布
print('其它影像云检测...')
cloud_stack = np.zeros((np.shape(cloud0)[0], np.shape(cloud0)[1], len(fn_list)-1), np.uint8)
rgb_stack = []
for i, item in enumerate(fn_list[1:]):
    print('    %s...' % item)
    img = cv2.imread(item)
    rgb_stack.append(img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_std = np.std(img.astype(float), axis=2)
    key_cloud = (img_gray==max_stack) * (img_gray>150) * (img_std<10) + (img_gray==255) # cloud
    dst_fn = fn_list[0].replace('.tif', '_cloud_shadow.tif')
    cloud = np.zeros(key_cloud.shape, np.uint8)
    cloud[key_cloud] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cloud = cv2.morphologyEx(cloud, cv2.MORPH_OPEN, kernel) # 开操作去除碎斑
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    for n in range(10):
        cloud = cv2.dilate(cloud, kernel)
    cloud_stack[:, :, i] = (cloud==255).astype(np.uint8)

# 遍历寻找最合适的影像进行替换
print('泊松融合...')
labels_struct = cv2.connectedComponentsWithStats(cloud0, connectivity=8)
label_cnt = np.shape(cloud0)[0] * np.shape(cloud0)[1]
img_bg = cv2.imread(fn_list[0])
kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
for i in range(labels_struct[0]):
    area_tmp = labels_struct[2][i][4]
    cloud_cnt = np.Inf
    if (area_tmp < label_cnt*0.5):
        key = labels_struct[1]==i
        key_uint8 = key.astype(np.uint8) * 255
        key_dilation = cv2.dilate(key_uint8, kernel_dilate)
        key_edge = (key_uint8==0) * (key_dilation==255)
        for j in range(len(fn_list)-1):
            cloud_tmp = cloud_stack[:,:,j]
            cloud_cnt_tmp = np.sum(cloud_tmp[key_edge])
            if cloud_cnt_tmp == 0:
                cloud_cnt = cloud_cnt_tmp
                select_index = j
                break
            elif cloud_cnt_tmp<cloud_cnt:
                cloud_cnt = cloud_cnt_tmp
                select_index = j
        if cloud_cnt!=0:
            continue
        img_fg = rgb_stack[select_index]
        block_w0 = int(labels_struct[2][i][0])
        block_w1 = int(labels_struct[2][i][0] + labels_struct[2][i][2])
        block_h0 = int(labels_struct[2][i][1])
        block_h1 = int(labels_struct[2][i][1] + labels_struct[2][i][3])
        img_fg_sub = img_fg[block_h0:block_h1, block_w0:block_w1, :]
        mask_sub = cloud0[block_h0:block_h1, block_w0:block_w1]
        center = (int((block_w0 + block_w1)/2), int((block_h0 + block_h1)/2))
        output = cv2.seamlessClone(img_fg_sub, img_bg, mask_sub, center, cv2.NORMAL_CLONE)
        img_bg = output
output = np.flip(output, axis=2)
dst_fn = fn_list[0].replace('.tif', '_mosaic.tif')
utils.raster2tif(output, geo_trans, proj_ref, dst_fn, type='uint8', mask=True)

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
