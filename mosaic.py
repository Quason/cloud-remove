import os
import sys

from cv2 import cv2
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import gdal
import skimage.io
from skimage import color

import utils

def adjust0(fn_list):
    fn_base = fn_list[0]
    for i in range(len(fn_list)-1):
        print('%d of %d...' % (i+1, len(fn_list)-1))
        fn_move = fn_list[i+1]
        ds = gdal.Open(fn_move)
        geo_trans = ds.GetGeoTransform()
        proj_ref = ds.GetProjection()
        data_base = cv2.imread(fn_base).astype(float)
        data_move = cv2.imread(fn_move).astype(float)
        # 色调匹配
        # diff = np.sum(np.abs(data_move - data_base) / data_base, axis=2)
        # valid_key = diff < 0.5
        # for n_band in range(3):
        #     band_base = data_base[:, :, n_band]
        #     band_move = data_move[:, :, n_band]
        #     valid_key_t = valid_key * (band_base>10) * (band_base<240) * (band_move>10) * (band_move<240)
        #     invalid_key = (band_base==0) * (band_move==0)
        #     band_base = band_base[valid_key_t]
        #     band_move = band_move[valid_key_t]
        #     slope, intercept, r_value, p_value, std_err = st.linregress(band_move, band_base)
        #     data_move_single = data_move[:, :, n_band] * slope + intercept
        #     data_move_single[invalid_key] = 0
        #     data_move[:, :, n_band] = data_move_single
        # 确定替换条件
        data_move_gray = color.rgb2gray(data_move.astype(np.uint8)) * 255
        data_base_gray = color.rgb2gray(data_base.astype(np.uint8)) * 255
        thresh = 150
        key = (data_move_gray <= thresh) * (data_base_gray > thresh)
        # 去除碎斑
        label_cnt = np.shape(key)[0] * np.shape(key)[1]
        labels_struct = cv2.connectedComponentsWithStats(key.astype(np.uint8), connectivity=4)
        key_simp = np.zeros(key.shape, np.uint8)
        area_thresh = 200
        base_std = np.std(data_base, axis=2)
        for j in range(labels_struct[0]):
            area_tmp = labels_struct[2][j][4]
            if (area_tmp > area_thresh) and (area_tmp < label_cnt*0.1):
                key_tmp = labels_struct[1]==j
                # 云块一般为灰白色,std较低
                base_gray_extract = base_std[key_tmp]
                base_patch_std = np.mean(base_gray_extract)
                if base_patch_std < 10:
                    key_simp[key_tmp] = 1 
        key = key_simp
        # 云区域膨胀
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
        key = cv2.dilate(key.astype(np.uint8), kernel) # 膨胀
        key = key==1
        # dilated2用于确定融合范围
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        dilated2 = cv2.dilate(key.astype(np.uint8), kernel) * 255
        # 目标替换
        for j in range(3):
            band_data = data_base[:, :, j]
            data_select = data_move[:, :, j]
            band_data[key] = data_select[key]
            data_base[:, :, j] = band_data
        # 泊松融合
        mask = key.astype(np.uint8) * 255
        labels_struct = cv2.connectedComponentsWithStats(mask, connectivity=8)
        label_cnt = np.shape(mask)[0] * np.shape(mask)[1]
        bg_img = data_base.astype(np.uint8)
        data_move = data_move.astype(np.uint8)
        for i in range(labels_struct[0]):
            area_tmp = labels_struct[2][i][4]
            if (area_tmp > 100) and (area_tmp < label_cnt*0.1):
                block_w0 = int(labels_struct[2][i][0])
                block_w1 = int(labels_struct[2][i][0] + labels_struct[2][i][2])
                block_h0 = int(labels_struct[2][i][1])
                block_h1 = int(labels_struct[2][i][1] + labels_struct[2][i][3])
                fg_img = data_move[block_h0:block_h1, block_w0:block_w1, :]
                mask_sub = dilated2[block_h0:block_h1, block_w0:block_w1]
                center = (int((block_w0 + block_w1)/2), int((block_h0 + block_h1)/2))
                output = cv2.seamlessClone(fg_img, bg_img, mask_sub, center, cv2.NORMAL_CLONE)
                bg_img = output
        dst_fn = fn_base.replace('.tif', '_mosaic.tif')
        output_flip = np.copy(output)
        output_flip[:, :, 0] = output[:, :, 2]
        output_flip[:, :, 2] = output[:, :, 0]
        utils.raster2tif(output_flip.astype(np.uint8), geo_trans, proj_ref, dst_fn, type='uint8', mask=True)

        fn_base = dst_fn


def adjust(fn_list, fn_base_clear):
    img_base_clear = cv2.imread(fn_base_clear)
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
            # key_dilation = cv2.dilate(key_uint8, kernel_dilate)
            # key_edge = (key_uint8==0) * (key_dilation==255)
            for j in range(len(fn_list)-1):
                cloud_tmp = cloud_stack[:,:,j]
                cloud_cnt_tmp = np.sum(cloud_tmp[key])
                if cloud_cnt_tmp == 0:
                    cloud_cnt = cloud_cnt_tmp
                    select_index = j
                    break
                elif cloud_cnt_tmp<cloud_cnt:
                    cloud_cnt = cloud_cnt_tmp
                    select_index = j
            if cloud_cnt!=0:
                img_fg = img_base_clear
            else:
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


def mosaic(fn1, fn2, dst_fn, vector):
    scripts = 'C:/Miniconda3/envs/python37/Lib/site-packages/osgeo/scripts/gdal_merge.py'
    # mosaic1
    dst_fn_tmp = dst_fn.replace('.tif', '_tmp.tif')
    os.system('python %s -o %s %s %s' % (scripts, dst_fn_tmp, fn1, fn2))
    options = gdal.WarpOptions(
        cutlineDSName=vector,
        cropToCutline=True,
        dstNodata=0
    )
    dst_fn1 = dst_fn.replace('.tif', '_1.tif')
    gdal.Warp(dst_fn1, dst_fn_tmp, options=options)
    os.remove(dst_fn_tmp)
    # mosaic2
    dst_fn2 = dst_fn.replace('.tif', '_2.tif')
    os.system('python %s -o %s %s %s' % (scripts, dst_fn_tmp, fn2, fn1))
    options = gdal.WarpOptions(
        cutlineDSName=vector,
        cropToCutline=True,
        dstNodata=0
    )
    gdal.Warp(dst_fn2, dst_fn_tmp, options=options)
    os.remove(dst_fn_tmp)


def add_proj(fn0, fn1):
    ds = gdal.Open(fn0)
    geo_trans = ds.GetGeoTransform()
    proj_ref = ds.GetProjection()
    dst_fn = fn1.replace('.tif', '_addproj.tif')
    data = skimage.io.imread(fn1)
    utils.raster2tif(data, geo_trans, proj_ref, dst_fn, type='uint8', mask=True)


if __name__ == '__main__':
    # fn_list = [
    #     'D:/tmp/pip-test/001-TCI/2020Q2/S2B_MSI_2020_05_04_02_55_39_T49QFE_tci.tif',
    #     'D:/tmp/pip-test/001-TCI/2020Q2/S2A_MSI_2020_05_19_02_55_51_T49QFE_tci.tif',
    # ]
    # fn_base_clear = 'D:/tmp/pip-test/001-TCI/S2A_MSI_2019_09_22_02_55_41_T49QFE_tci.tif'
    # adjust(fn_list, fn_base_clear)

    # mosaic(
    #     'D:/tmp/pip-test/001-TCI/2020Q2/S2B_MSI_2020_05_04_02_55_39_T49QEE_tci_mosaic.tif',
    #     'D:/tmp/pip-test/001-TCI/2020Q2/S2B_MSI_2020_05_04_02_55_39_T49QFE_tci_mosaic.tif',
    #     'D:/tmp/pip-test/001-TCI/2020Q2/mosaic.tif',
    #     'D:/data/vector/yangdongqu.json',
    # )

    # add proj after PS
    fn0 = 'D:/tmp/pip-test/001-TCI/2020Q2/mosaic_1.tif'
    fn1 = 'D:/tmp/pip-test/001-TCI/2020Q2/mosaic.tif'
    add_proj(fn0, fn1)
