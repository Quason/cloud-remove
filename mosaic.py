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

def adjust(fn_list):
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


def mosaic(fn1, fn2, dst_fn, vector):
    scripts = 'C:/Miniconda3/envs/python37/Lib/site-packages/osgeo/scripts/gdal_merge.py'
    dst_fn_tmp = dst_fn.replace('.tif', '_tmp.tif')
    os.system('python %s -o %s %s %s' % (scripts, dst_fn_tmp, fn1, fn2))
    options = gdal.WarpOptions(
        cutlineDSName=vector,
        cropToCutline=True,
        dstNodata=0
    )
    gdal.Warp(dst_fn, dst_fn_tmp, options=options)
    os.remove(dst_fn_tmp)


if __name__ == '__main__':
    # fn_list = [
    #     'D:/tmp/pip-test/001-TCI/S2B_MSI_2020_09_01_02_55_49_T49QFE_tci.tif',
    #     'D:/tmp/pip-test/001-TCI/S2B_MSI_2020_09_11_02_55_49_T49QFE_tci.tif',
    #     'D:/tmp/pip-test/001-TCI/S2B_MSI_2020_08_22_02_55_49_T49QFE_tci.tif',
    #     'D:/tmp/pip-test/001-TCI/S2B_MSI_2020_07_23_02_55_49_T49QFE_tci.tif',
    #     'D:/tmp/pip-test/001-TCI/S2A_MSI_2020_07_28_02_55_51_T49QFE_tci.tif',
    # ]
    # adjust(fn_list)

    mosaic(
        'D:/tmp/pip-test/001-TCI/S2B_MSI_2020_09_01_02_55_49_T49QEE_tci_mosaic_mosaic_mosaic_mosaic.tif',
        'D:/tmp/pip-test/001-TCI/S2B_MSI_2020_09_01_02_55_49_T49QFE_tci_mosaic_mosaic_mosaic_mosaic.tif',
        'D:/tmp/pip-test/001-TCI/mosaic.tif',
        'D:/data/vector/yangdongqu.json',
    )
