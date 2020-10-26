import os
from glob import glob
import shutil
import subprocess
import zipfile
import uuid
import time

from osgeo import gdal, osr, ogr
import numpy as np
from cv2 import cv2

import utils

class AcoliteModel():
    def __init__(self, acolite_dir, res_dir, sensor='S2A'):
        self.acolite_dir = acolite_dir
        self.rhos_fns = glob(os.path.join(acolite_dir, '*rhos*'))
        self.rhot_fns = glob(os.path.join(acolite_dir, '*rhot*'))
        self.pre_name = os.path.split((self.rhot_fns)[0])[1][0:-13]
        self.res_dir = res_dir
        if 'S2A' in os.path.split(self.rhot_fns[1])[1]:
            self.sensor ='S2A'
        else:
            self.sensor ='S2B'
        rrs_red = [item for item in self.rhot_fns if '665' in item][0]  
        ds = gdal.Open(rrs_red)
        self.proj_ref = ds.GetProjection()
        self.geo_trans = ds.GetGeoTransform()
        data = ds.GetRasterBand(1).ReadAsArray()
        self.width = np.shape(data)[1]
        self.height = np.shape(data)[0]
        self.water = None
        self.vector_mask = None

    def tci_gen(self):
        if self.sensor == 'S2A':
            blue_fn = [item for item in self.rhot_fns if 'rhot_492' in item][0]
            green_fn = [item for item in self.rhot_fns if 'rhot_560' in item][0]
            red_fn = [item for item in self.rhot_fns if 'rhot_665' in item][0]
        else:
            blue_fn = [item for item in self.rhot_fns if 'rhot_492' in item][0]
            green_fn = [item for item in self.rhot_fns if 'rhot_559' in item][0]
            red_fn = [item for item in self.rhot_fns if 'rhot_665' in item][0]
        dst_fn = os.path.join(self.res_dir, self.pre_name+'tci.tif')
        mask = (self.vector_mask != 0) + self.cloud
        # 膨胀
        mask_uint8 = np.copy(mask).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(mask_uint8, kernel) # 膨胀图像
        mask = dilated==1
        utils.rgb_generator(
            red_fn, green_fn, blue_fn, dst_fn=dst_fn, scale=1, threshold=0.3, mask=self.vector_mask != 0)

    def cloud_shadow_detection(
        self, solz, sola, vector=None, cloud_height=500.0, model_fn=None,
        pixel_size=10.0):
        if self.sensor == 'S2A':
            blue_fn = [item for item in self.rhot_fns if 'rhot_492' in item][0]
            green_fn = [item for item in self.rhot_fns if 'rhot_560' in item][0]
            red_fn = [item for item in self.rhot_fns if 'rhot_665' in item][0]
            nir_fn = [item for item in self.rhot_fns if 'rhot_833' in item][0]
            swir1_fn = [item for item in self.rhot_fns if 'rhot_1614' in item][0]
            swir2_fn = [item for item in self.rhot_fns if 'rhot_2202' in item][0]
            cirrus_fn = [item for item in self.rhot_fns if 'rhot_1373' in item][0]
        else:
            blue_fn = [item for item in self.rhot_fns if 'rhot_492' in item][0]
            green_fn = [item for item in self.rhot_fns if 'rhot_559' in item][0]
            red_fn = [item for item in self.rhot_fns if 'rhot_665' in item][0]
            nir_fn = [item for item in self.rhot_fns if 'rhot_833' in item][0]
            swir1_fn = [item for item in self.rhot_fns if 'rhot_1610' in item][0]
            swir2_fn = [item for item in self.rhot_fns if 'rhot_2186' in item][0]
            cirrus_fn = [item for item in self.rhot_fns if 'rhot_1377' in item][0]
        blue = utils.band_math([blue_fn], 'B1')
        green = utils.band_math([green_fn], 'B1')
        red = utils.band_math([red_fn], 'B1')
        ndsi = utils.band_math([green_fn, swir1_fn], '(B1-B2)/(B1+B2)')
        swir2 = utils.band_math([swir2_fn], 'B1')
        ndvi = utils.band_math([nir_fn, red_fn], '(B1-B2)/(B1+B2)')
        blue_swir = utils.band_math([blue_fn, swir1_fn], 'B1/B2')
        # step 1
        cloud_prob1 = (swir2 > 0.03) * (ndsi<0.8) * (ndvi<0.5) * (red>0.15)
        mean_vis = (blue + green + red) / 3
        cloud_prob2 = (
            np.abs(blue - mean_vis)/mean_vis
            + np.abs(green - mean_vis)/mean_vis
            + np.abs(red - mean_vis)/mean_vis) < 0.7
        cloud_prob3 = (blue - 0.5*red) > 0.08
        cloud_prob4 = utils.band_math([nir_fn,swir1_fn], 'B1/B2>0.75')
        cloud = cloud_prob1 * cloud_prob2 * cloud_prob3 * cloud_prob4
        cloud = cloud.astype(np.uint8)
        cnt_cloud = len(cloud[cloud==1])
        cloud_level = cnt_cloud / np.shape(cloud)[0] / np.shape(cloud)[1]
        print('cloud level:%.3f' % cloud_level)
        cloud_large = np.copy(cloud) * 0
        # -- only the cloud over water was saved --
        # labels_struct[0]: count;
        # labels_struct[1]: label matrix;
        # labels_struct[2]: [minY,minX,block_width,block_height,cnt]
        labels_struct = cv2.connectedComponentsWithStats(cloud, connectivity=4)
        img_h, img_w = cloud.shape
        for i in range(1, labels_struct[0]):
            patch = labels_struct[2][i]
            if patch[4] > 2000:
                cloud_large[labels_struct[1]==i] = 1
        # cloud shadow detection
        PI = 3.1415
        shadow_dire = sola + 180.0
        if shadow_dire > 360.0:
            shadow_dire -= 360.0
        cloud_height = [100+100*i for i in range(100)]
        shadow_mean = []
        for item in cloud_height:
            shadow_dist = item * np.tan(solz/180.0*PI) / 10.0
            w_offset = np.sin(shadow_dire/180.0*PI) * shadow_dist
            h_offset = np.cos(shadow_dire/180.0*PI) * shadow_dist * -1
            affine_m = np.array([[1,0,w_offset], [0,1,h_offset]])
            cloud_shadow = cv2.warpAffine(cloud_large, affine_m, (img_w,img_h))
            cloud_shadow = (cloud_shadow==1) * (cloud_large!=1)
            shadow_mean.append(np.mean(green[cloud_shadow]))
        cloud_hight_opt = cloud_height[shadow_mean.index(min(shadow_mean))]
        shadow_dist_metric = cloud_hight_opt * np.tan(solz/180.0*PI)
        if cloud_hight_opt>200 and cloud_hight_opt<10000 and shadow_dist_metric<5000:
            print('cloud height: %dm' % cloud_hight_opt)
            shadow_dist = cloud_hight_opt * np.tan(solz/180.0*PI) / pixel_size
            w_offset = np.sin(shadow_dire/180.0*PI) * shadow_dist
            h_offset = np.cos(shadow_dire/180.0*PI) * shadow_dist * -1
            affine_m = np.array([[1,0,w_offset], [0,1,h_offset]])
            cloud_shadow = cv2.warpAffine(cloud_large, affine_m, (img_w,img_h))
            cloud_shadow = (cloud_shadow==1) * (cloud_large!=1)

            cloud1 = np.copy(cloud)
            cloud1[cloud_shadow] = 2
            cloud_all = np.copy(cloud)
            cloud_all[cloud_shadow] = 1
            self.cloud = (cloud_all == 1)
            dst_fn = os.path.join(self.res_dir, self.pre_name+'cloud.tif')
            utils.raster2tif(cloud1, self.geo_trans, self.proj_ref, dst_fn, type='uint8')

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            cloud_shadow = cv2.morphologyEx(
                cloud_shadow.astype(np.uint8), cv2.MORPH_OPEN, kernel)
            cloud_shadow_key = True
        else:
            cloud_shadow_key = False
            self.cloud = cloud == 1
        cirrus = utils.band_math([cirrus_fn], 'B1')
        # step 1
        cloud_prob1 = (red - 0.07) / (0.25 - 0.07)
        cloud_prob1[cloud_prob1<0] = 0
        cloud_prob1[cloud_prob1>1] = 1
        # step 2
        cloud_prob2 = (ndsi + 0.1) / (0.2 + 0.1)
        cloud_prob2[cloud_prob2<0] = 0
        cloud_prob2[cloud_prob2>1] = 1
        cloud_prob = cloud_prob1 * cloud_prob2
        # step 3: water
        cloud_prob[blue_swir>2.5] = 0
        cloud_prob = (cloud_prob * 100).astype(np.int)
        if cloud_shadow_key:
            self.water = (ndsi > -0.1) * (cloud_prob == 0) * (cirrus<0.012) * (cloud_shadow==0)
        else:
            self.water = (ndsi > -0.1) * (cloud_prob == 0) * (cirrus<0.012)
        # vector mask
        if vector is not None:
            self.vector_mask = utils.vector2mask(blue_fn, vector)
            self.water = self.water * (self.vector_mask==0)
        else:
            self.vector_mask = self.water * 0


def atms_corr_acolite(
    src_dir,
    export_dir,
    src_config_fn,
    aerosol_corr,
    l2w_parameters,
    sub_lim=''):
    """atmospheric correction with ACOLITE

    Args:
        src_dir (str): L1C file directory
        export_dir (str): export directory
        src_config_fn (str): configuration file
        config_fn (str): original configure file (.txt)
        aerosol_corr (str): aerosor correction method
    """
    old_list = glob(os.path.join(export_dir, '*'))
    for item in old_list:
        os.remove(item)
    dst_config_fn = os.path.join(export_dir, os.path.split(src_config_fn)[1])
    fp_dst = open(dst_config_fn, 'w')
    with open(src_config_fn, 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            line_user = line
            line_split = line.split('=')
            if len(line_split) == 2:
                if line_split[0] == 'l2w_parameters':
                    if sub_lim:
                        line_add = 'limit=%s\n' % sub_lim
                        line_user = '%s%s=%s\n' % (line_add, line_split[0], l2w_parameters)
                    else:
                        line_user = '%s=%s\n' % (line_split[0], l2w_parameters)
                elif line_split[0] == 'aerosol_correction':
                    line_user = '%s=%s\n' % (line_split[0], aerosol_corr)
                elif line_split[0] == 'inputfile':
                    line_user = '%s=%s\n' % (line_split[0], src_dir)
                elif line_split[0] == 'output':
                    line_user = '%s=%s\n' % (line_split[0], export_dir)
            fp_dst.write(line_user)
    fp_dst.close()
    # run in command line
    print('ACOLITE is running...')
    try:
        process_flag = subprocess.run(
            ['acolite', '--cli', '--settings=%s' % dst_config_fn],
            stdout=subprocess.PIPE
        )
        if process_flag.returncode == 0:
            print('acolite process success!')
        else:
            print('acolite process failed!')
    except:
        print('acolite process failed!')


def preprocess(ifile, opath, vector):
    if os.path.isfile(ifile):
        # extract file
        fz = zipfile.ZipFile(ifile, 'r')
        for file in fz.namelist():
            fz.extract(file, os.path.split(ifile)[0])
        path_name = os.path.split(fz.namelist()[0])[0]
        path_L1C = os.path.join(os.path.split(ifile)[0], path_name)
    else:
        path_L1C = ifile
    if path_L1C[-1] == '\\' or path_L1C[-1] == '/':
        path_L1C = path_L1C[0:-1]
    root_dir = os.path.dirname(os.path.abspath(__file__))
    acolite_dir = os.path.join(opath, '__temp__', 'ACOLITE')
    os.makedirs(acolite_dir, exist_ok=True)
    acolite_config = os.path.join(root_dir, 'resource/acolite_python_settings.txt')
    l2_list = ''
    jp2_b2_mid = os.path.join(path_L1C, 'GRANULE')
    jp2_b2_mid = glob(os.path.join(jp2_b2_mid, '*'))[0]
    jp2_b2 = glob(os.path.join(jp2_b2_mid, 'IMG_DATA', '*B02.jp2'))[0]
    # sub-region
    jp2_b2_mid = os.path.join(path_L1C, 'GRANULE')
    jp2_b2_mid = glob(os.path.join(jp2_b2_mid, '*'))[0]
    jp2_b2 = glob(os.path.join(jp2_b2_mid, 'IMG_DATA', '*B02.jp2'))[0]
    ds = gdal.Open(jp2_b2)
    geo_trans = ds.GetGeoTransform()
    proj_ref = ds.GetProjection()
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    ds = None
    vds = ogr.Open(vector)
    lyr = vds.GetLayer()
    v_bound = lyr.GetExtent()
    img_bound = (geo_trans[0], geo_trans[0]+(geo_trans[1])*xsize,
        geo_trans[3]+geo_trans[5]*ysize, geo_trans[3])
    src_epsg = int(utils.get_epsg(jp2_b2))
    if src_epsg != 4326:
        point_SW = utils.coord_trans(src_epsg, 4326, img_bound[0], img_bound[2])
        point_NE = utils.coord_trans(src_epsg, 4326, img_bound[1], img_bound[3])
        # China only
        if point_SW[1] > point_SW[0]:
            point_SW = [point_SW[1], point_SW[0]]
        if point_NE[1] > point_NE[0]:
            point_NE = [point_NE[1], point_NE[0]]
        img_bound = (point_SW[0], point_NE[0], point_SW[1], point_NE[1])
    # lat_min, lon_min, lat_max, lon_max
    sub_lim = (
        round(max(v_bound[2], img_bound[2])-0.01, 4),
        round(max(v_bound[0], img_bound[0])-0.01, 4),
        round(min(v_bound[3], img_bound[3])+0.01, 4),
        round(min(v_bound[1], img_bound[1])+0.01, 4)
    )
    center_lonlat = [
        (sub_lim[1] + sub_lim[3]) / 2,
        (sub_lim[0] + sub_lim[2]) / 2
    ]
    # solar position
    date_str0 = os.path.split(jp2_b2)[1]
    date_str = date_str0.split('_')[1]
    year = int(date_str[0:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])
    hour = int(date_str[9:11])
    minute = int(date_str[11:13])
    second = int(date_str[13:15])
    date = '%d/%d/%d %d:%d:%d' % (year, month, day, hour, minute, second)
    [solz, sola] = utils.calc_sola_position(center_lonlat[0], center_lonlat[1], date)
    # sub-region
    ds = gdal.Open(jp2_b2)
    geo_trans = ds.GetGeoTransform()
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    ds = None
    vds = ogr.Open(vector)
    lyr = vds.GetLayer()
    v_bound = lyr.GetExtent()
    img_bound = (geo_trans[0], geo_trans[0]+geo_trans[1]*xsize,
        geo_trans[3]+geo_trans[5]*ysize, geo_trans[3])
    src_epsg = int(utils.get_epsg(jp2_b2))
    if src_epsg != 4326:
        point_SW = utils.coord_trans(src_epsg, 4326, img_bound[0], img_bound[2])
        point_NE = utils.coord_trans(src_epsg, 4326, img_bound[1], img_bound[3])
        img_bound = (point_SW[0], point_NE[0], point_SW[1], point_NE[1])
    sub_lim_str = ','.join([str(i) for i in sub_lim])
    atms_corr_acolite(
        src_dir = path_L1C,
        export_dir = acolite_dir,
        src_config_fn = acolite_config,
        aerosol_corr = 'dark_spectrum',
        l2w_parameters = l2_list,
        sub_lim=sub_lim_str
    )
    return acolite_dir, solz, sola


if __name__ == '__main__':
    ifiles = [
        '/mnt/d/data/L1/with-insitu/yangjiang/S2A_MSIL1C_20200519T025551_N0209_R032_T49QEE_20200519T060344.SAFE',
        '/mnt/d/data/L1/with-insitu/yangjiang/S2A_MSIL1C_20200519T025551_N0209_R032_T49QFE_20200519T060344.SAFE',
    ]
    # ifiles = [
    #     '/mnt/d/data/L1/with-insitu/yangjiang/S2B_MSIL1C_20200723T025549_N0209_R032_T49QFE_20200723T055401.SAFE',
    # ]
    for ifile in ifiles:
        print('%s...' % ifile)
        opath = os.path.join('/mnt/d/tmp/pip-test/', str(uuid.uuid1()))
        vector = '/mnt/d/data/vector/yangdongqu.json'
        # vector = '/mnt/d/data/vector/tmp/yashaozhen.geojson'
        time_b = time.time()
        acolite_dir, solz, sola = preprocess(ifile, opath, vector)
        print('%.2f' % (time.time()-time_b))
        # Rrs extraction
        if 'S2A' in os.path.split(ifile[0:-1])[1]:
            acolite = AcoliteModel(acolite_dir, opath, sensor='S2A')
        else:
            acolite = AcoliteModel(acolite_dir, opath, sensor='S2B')
        acolite.cloud_shadow_detection(solz, sola, vector=None)
        acolite.tci_gen()
        shutil.rmtree(acolite_dir)
    