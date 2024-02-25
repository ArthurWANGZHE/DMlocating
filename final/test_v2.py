#!/usr/bin/env python
# -- coding: utf-8 --
import datetime
import re
import rospy
import sys
from ctypes import *

import halcon as ha
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

sys.path.append("/opt/MVS/Samples/64/Python/MvImport")
from MvCameraControl_class import *


# 海康相机类，使用海康sdk采图
# 该类rospy.loginfof均为严重问题,表示相机未初始化成功/相机断开/采集图片失败等
class HikCamera:
    # 类内参数
    SerialIndex = ''  # 相机序列号
    IsConnect = 0  # 相机是否连接
    nPayloadSize = -1

    # cam = MvCamera()

    # CALL_BACK_FUN = EventInfoCallBack(event_callback)

    def __init__(self, serialIndex):
        self.SerialIndex = serialIndex
        self.cam = MvCamera()

    def Connect(self):
        deviceList = MV_CC_DEVICE_INFO_LIST()
        tlayerType = MV_GIGE_DEVICE

        # ch:枚举设备 | en:Enum device
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        if ret != 0:
            rospy.loginfo("enum devices fail! ret[0x%x]" % ret)
            return

        if deviceList.nDeviceNum == 0:
            rospy.loginfo("find no device!")
            return

        # ch:查找设备
        nConnectionNum = -1

        for i in range(0, deviceList.nDeviceNum):
            mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
                strSerialNumber = ""
                for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chSerialNumber:
                    strSerialNumber = strSerialNumber + chr(per)
                tempStr = re.sub(u'\x00', '', strSerialNumber)
                rospy.loginfo(f'tempStr: {tempStr}')
                rospy.loginfo(f'self.SerialIndex: {self.SerialIndex}')
                if tempStr == self.SerialIndex:
                    nConnectionNum = i
                    rospy.loginfo(f'i: {i}')

        if nConnectionNum < 0:
            rospy.loginfo("Search camera(" + self.SerialIndex + ") failed!")
            return

        # ch:创建相机实例 | en:Creat Camera Object
        # self.cam = MvCamera()

        # ch:选择设备并创建句柄 | en:Select device and create handle
        stDeviceList = cast(deviceList.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents

        ret = self.cam.MV_CC_CreateHandle(stDeviceList)
        if ret != 0:
            rospy.loginfo("create handle fail! ret[0x%x]" % ret)
            return

        # ch:打开设备 | en:Open device
        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            rospy.loginfo("open device fail! ret[0x%x]" % ret)
            return

        # ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
        if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
            nPacketSize = self.cam.MV_CC_GetOptimalPacketSize()
            if int(nPacketSize) > 0:
                ret = self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
                if ret != 0:
                    rospy.loginfo("Warning: Set Packet Size fail! ret[0x%x]" % ret)
                    return
            else:
                rospy.loginfo("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)
                return

        # 设置心跳时间，防止程序中断，再次连接相机占用
        #        if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
        #            stParam = 3000
        #            ret = self.cam.MV_CC_SetIntValueEx("GevHeartbeatTimeout", stParam)
        #            if ret != 0:
        #                rospy.loginfo("set GevHeartbeatTimeout fail! ret[0x%x]" % ret)
        #                return

        stBool = c_bool(False)
        ret = self.cam.MV_CC_GetBoolValue("AcquisitionFrameRateEnable", stBool)
        if ret != 0:
            rospy.loginfo("get AcquisitionFrameRateEnable fail! ret[0x%x]" % ret)
            return

        # ch:设置触发模式为on | en:Set trigger mode as off
        ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_ON)
        if ret != 0:
            rospy.loginfo("set trigger mode fail! ret[0x%x]" % ret)
            return

        ret = self.cam.MV_CC_SetEnumValue("TriggerSource", MV_TRIGGER_SOURCE_SOFTWARE)
        if ret != 0:
            rospy.loginfo("set software trigger mode fail! ret[0x%x]" % ret)
            return

        # ch:获取数据包大小 | en:Get payload size
        stParam = MVCC_INTVALUE()
        memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))

        ret = self.cam.MV_CC_GetIntValue("PayloadSize", stParam)
        if ret != 0:
            rospy.loginfo("get payload size fail! ret[0x%x]" % ret)
            return

        self.nPayloadSize = stParam.nCurValue

        # ch:注册事件回调 | en:Register event callback
        # ret = cam.MV_CC_RegisterExceptionCallBack(CALL_BACK_FUN,None)
        # if ret != 0:
        #    rospy.loginfo ("register event callback fail! ret [0x%x]" % ret)

        # ch:开始取流 | en:Start grab image
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            rospy.loginfo("start grabbing fail! ret[0x%x]" % ret)
            return

        self.IsConnect = 1

    # 采集回调，已弃用
    # def event_callback(pEventInfo, pUser):
    #    pass

    # 采集图像
    # Return：
    #   0/1是否采集成功,HImage（halcon中图像数据类型）
    def CaptureImage(self):
        # 如果相机断开，尝试重连
        if self.IsConnect == 0:
            self.Connect()

        # 发送软件触发命令
        ret = self.cam.MV_CC_SetCommandValue("TriggerSoftware")
        if ret != 0:
            print("Set command value failed! ret[0x%x]" % ret)
            return 0, 0

        stFrameInfo = MV_FRAME_OUT_INFO_EX()
        memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
        pData = (c_ubyte * self.nPayloadSize)()
        ret = self.cam.MV_CC_GetOneFrameTimeout(pData, self.nPayloadSize, stFrameInfo, 1000)
        if ret == 0:
            # rospy.loginfo("get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (
            #    stFrameInfo.nWidth, stFrameInfo.nHeight, stFrameInfo.nFrameNum))

            temp = np.frombuffer(pData, dtype=np.uint8)
            x = np.ascontiguousarray(temp).reshape(-1).astype(np.uint8)
            HImage = ha.gen_image1('byte', stFrameInfo.nWidth, stFrameInfo.nHeight, x.ctypes.data)
            temp = np.asarray(pData)  # 将c_ubyte_Array转化成ndarray得到（3686400，）
            temp = temp.reshape(3072, 2048, 1)
            gray = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
            return 1, gray
        del pData
        rospy.loginfo("Grab image failed!")
        return 0, 0

    def DisConnect(self):
        # ch:停止取流 | en:Stop grab image
        ret = self.cam.MV_CC_StopGrabbing()
        if ret != 0:
            rospy.loginfo("stop grabbing fail! ret: %s" % ret)
            return

        # ch:关闭设备 | Close device
        ret = self.cam.MV_CC_CloseDevice()
        if ret != 0:
            rospy.loginfo("close deivce fail! ret[0x%x]" % ret)
            return

        # ch:销毁句柄 | Destroy handle
        ret = self.cam.MV_CC_DestroyHandle()
        if ret != 0:
            rospy.loginfo("destroy handle fail! ret[0x%x]" % ret)
            return
        self.IsConnect = 0
        return 0

    def __del__(self):
        self.DisConnect()


def undistort_brown_single_pt(pt, kappa, max_iter=10):
    x_distorted, y_distorted = pt
    r_distorted_sq = x_distorted ** 2 + y_distorted ** 2

    # 初始估计
    x_undistorted, y_undistorted = pt

    for _ in range(max_iter):
        r_undistorted_sq = x_undistorted ** 2 + y_undistorted ** 2
        distortion = 1 + kappa * r_undistorted_sq
        x_undistorted = x_distorted / distortion
        y_undistorted = y_distorted / distortion

    return np.array([x_undistorted, y_undistorted])


def undistort_brown(pts, kappa, max_iter=10):
    return np.array([undistort_brown_single_pt(pt, kappa, max_iter) for pt in pts])


class FindBox:
    # 类内变量
    CameraHandle = HikCamera("")  # 相机句柄
    BarCodeHandle = ha.hhandle  # 一维码模板
    leftRefPose = [1, 1, 1, 1, 1, 1]  # 参考位姿
    rightRefPose = [1, 1, 1, 1, 1, 1]  # 参考位姿
    FilePath = ''

    def __init__(self, left_file_name, right_file_name):
        # 由于机器人小叉不平整，左右两侧料箱需要分别标定，连接相机和cad及保存图片（放在左侧标定内）文件公用一份
        # 报错问题未标注,部分错误由halcon自带的提示,但初始化出错属于严重问题

        # 读取相机参数文件，该文件应该包括：
        # 相机序列号，应该利用halcon软件尝试连接相机，读取其序列号(不是海康sdk软件显示的序列号)
        # 相机参数文件名 (绝对路径)
        # 相机位姿文件名 (绝对路径)
        # 二维码cad文件名 (绝对路径)
        # 正确摆放料箱位姿文件名(绝对路径)
        # 照片保存文件夹 (绝对路径),本参数目前不使用，为预留参数
        left_file = open(left_file_name, 'r')  # 打开左侧标定文件
        right_file = open(right_file_name, 'r')  # 打开右侧标定文件
        left_file_data = left_file.readlines()  # 读取所有行
        right_file_data = right_file.readlines()  # 读取所有行

        # 头两行为预留说明
        self.camIndex = left_file_data[2].replace('\n', '')
        leftCamParaPath = left_file_data[3].replace('\n', '')
        leftCamPosePath = left_file_data[4].replace('\n', '')
        rightCamParaPath = right_file_data[3].replace('\n', '')
        rightCamPosePath = right_file_data[4].replace('\n', '')
        cadPath = left_file_data[5].replace('\n', '')
        leftRefPosePath = left_file_data[6].replace('\n', '')
        rightRefPosePath = right_file_data[6].replace('\n', '')
        self.FilePath = left_file_data[7].replace('\n', '')

        # 读取相机参数和相机位姿
        leftCamParam = ha.read_cam_par(leftCamParaPath)
        leftCamPose = ha.read_pose(leftCamPosePath)
        rightCamParam = ha.read_cam_par(rightCamParaPath)
        rightCamPose = ha.read_pose(rightCamPosePath)
        # 读取二维码模板
        Contours, DxfStatus = ha.read_contour_xld_dxf(cadPath, [], [])
        HomMat2DIdentity = ha.hom_mat2d_identity()
        HomMat2DScale = ha.hom_mat2d_scale(HomMat2DIdentity, 0.0001, 0.0001, 0.5, 0.5)
        ContoursTrans = ha.affine_trans_contour_xld(Contours, HomMat2DScale)

        # 建立二维码模板
        self.leftModelID = ha.create_planar_calib_deformable_model_xld(ContoursTrans, leftCamParam, leftCamPose, 'auto',
                                                                       [], [],
                                                                       'auto', 0.9, [], 'auto', 0.9, [], 'auto', 'none',
                                                                       'ignore_local_polarity', 5, [], [])
        self.rightModelID = ha.create_planar_calib_deformable_model_xld(ContoursTrans, rightCamParam, rightCamPose,
                                                                        'auto', [], [],
                                                                        'auto', 0.9, [], 'auto', 0.9, [], 'auto',
                                                                        'none',
                                                                        'ignore_local_polarity', 5, [], [])
        # 建立一维码识别库
        self.BarCodeHandle = ha.create_bar_code_model([], [])
        ha.set_bar_code_param(self.BarCodeHandle, 'element_size_min', 1.5)

        # 读取正确拍照参考位姿
        self.leftRefPose = ha.read_pose(leftRefPosePath)
        self.rightRefPose = ha.read_pose(rightRefPosePath)

    # 拍照二次定位
    # 参数：
    #   IsAlwaysSaveImage(0/1):是否总是存图
    # Return:
    #   1,dx,dy,dz,rad,DecodedDataStrings[0]
    #   是否成功，dx，dy，dz(单位：mm)，rad(单位:°)，一维码code(string)

    def LocateAndRecong(self, IsAlwaysSaveImage, LeftOrRight):
        # 拍照
        print(f"@#$ {datetime.datetime.now()}, Begin LocateAndRecong")
        tempIndex, gray = self.CameraHandle.CaptureImage()
        retry_times = 0
        while retry_times < 3 and tempIndex == 0:
            rospy.loginfo(f"{datetime.datetime.now()}, Capture Image failed.Try again. retry_times: {retry_times}")
            self.CameraHandle = HikCamera(self.camIndex)
            self.CameraHandle.Connect()
            tempIndex, gray = self.CameraHandle.CaptureImage()
            retry_times += 1
        if tempIndex == 0:
            # 严重问题
            rospy.loginfo(f"@#$ {datetime.datetime.now()}, Capture Image failed.")
            return 0, 0, 0, 0, 0, 0

        # 查找二维码定位
        MIN_MATCH_COUNT = 10
        template = cv2.imread('/home/zhang/my_hocol/src/ssr_pkg/scripts/template_.png', 0)

        # Create a pyramid for the template image
        template_pyr = [template]
        for i in range(3):
            template_pyr.append(cv2.pyrDown(template_pyr[i]))

        target = gray
        target = cv2.normalize(target, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

        # Create a pyramid for the target image
        target_pyr = [target]
        for i in range(3):
            target_pyr.append(cv2.pyrDown(target_pyr[i]))

        sift = cv2.SIFT_create()

        # Choose the level of the pyramid to perform feature detection and matching
        level = 3
        template = template_pyr[level]
        target = target_pyr[level]

        kp1, des1 = sift.detectAndCompute(template, None)
        kp2, des2 = sift.detectAndCompute(target, None)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=30)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)


        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = template.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        scale = 2 ** level
        dst *= scale

        target = target_pyr[0]  # Use the original size target image
        target = cv2.polylines(target, [np.int32(dst)], True, (255, 0, 0), 3, cv2.LINE_AA)

        print("Coordinates of the four corners: ")
        for pt in dst:
            print(pt[0])


        x1, y1 = dst[0][0][0], dst[0][0][1]
        x2, y2 = dst[1][0][0], dst[1][0][1]
        x3, y3 = dst[2][0][0], dst[2][0][1]
        x4, y4 = dst[3][0][0], dst[3][0][1]

        # 相机参数
        fx, fy, cx, cy = 0.0158218, 0.0158218, 1426.27, 1072.45
        # 三联二维码的实际尺寸（以米为单位）
        qr_width = 0.1
        qr_height = 0.033

        # 旋转向量（角度）
        # 1.32086, 4.93809, 180.669
        r_deg = np.array([1.32086, 4.93809, 180.669])
        R, _ = cv2.Rodrigues(r_deg)
        R_inv = np.linalg.inv(R)
        # 平移向量
        # 0.0545613, 0.0319774, 0.214913]
        t = np.array([0.0545613, 0.0319774, 0.214913])
        kappa = -179.202
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]])

        # 相机内参
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]])

        # 三联二维码矩形的四个角点的像素坐标
        qr_corners = np.array([[x1, y1],
                               [x2, y2],
                               [x3, y3],
                               [x4, y4]], dtype='float32')

        # 计算三联二维码矩形的中心点的像素坐标
        qr_center = np.mean(qr_corners, axis=0)

        # 将二维码的像素坐标转换为归一化的图像平面坐标
        qr_center_normalized = undistort_brown([qr_center], kappa)
        u, v = qr_center_normalized[0]
        s = np.array([[u], [v], [1]])
        k_inv = np.linalg.inv(K)
        w = np.linalg.norm(qr_corners[0] - qr_corners[1])
        a = R_inv @ k_inv @ s
        x = a[0]
        y = a[1]
        z = a[2]
        b = R_inv @ t
        x_1 = b[0]
        y_1 = b[1]
        z_1 = b[2]
        dz = fx * qr_width / w
        dx = (x - x_1) / dz
        dy = (y - y_1) / dz
        dz = (z - z_1) / dz
        rad = math.atan2(R[2][1], R[2][2]) * 180 / math.pi
        print(dx, dy, dz, rad)

        """
        # 一维码识别，DecodedDataStrings是string类型
        SymbolRegions, DecodedDataStrings = ha.find_bar_code(Image, self.BarCodeHandle, 'Code 128')

        # 预留接口，保存图片以备后续查看
        if IsAlwaysSaveImage == 1:
            # HomMat3D = ha.pose_to_hom_mat3d(self.RefPose)
            # FoundContour = ha.gen_empty_obj()
            # ModelContours = ha.get_deformable_model_contours(self.ModelID, 1)
            # NumberContour = ha.count_obj(ModelContours)
            # for x in [NumberContour]:
            #    ObjectSelected = ha.select_obj(ModelContours, x)
            #    Y, X = ha.get_contour_xld(ObjectSelected)
            #    Z = ha.gen_tuple_const(X.count, 0)
            #    Xc, Yc, Zc = ha.affine_trans_point_3d(HomMat3D, X, Y, Z)
            #    R, C = ha.project_3d_point(Xc, Yc, Zc, CamParam)
            #    ModelWorld = ha.gen_contour_polygon_xld(R, C)
            #    FoundContour = ha.concat_obj(FoundContour, ModelWorld)

            # Region = ha.gen_region_contour_xld(FoundContour, 'filled')
            # Region = ha.dilation_circle(Region, 1.5)
            # ImageR = ha.paint_region(Region, Image, 0, 'fill')
            # ImageG = ha.paint_region(Region, Image, 255, 'fill')
            # ImageResult = ha.compose3(ImageR, ImageG, ImageG)

            now = datetime.datetime.now()
            other_StyleTime = now.strftime("%Y%m%d%H%M%S")
            rospy.loginfo(self.FilePath + other_StyleTime + 'jpeg. OK')
            ha.write_image(Image, 'jpeg', 0, self.FilePath + other_StyleTime + '.jpeg')
        """
        # 返回执行成功和参数
        rospy.loginfo(f"@#$ {datetime.datetime.now()}, dx, dy, dz, rad: {dx, dy, dz, rad}")
        return 1, dx, dy, dz, rad  # DecodedDataStrings[0] if DecodedDataStrings else 0

    def __del__(self):
        # 软件关闭前或软件重启前等应该尝试关闭相机
        self.CameraHandle.DisConnect()


if __name__ == '__main__':
    # find_box = FindBox("/home/gort/Desktop/camera/boxpileup.cofig")
    # find_box.LocateAndRecong(1)
    find_box = FindBox(
        "/home/zhang/Desktop/camera/boxpileup.cofig", "/home/zhang/Desktop/camera/boxpileup.cofig")
    find_box.LocateAndRecong(IsAlwaysSaveImage=1, LeftOrRight=True)
