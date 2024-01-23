import datetime
import re
import rospy
import sys
from ctypes import *

import halcon as ha
import numpy as np

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
            rospy.loginfo("Set command value failed! ret[0x%x]" % ret)
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
            return 1, HImage
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


class FindBox:
    # 类内变量
    CameraHandle = HikCamera("")  # 相机句柄
    leftModelID = ha.hhandle  # 二维码模板
    rightModelID = ha.hhandle  # 二维码模板
    BarCodeHandle = ha.hhandle  # 一维码模板
    leftRefPose = [1, 1, 1, 1, 1, 1]  # 参考位姿
    rightRefPose = [1, 1, 1, 1, 1, 1]  # 参考位姿
    FilePath = ''

    # 初始化
    # 应该用一个相机参数文件(boxpileup.config)初始化本类
    # 每台相机参数单独文件夹保存
    # 参数：
    #   file_name:相机参数文件(boxpileup.config)的绝对路径
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
        rospy.loginfo(f"@#$ {datetime.datetime.now()}, Begin LocateAndRecong")
        tempIndex, Image = self.CameraHandle.CaptureImage()
        retry_times = 0
        while retry_times < 3 and tempIndex == 0:
            rospy.loginfo(f"{datetime.datetime.now()}, Capture Image failed.Try again. retry_times: {retry_times}")
            self.CameraHandle = HikCamera(self.camIndex)
            self.CameraHandle.Connect()
            tempIndex, Image = self.CameraHandle.CaptureImage()
            retry_times += 1
        if tempIndex == 0:
            # 严重问题
            rospy.loginfo(f"@#$ {datetime.datetime.now()}, Capture Image failed.")
            return 0, 0, 0, 0, 0, 0

        # 查找二维码定位
        if LeftOrRight:
            Pose, CovPose, Score = ha.find_planar_calib_deformable_model(Image, self.leftModelID, -0.39, 0.39, 1.3, 1.4, 1.0,
                                                                         1.0, 0.7, 1, 1, 0, 1.0, [], [])
        else:
            Pose, CovPose, Score = ha.find_planar_calib_deformable_model(Image, self.rightModelID, -0.39, 0.39, 1.3, 1.4, 1.0,
                                                                         1.0, 0.7, 1, 1, 0, 1.0, [], [])
        if len(Score) == 1:
            rospy.loginfo(f"{datetime.datetime.now()}, first find success!")
        elif len(Score) == 0:
            if LeftOrRight:
                Pose, CovPose, Score = ha.find_planar_calib_deformable_model(Image, self.leftModelID, -0.39, 0.39, 1.1, 1.6,
                                                                             1.0, 1.0, 0.6, 1, 1, 0, 1.0, [], [])
            else:
                Pose, CovPose, Score = ha.find_planar_calib_deformable_model(Image, self.rightModelID, -0.39, 0.39, 1.1, 1.6,
                                                                             1.0, 1.0, 0.6, 1, 1, 0, 1.0, [], [])
            if len(Score) == 1:
                rospy.loginfo(f"{datetime.datetime.now()}, second find success!")
            if len(Score) == 0:
                # 中等严重问题
                # 查找不到二维码,必定存图
                now = datetime.datetime.now()
                other_StyleTime = now.strftime("%Y%m%d%H%M%S")
                rospy.loginfo(self.FilePath + other_StyleTime + 'error.jpeg' + " find QR code failed!")
                ha.write_image(Image, 'jpeg', 0, self.FilePath + other_StyleTime + 'error.jpeg')
                return 2, 0, 0, 0, 0, 0

        # rospy.loginfo(len(Score))
        # rospy.loginfo(Pose)
        if LeftOrRight:
            PoseCompose = ha.pose_compose(Pose, self.leftRefPose)
        else:
            PoseCompose = ha.pose_compose(Pose, self.rightRefPose)

        # 输出dx，dy，dz，rad(单位mm和°)和一维码
        dx = PoseCompose[0] * 1000.0
        dy = PoseCompose[1] * 1000.0
        dz = PoseCompose[2] * 1000.0
        rad = PoseCompose[4]
        if rad > 180.0:
            rad = -(360.0 - rad)

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

        # 返回执行成功和参数
        rospy.loginfo(f"@#$ {datetime.datetime.now()}, dx, dy, dz, rad, DecodedDataStrings[0]: {dx, dy, dz, rad, DecodedDataStrings[0] if DecodedDataStrings else 0}")
        return 1, dx, dy, dz, rad, DecodedDataStrings[0] if DecodedDataStrings else 0

    def __del__(self):
        # 软件关闭前或软件重启前等应该尝试关闭相机
        self.CameraHandle.DisConnect()


if __name__ == '__main__':
    # find_box = FindBox("/home/gort/Desktop/camera/boxpileup.cofig")
    # find_box.LocateAndRecong(1)
    find_box = FindBox(
        "/home/gort/Desktop/camera_left/boxpileup.cofig", "/home/gort/Desktop/camera_right/boxpileup.cofig")
    find_box.LocateAndRecong(IsAlwaysSaveImage=1, LeftOrRight=True)
