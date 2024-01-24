#!/usr/bin/env python
# -- coding: utf-8 --
import cv2
from pylibdmtx import pylibdmtx
from pylibdmtx.pylibdmtx import decode
import numpy as np
import time

# 读取图片
t1 = time.time()
img = cv2.imread('20240122143147.jpeg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# _, img = cv2.threshold(img, 40, 255, cv2.THRESH_BINARY)
# structureElement = cv2.getStructuringElement(cv2.MORPH_RECT, (29, 29), (-1, -1))
#img = cv2.erode(img, structureElement)

# 展示处理后的图片
# t2 = time.time()
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

t3 = time.time()
# 识别二维码
dec = pylibdmtx.decode(img, timeout=500, max_count=1)

"""
reg = pylibdmtx.dmtxRegionFindNext(dec, 0)
if reg != 0:
    msg = pylibdmtx.dmtxDecodeMatrixRegion(dec, reg, 1)
    if msg != 0:
     print(msg)
    pylibdmtx.dmtxMessageDestroy(msg)
    pylibdmtx.dmtxMessageDestroy(reg)
else:
  print("Get region failed!")
"""

result = decode(img)
t4 = time.time()
# 识别结果
# print(result)
if result:
    for barcode in result:
        print("DM code: ", barcode.data)
        print("DM code corners: ", barcode.rect)

else:
    print("没有检测到二维码")

t5 = time.time()
# 数据处理
barcode_rect = result[1].rect
u = barcode_rect.left + barcode_rect.width / 2
v = barcode_rect.top + barcode_rect.height / 2
image_width = barcode_rect.width
image_height = barcode_rect.height
dm_center = (u + image_width / 2, v + image_height / 2)

# 相机参数
fx, fy, cx, cy =15.8218,15.8218,1426.27,1072.45
real_width, real_height =0.03,0.03


# 计算相机坐标系下的坐标
depth_x = fx * real_width / image_width
depth_y = fy * real_height / image_height
depth = (depth_x + depth_y) / 2
image_point = np.array([u, v, 1]).reshape(3, 1)
camera_point = depth * np.linalg.inv(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])).dot(image_point)
print("Point in camera coordinates:\n", camera_point)

t6 = time.time()
# 计算yaw角度
image_center = (image_width / 2, image_height / 2)
dm_center = (u + image_width / 2, v + image_height / 2)
du = dm_center[0] - image_center[0]
dv = dm_center[1] - image_center[1]
yaw_angle = np.arctan2(du, fx)  # u方向的偏转角度
pitch_angle = np.arctan2(dv, fy)  # v方向的偏转角度

# 将角度从弧度转换为度
yaw_angle = np.degrees(yaw_angle)
pitch_angle = np.degrees(pitch_angle)
t7 = time.time()
print("Yaw angle: ", yaw_angle)
print("Pitch angle: ", pitch_angle)

print("识别二维码时间：", t4 - t3)
print("数据处理时间：", t6 - t5)
print("计算角度时间：", t7 - t6)
print("总时间：", t7 - t1)