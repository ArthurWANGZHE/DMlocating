import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
import math

t1 = time.time()
MIN_MATCH_COUNT = 10
template = cv2.imread('template_.png', 0)

# Create a pyramid for the template image
template_pyr = [template]
for i in range(3):
    template_pyr.append(cv2.pyrDown(template_pyr[i]))

target = cv2.imread('20240124163906.jpeg', 0)
target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

# Create a pyramid for the target image
target_pyr = [target]
for i in range(3):
    target_pyr.append(cv2.pyrDown(target_pyr[i]))

sift = cv2.xfeatures2d.SIFT_create()

# Choose the level of the pyramid to perform feature detection and matching
level = 3
template = template_pyr[level]
target = target_pyr[level]

kp1, des1 = sift.detectAndCompute(template, None)
kp2, des2 = sift.detectAndCompute(target, None)
des1=np.array(des1,dtype=np.float32)
des2=np.array(des2,dtype=np.float32)

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

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    h, w = template.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    # Scale the coordinates of the corners back to the original size
    scale = 2 ** level
    dst *= scale

    target = target_pyr[0]  # Use the original size target image
    target = cv2.polylines(target, [np.int32(dst)], True, (255, 0, 0), 3, cv2.LINE_AA)

    print("Coordinates of the four corners: ")
    for pt in dst:
        print(pt[0])
else:
    print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
    matchesMask = None

draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=None,
                   matchesMask=matchesMask,
                   flags=2)

result = cv2.drawMatches(template, kp1, target, kp2, good, None, **draw_params)
plt.imshow(result, 'gray')
plt.show()

x1,y1 = dst[0][0][0],dst[0][0][1]
x2,y2 = dst[1][0][0],dst[1][0][1]
x3,y3 = dst[2][0][0],dst[2][0][1]
x4,y4 = dst[3][0][0],dst[3][0][1]

print("Coordinates of the four corners: ")
print("({},{})".format(x1,y1))
print("({},{})".format(x2,y2))
print("({},{})".format(x3,y3))
print("({},{})".format(x4,y4))


# 相机参数
fx, fy, cx, cy =0.0158218,0.0158218,1426.27,1072.45
real_width, real_height =0.1,0.033


# 计算宽度
image_width = math.sqrt((x4 - x1)**2 + (y4 - y1)**2)
image_height = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

u = (x1 + x2 + x3 + x4) / 4
v = (y1 + y2 + y3 + y4) / 4

# 计算物体在相机坐标系下的三维坐标
depth_x = (real_width / image_width) * (u - cx) * fx
depth_y = (real_height / image_height) * (v - cy) * fy
depth = (real_width / image_width) * fx

# 计算物体的yaw角
yaw_angle = math.atan2(depth_x, depth)

# 将yaw角转换为度
yaw_angle = 90-math.degrees(yaw_angle)

# 计算以物体中心为原点的相机坐标
camera_x = -depth_x*1000
camera_y = -depth_y*1000
camera_z = -depth*1000

# 输出以物体中心为原点的相机坐标
print("Camera coordinates in object's frame: ({}, {}, {})".format(camera_x, camera_y, camera_z))
print("Yaw angle: ", yaw_angle)
