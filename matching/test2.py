import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
import math

t1 = time.time()
MIN_MATCH_COUNT = 10
template = cv2.imread('template4.png', 0)

# Create a pyramid for the template image
template_pyr = [template]
for i in range(3):
    template_pyr.append(cv2.pyrDown(template_pyr[i]))

target = cv2.imread('3.jpeg', 0)
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

t2 = time.time()
print('Time for matching: ', t2-t1)

x1,y1 = dst[0][0][0],dst[0][0][1]
x2,y2 = dst[1][0][0],dst[1][0][1]
x3,y3 = dst[2][0][0],dst[2][0][1]
x4,y4 = dst[3][0][0],dst[3][0][1]

# 相机参数
fx, fy, cx, cy =15.8218,15.8218,1426.27,1072.45
real_width, real_height =0.03,0.03


# 计算宽度
image_width = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# 计算高度
image_height = math.sqrt((x4 - x1)**2 + (y4 - y1)**2)

u = (x1 + x2 + x3 + x4) / 4
v = (y1 + y2 + y3 + y4) / 4

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
yaw_angle = 90-np.degrees(yaw_angle)
t7 = time.time()
print("Yaw angle: ", yaw_angle)
print( t7 - t1)


