import numpy as np
import cv2
from matplotlib import pyplot as plt
import time

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


def perspective_transform(qr_corners):
    # Define the destination points. These points are chosen such that
    # the QR code, after perspective transformation, is upright and centered.
    dst_pts = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)

    # Compute the perspective transformation matrix
    M = cv2.getPerspectiveTransform(qr_corners, dst_pts)

    return M
def compute_qr_center(qr_corners):
    M = perspective_transform(qr_corners)

    # Apply the perspective transformation to the QR code corners
    transformed_corners = cv2.perspectiveTransform(qr_corners[None, :, :], M)

    # Compute the center of the transformed QR code
    qr_center = np.mean(transformed_corners, axis=1)

    return qr_center
"""
def pixel_to_world(qr_center, K, R, t):
    # Convert the QR center to homogeneous coordinates
    qr_center_hom = np.append(qr_center, 1)

    # Apply the inverse of the intrinsic matrix
    qr_center_cam = np.linalg.inv(K) @ qr_center_hom

    # Apply the inverse of the extrinsic matrix
    qr_center_world = np.linalg.inv(np.hstack((R, t))) @ qr_center_cam

    # Convert back from homogeneous coordinates
    qr_center_world /= qr_center_world[2]

    return qr_center_world[:2]

def pixel_to_world(qr_center, K, R, t):
    # Convert the QR center to homogeneous coordinates
    qr_center_hom = np.append(qr_center, 1)

    # Apply the inverse of the intrinsic matrix
    qr_center_cam = np.linalg.inv(K) @ qr_center_hom

    # Convert qr_center_cam to homogeneous coordinates
    qr_center_cam_hom = np.append(qr_center_cam, 1)

    # Ensure t is a 2D array
    t = np.expand_dims(t, axis=-1)

    # Apply the inverse of the extrinsic matrix
    qr_center_world = np.linalg.inv(np.hstack((R, t))) @ qr_center_cam_hom

    # Convert back from homogeneous coordinates
    qr_center_world /= qr_center_world[2]

    return qr_center_world[:2]
"""
t1 = time.time()
MIN_MATCH_COUNT = 10
template = cv2.imread('template_01.png', 0)

# Create a pyramid for the template image
template_pyr = [template]
for i in range(3):
    template_pyr.append(cv2.pyrDown(template_pyr[i]))

target = cv2.imread('test0309.jpg', 0)
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

# result = cv2.drawMatches(template, kp1, target, kp2, good, None, **draw_params)
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

fx, fy, cx, cy = 0.0158218, 0.0158218, 1426.27, 1072.45
# 三联二维码的实际尺寸（以米为单位）
qr_width = 0.1
qr_height = 0.033

# 旋转向量（角度）
# 1.32086, 4.93809, 180.669
r_deg = np.array([[-0.99622, 0.0116326, 0.0860793],
                  [-0.013657, -0.999643, -0.0229658],
                  [0.0857814, -0.0240545, 0.996024]])

# 相机坐标
c_p = np.array([[-0.0109705],
                [-0.0136712],
                [0.171002],
                [1]])

# 平移向量
# 0.0545613, 0.0319774, 0.214913]
t = np.array([0.0545613, 0.0319774, 0.214913])
kappa = 0.6

# 4*4
p_ = np.array([[-0.99622, 0.0116326, 0.0860793, 0.0545613],
               [-0.013657, -0.999643, -0.0229658, 0.0319774],
               [0.0857814, -0.0240545, 0.996024, 0.214913],
               [0, 0, 0, 1]])
R = np.array([[-0.99622, 0.0116326, 0.0860793],
                [-0.013657, -0.999643, -0.0229658],
                [0.0857814, -0.0240545, 0.996024]])

# 相机内参
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])
k_inv = np.linalg.inv(K)

qr_corners = np.array([[x1, y1],
                       [x2, y2],
                       [x3, y3],
                       [x4, y4]], dtype='float32')
"""
# 三联二维码矩形的四个角点的像素坐标
qr_corners = np.array([[x1, y1],
                       [x2, y2],
                       [x3, y3],
                       [x4, y4]], dtype='float32')

# 计算三联二维码矩形的中心点的像素坐标
qr_center = np.mean(qr_corners, axis=0)
qr_center_normalized = undistort_brown([qr_center], kappa)
u, v = qr_center_normalized[0]
s = np.array([[u], [v], [1]])
a1 = k_inv @ s
mylist = []
for i in [0, 1, 2]:
    mylist.append(a1[i][0])
mylist.append(0)
ar = np.array(mylist)

p_inv = np.linalg.inv(p_)
result = p_inv @ ar

dx = result[0]/10
dy = result[1]/10
dz = result[2]/10
print(dx, dy, dz)
"""

# Assume qr_corners is a 4x2 array containing the corners of the QR code
qr_center = compute_qr_center(qr_corners)

# Assume K, R, and t are the intrinsic and extrinsic parameters of the camera

u, v = qr_center[0]
s = np.array([[u], [v], [1]])
a1 = k_inv @ s
mylist = []
for i in [0, 1, 2]:
    mylist.append(a1[i][0])
mylist.append(0)
ar = np.array(mylist)

p_inv = np.linalg.inv(p_)
result = p_inv @ ar

dx = result[0]/10
dy = result[1]/10
dz = result[2]/10
print(dx, dy, dz)
