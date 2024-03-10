import cv2
import numpy as np
import matplotlib.pyplot as plt

# 相机参数和二维码参数
fx, fy, cx, cy = 2260,2260, 1426.27, 1072.45
r_deg = np.array([[-0.99622, 0.0116326, 0.0860793],
                  [-0.013657, -0.999643, -0.0229658],
                  [0.0857814, -0.0240545, 0.996024]])
c_p = np.array([[-0.0109705], [-0.0136712], [0.171002], [1]])
t = np.array([0.0545613, 0.0319774, 0.214913])
kappa = 0.6
p_ = np.array([[-0.99622, 0.0116326, 0.0860793, 0.0545613],
               [-0.013657, -0.999643, -0.0229658, 0.0319774],
               [0.0857814, -0.0240545, 0.996024, 0.214913],
               [0, 0, 0, 1]])
R = np.array([[-0.99622, 0.0116326, 0.0860793],
              [-0.013657, -0.999643, -0.0229658],
              [0.0857814, -0.0240545, 0.996024]])
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])  # 相机内参

focal_length = 2260 # 焦距为700像素
real_width = 0.3 # 二维码的实际宽度为0.2米


# 去畸变函数
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

# 像素坐标到世界坐标的转换
def pixel_to_world(qr_center, K, R, t,p_):
    qr_center_hom = np.append(qr_center, 1)
    qr_center_cam = np.linalg.inv(K) @ qr_center_hom
    qr_center_cam_hom = np.append(qr_center_cam, 1)
    t = np.expand_dims(t, axis=-1)
    qr_center_world = np.linalg.inv(p_) @ qr_center_cam_hom
    qr_center_world /= qr_center_world[2]

    return qr_center_world[:2]

# 计算二维码中心
def calculate_qr_center(qr_corners):
    qr_center = np.mean(qr_corners, axis=0)
    qr_center_normalized = undistort_brown([qr_center], kappa)
    u, v = qr_center_normalized[0]
    s = np.array([[u], [v]])
    return s

# 计算深度
def calculate_depth(focal_length, real_width, image_width):
    depth = (focal_length * real_width) / image_width
    return depth


# 初始化SIFT检测器和FLANN匹配器
sift = cv2.xfeatures2d.SIFT_create()
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=30)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# 加载目标图像和模板图像
target = cv2.imread('test0309.jpg')
target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
templates = [cv2.imread('template_01.png', 0), cv2.imread('template_02.png', 0), cv2.imread('template_03.png', 0)]

# 创建目标图像的金字塔
target_gray_pyr = [target_gray]
for i in range(3):
    target_gray_pyr.append(cv2.pyrDown(target_gray_pyr[i]))

# 不同的QR码使用不同的颜色
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

# 二维码检测和匹配
qr_corners = {}  # 用于存储二维码角点的字典
qr_code_count = 1  # 二维码计数器

for i in range(3):  # Assume there are three QR codes
    # Choose the level of the pyramid to perform feature detection and matching
    level = 3
    template = templates[i]
    target_gray = target_gray_pyr[level]

    # Compute SIFT features for the template and target images
    kp1, des1 = sift.detectAndCompute(template, None)
    kp2, des2 = sift.detectAndCompute(target_gray, None)

    # Match features using FLANN
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply ratio test to find good matches
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > 10:  # Assume at least 10 good matches are needed to find a QR code
        # Find homography
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Apply homography to find the position of the QR code in the target image
        h, w = template.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        # Scale the coordinates of the corners back to the original size
        scale = 2 ** level
        dst *= scale

        # 保存四个角点的坐标
        qr_corners["qr_center{}".format(i+1)] = [(corner[0][0], corner[0][1]) for corner in dst]

        # Draw the QR code on the target image
        target = cv2.polylines(target, [np.int32(dst)], True, colors[i], 3, cv2.LINE_AA)

        # Remove the detected QR code from the target image
        mask = np.zeros(target_gray.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [np.int32(dst)], 255)
        target_gray = cv2.bitwise_and(target_gray, cv2.bitwise_not(mask))
    else:
        print("Not enough matches are found - %d/%d" % (len(good), 10))
        break

# 显示绘制了二维码的目标图像
plt.imshow(cv2.cvtColor(target, cv2.COLOR_BGR2RGB))
qr_centers_world = []
# 计算二维码中心并在图像中画出
for i in range(1, 4):  # Assume there are three QR codes
    qr_center = np.mean(qr_corners["qr_center{}".format(i)], axis=0)
    color = tuple([x/255 for x in colors[i-1][::-1]])
    plt.scatter(qr_center[0], qr_center[1], c=color)
    print("qr_center{}: ".format(i), qr_center)

    qr_center_world = pixel_to_world(qr_center, K, R, t, p_)
    qr_centers_world.append(qr_center_world)
    print("World coordinates of QR code {}: {}".format(i, qr_center_world))

plt.show()


# Compute the position of the box
position = np.mean(qr_centers_world, axis=0)
print("Position of the box: {}".format(position))

# Compute the direction of the box
direction_vector = qr_centers_world[0] - qr_centers_world[1]  # Use the first and second QR codes
yaw = np.arctan2(direction_vector[1], direction_vector[0])
print("Direction of the box: {}".format(yaw))


# 计算二维码中心
qr_corners_np = {k: np.array(v) for k, v in qr_corners.items()}
qr_center1 = np.array(qr_corners['qr_center1'])
qr_center2 = np.array(qr_corners['qr_center2'])
qr_center3 = np.array(qr_corners['qr_center3'])

qr1_center = calculate_qr_center(qr_center1)
qr2_center = calculate_qr_center(qr_center2)
qr3_center = calculate_qr_center(qr_center3)


# 计算物体位置和姿态
qr1_center_world = pixel_to_world(qr1_center, K, R, t,p_)
qr2_center_world = pixel_to_world(qr2_center, K, R, t,p_)
qr3_center_world = pixel_to_world(qr3_center, K, R, t,p_)


object_position = (qr1_center_world + qr2_center_world + qr3_center_world) / 3
dx = -qr2_center_world[0] + qr1_center_world[0]
dy = qr2_center_world[1] - qr1_center_world[1]
yaw = np.arctan2(dy, dx)

# 计算深度
image_width = qr1_center[0]-qr2_center[0]
dz = calculate_depth(focal_length, real_width, image_width)
dz = dz[0]

# 输出结果
print("Yaw: ", yaw)
print("dx: ", dx, "dy: ", dy, "dz: ", dz)

