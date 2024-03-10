import cv2
import numpy as np
import matplotlib.pyplot as plt


fx, fy, cx, cy = 0.0158218, 0.0158218, 1426.27, 1072.45
# 三联二维码的实际尺寸（以米为单位）
# qr_width = 0.1
# qr_height = 0.033

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

"""
def perspective_transform(qr_corners):
    dst_pts = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(qr_corners, dst_pts)

    return M
def compute_qr_center(qr_corners):
    M = perspective_transform(qr_corners)
    transformed_corners = cv2.perspectiveTransform(qr_corners[None, :, :], M)
    qr_center = np.mean(transformed_corners, axis=1)

    return qr_center
    
"""
def pixel_to_world(qr_center, K, R, t,p_):
    qr_center_hom = np.append(qr_center, 1)
    qr_center_cam = np.linalg.inv(K) @ qr_center_hom
    qr_center_cam_hom = np.append(qr_center_cam, 1)
    t = np.expand_dims(t, axis=-1)
    qr_center_world = np.linalg.inv(p_) @ qr_center_cam_hom
    qr_center_world /= qr_center_world[2]

    return qr_center_world[:2]



# Initialize SIFT detector and FLANN matcher
sift = cv2.xfeatures2d.SIFT_create()
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=30)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Load the target image
target = cv2.imread('57.jpeg')
target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

# Load the template images
templates = [cv2.imread('template_01.png', 0), cv2.imread('template_02.png', 0), cv2.imread('template_03.png', 0)]

# Create a pyramid for the target image
target_gray_pyr = [target_gray]
for i in range(3):
    target_gray_pyr.append(cv2.pyrDown(target_gray_pyr[i]))

# Different colors for different QR codes
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

qr_corners = {}  # Initialize an empty dictionary to store the corners of the QR codes
qr_code_count = 1  # Initialize a counter for the QR codes

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
        qr_corners["qr_center{}".format(qr_code_count)] = [(corner[0][0], corner[0][1]) for corner in dst]
        qr_code_count += 1

        # Draw the QR code on the target image
        target = cv2.polylines(target, [np.int32(dst)], True, colors[i], 3, cv2.LINE_AA)

        # Remove the detected QR code from the target image
        mask = np.zeros(target_gray.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [np.int32(dst)], 255)
        target_gray = cv2.bitwise_and(target_gray, cv2.bitwise_not(mask))
    else:
        print("Not enough matches are found - %d/%d" % (len(good), 10))
        break

# Display the target image with the QR codes drawn on it
plt.imshow(cv2.cvtColor(target, cv2.COLOR_BGR2RGB))
plt.show()

qr_corners_np = {k: np.array(v) for k, v in qr_corners.items()}


qr_center1 = np.array(qr_corners['qr_center1'])
qr_center2 = np.array(qr_corners['qr_center2'])
qr_center3 = np.array(qr_corners['qr_center3'])

def calculate_qr_center(qr_corners):
    qr_center = np.mean(qr_corners, axis=0)
    qr_center_normalized = undistort_brown([qr_center], kappa)
    u, v = qr_center_normalized[0]
    s = np.array([[u], [v]])
    return s

qr1_center = calculate_qr_center(qr_center1)
qr2_center = calculate_qr_center(qr_center2)
qr3_center = calculate_qr_center(qr_center3)

z=qr1_center[0]-qr2_center[0]


qr1_center_world = pixel_to_world(qr1_center, K, R, t,p_)
qr2_center_world = pixel_to_world(qr2_center, K, R, t,p_)
qr3_center_world = pixel_to_world(qr3_center, K, R, t,p_)

object_position = (qr1_center_world + qr2_center_world + qr3_center_world) / 3
dx = -qr2_center_world[0] + qr1_center_world[0]
dy = qr2_center_world[1] - qr1_center_world[1]
# dz = qr2_center_world[2] - qr1_center_world[2]
yaw = np.arctan2(dy, dx)

def calculate_depth(focal_length, real_width, image_width):
    # 使用上述公式计算深度
    depth = (focal_length * real_width) / image_width
    return depth


focal_length = 2260 # 焦距为700像素
real_width = 0.3 # 二维码的实际宽度为0.2米
image_width = z[0] # 二维码在图像中的宽度为100像素

dz = calculate_depth(focal_length, real_width, image_width)



print("Yaw: ", yaw)
print("dx: ", dx, "dy: ", dy, "dz: ", dz)



