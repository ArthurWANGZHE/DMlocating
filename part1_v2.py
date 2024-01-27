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
