import numpy as np
x1,y1 = dst[0][0][0],dst[0][0][1]
x2,y2 = dst[1][0][0],dst[1][0][1]
x3,y3 = dst[2][0][0],dst[2][0][1]
x4,y4 = dst[3][0][0],dst[3][0][1]


# 相机参数
fx, fy, cx, cy =0.0158218,0.0158218,1426.27,1072.45
real_width, real_height =0.1,0.033
kappa = -179.202


# 去畸变函数
def undistort_brown_single_pt(pt, kappa, max_iter=10):
    x_distorted, y_distorted = pt
    r_distorted_sq = x_distorted**2 + y_distorted**2

    # 初始估计
    x_undistorted, y_undistorted = pt

    for _ in range(max_iter):
        r_undistorted_sq = x_undistorted**2 + y_undistorted**2
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

# 畸变系数 kappa
kappa = -179

# 三联二维码矩形的四个角点的像素坐标
qr_corners = np.array([[x1, y1],
                       [x2, y2],
                       [x3, y3],
                       [x4, y4]], dtype='float32')

# 三联二维码的实际尺寸（以米为单位）
qr_width = 0.1
qr_height = 0.033

# 计算三联二维码矩形的中心点的像素坐标
qr_center = np.mean(qr_corners, axis=0)

# 将二维码的像素坐标转换为归一化的图像平面坐标
qr_center_normalized = undistort_brown([qr_center], kappa)

# 计算三联二维码在图像中的像素宽度
w = np.linalg.norm(qr_corners[0] - qr_corners[1])

# 计算二维码在相机坐标系中的深度（Z坐标）
dz = fx * qr_width / w

# 计算二维码在相机坐标系中的X和Y坐标
dx = qr_center_normalized[0, 0] * dz
dy = qr_center_normalized[0, 1] * dz

print(dx,dy,dz)