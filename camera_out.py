import numpy as np
import cv2

# 旋转向量（角度）
r_deg = np.array([1.32086, 4.93809, 180.669])
#r_deg_inv = np.linalg.inv(r_deg)
R, _ = cv2.Rodrigues(r_deg)
R_inv = np.linalg.inv(R)
# 平移向量
t = np.array([0.0545613, 0.0319774, 0.214913])

# 将旋转向量的角度转换为弧度
r_rad = np.deg2rad(r_deg)

# 使用Rodrigues' rotation formula将旋转向量转换为旋转矩阵
R, _ = cv2.Rodrigues(r_rad)

print('旋转矩阵 R:')
print(R)
print('平移向量 t:')
print(t)

print('旋转矩阵 R 的逆:')
print(R_inv)