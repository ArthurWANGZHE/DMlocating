import numpy as np
from scipy.spatial.transform import Rotation

# 6自由度位姿，包括平移向量和四元数
pose_6dof = []

# 提取平移向量
translation = pose_6dof[:3]

# 提取四元数
quaternion = pose_6dof[3:]

# 将四元数转换为旋转矩阵
rotation_matrix = Rotation.from_quat(quaternion).as_matrix()

# 将旋转矩阵转换为欧拉角
yaw_pitch_roll = np.degrees(np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))

# 加上yaw角
yaw_angle = 45.0  # 设置yaw角度
yaw_pitch_roll[2] += yaw_angle

# 将平移向量和旋转表示组合为物体相对于相机的三维坐标
object_position = np.concatenate((translation, yaw_pitch_roll))

print("物体相对于相机的三维坐标：", object_position)