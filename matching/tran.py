import numpy as np
from scipy.spatial.transform import Rotation

# 6自由度位姿，包括平移向量和四元数
pose_6dof = [0.06669979906670839, -0.02478541374136259, 0.18369319154719987, 2.6995039189176984, 4.761050217905704, 180.26645980876273,0]

# 提取平移向量
translation = pose_6dof[:3]

# 提取四元数
quaternion = pose_6dof[3:]

# 将四元数转换为旋转矩阵
rotation_matrix = Rotation.from_quat(quaternion).as_matrix()

# 将旋转矩阵转换为欧拉角
yaw_pitch_roll = Rotation.from_matrix(rotation_matrix).as_euler('zyx', degrees=True)

# 加上yaw角
yaw_angle = 45.0  # 设置yaw角度
yaw_pitch_roll[0] += yaw_angle  # 注意：在'zyx'顺序中，yaw对应的是第一个元素

# 将平移向量和旋转表示组合为物体相对于相机的三维坐标
object_position = np.concatenate((translation, yaw_pitch_roll))



print("物体相对于相机的三维坐标：", object_position)