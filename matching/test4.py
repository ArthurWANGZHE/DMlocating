import cv2
import numpy as np
import ezdxf
from matplotlib import pyplot as plt
import time
def extract_template_from_dxf(dxf_path):
    doc = ezdxf.readfile(dxf_path)
    modelspace = doc.modelspace()

    template_points = []

    for entity in modelspace:
        if entity.dxftype() == 'LINE':
            start_point = (entity.dxf.start[0], entity.dxf.start[1])  # 只提取 x 和 y 坐标
            end_point = (entity.dxf.end[0], entity.dxf.end[1])  # 只提取 x 和 y 坐标
            template_points.append((start_point, end_point))

    return template_points

def create_template_image(template_points, size):
    template = np.zeros(size, dtype=np.uint8)

    for start_point, end_point in template_points:
        cv2.line(template, tuple(map(int, start_point)), tuple(map(int, end_point)), 255, 1)

    return template

def find_template(image, template):
    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(image, None)
    kp2, des2 = sift.detectAndCompute(template, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        h, w = template.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        return dst, mask

    else:
        return None, None


t1 = time.time()
# 读取原始图像
image_path = '20240122143147.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image = cv2.Canny(image, 128, 200)

# 创建图像金字塔
image_pyr = [image]
for i in range(3):
    image_pyr.append(cv2.pyrDown(image_pyr[i]))

# 提取模板数据
template_data = extract_template_from_dxf('Drawing7.dxf')

# 创建模板图像
template = create_template_image(template_data, image_pyr[3].shape)
template_pyr = [template]
for i in range(3):
    template_pyr.append(cv2.pyrDown(template_pyr[i]))

# 找到模板在图像中的位置
dst, mask = find_template(image_pyr[2], template_pyr[2])

if dst is not None:
    # Scale the coordinates of the corners back to the original size
    scale = 2 ** 2
    dst *= scale

    # 在原始图像上标记匹配结果
    image_color = cv2.cvtColor(image_pyr[0], cv2.COLOR_GRAY2BGR)  # Convert to color image
    image_color = cv2.polylines(image_color, [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)  # Draw red lines

    # 使用matplotlib展示结果图像
    plt.imshow(cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for matplotlib
    plt.show()

    # 打印四个角的坐标
    for pt in dst:
        print(pt[0])
else:
    print('Template not found')

t2 = time.time()
print('Time for matching: ', t2-t1)
