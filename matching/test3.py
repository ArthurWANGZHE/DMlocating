import cv2
import numpy as np
import ezdxf

def template_matching(template, image):
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.5  # 设置匹配阈值，可以根据需要调整

    # 获取匹配结果的位置
    locations = np.where(result >= threshold)
    locations = list(zip(*locations[::-1]))  # 转换为(x, y)坐标格式

    return locations

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

# 读取原始图像
image_path = '20240122143147.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image = cv2.Canny(image, 128, 200)

# 提取模板数据
template_data = extract_template_from_dxf('Drawing7.dxf')

# 进行模板匹配
for start_point, end_point in template_data:
    # 创建模板图像
    template = np.zeros_like(image)
    cv2.line(template, tuple(map(int, start_point)), tuple(map(int, end_point)), 255, 1)

    # 执行模板匹配
    locations = template_matching(template, image)

    # 在原始图像上标记匹配结果
    for loc in locations:
        top_left = loc
        bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
        cv2.rectangle(image, top_left, bottom_right, 255, 2)

# 显示结果图像
cv2.imshow("Matching Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
