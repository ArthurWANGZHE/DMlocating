import cv2
import time

import ezdxf

def extract_template_from_dxf(dxf_path):
    doc = ezdxf.readfile(dxf_path)
    modelspace = doc.modelspace()

    template_points = []

    for entity in modelspace:
        if entity.dxftype() == 'LINE':
            start_point = entity.dxf.start
            end_point = entity.dxf.end
            template_points.append((start_point, end_point))

    return template_points

# 用于测试的DXF文件路径
dxf_file_path = 'Drawing7.dxf'

# 提取模板数据
template_data = extract_template_from_dxf(dxf_file_path)


o=cv2.imread("20240122143147.jpg",cv2.IMREAD_GRAYSCALE)
r1=cv2.Canny(o,128,200)
cv2.imshow("original",o)
cv2.imshow("result1",r1)


