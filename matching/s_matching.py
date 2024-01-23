import numpy as np
from skimage.io import imshow, imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from skimage.feature import match_template
import imageio
import time

# 读取图片
t1 = time.time()
sample = imread('20240122143147.jpg')
#sample = imageio.imread('20240122143147.jpg')[:,:3]
#sample_g = rgb2gray(sample)
fig, ax = plt.subplots(1,2,figsize=(10,5))
ax[0].imshow(sample)
ax[1].imshow(sample,cmap='gray')
ax[0].set_title('Colored Image',fontsize=15)
ax[1].set_title('Grayscale Image',fontsize=15)
t2 = time.time()
print('Time for reading image: ', t2-t1)
plt.show()

# 读取模板
fig, ax = plt.subplots(1,2,figsize=(10,10))
t3 = time.time()
template = imread('template.jpg')
ax[1].imshow(template,cmap='gray')
ax[1].set_title('Template',fontsize=15)
t4 = time.time()
print('Time for reading template: ', t4-t3)
plt.show()

# 模板匹配
t5 = time.time()
sample_mt = match_template(sample, template)
fig, ax = plt.subplots(1,2,figsize=(10,10))
ax[0].imshow(sample,cmap='gray')
ax[1].imshow(sample_mt,cmap='gray')
ax[0].set_title('Grayscale',fontsize=15)
ax[1].set_title('Template Matching',fontsize=15)
t6 = time.time()
print('Time for matching: ', t6-t5)