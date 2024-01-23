import numpy as np
import cv2
from matplotlib import pyplot as plt
import time

t1 = time.time()
MIN_MATCH_COUNT = 10
template = cv2.imread('template4.png', 0)

# Create a pyramid for the template image
template_pyr = [template]
for i in range(3):
    template_pyr.append(cv2.pyrDown(template_pyr[i]))

target = cv2.imread('3.jpeg', 0)
target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

# Create a pyramid for the target image
target_pyr = [target]
for i in range(3):
    target_pyr.append(cv2.pyrDown(target_pyr[i]))

sift = cv2.xfeatures2d.SIFT_create()
i=1
# Perform feature detection and matching at each level of the pyramid
if i ==1:
    template = template_pyr[i]
    target = target_pyr[i]

    kp1, des1 = sift.detectAndCompute(template, None)
    kp2, des2 = sift.detectAndCompute(target, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=30)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = template.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        target = cv2.polylines(target, [np.int32(dst)], True, (255, 0, 0), 3, cv2.LINE_AA)

        print("Coordinates of the four corners: ")
        for pt in dst:
            print(pt[0])
        t2 = time.time()
        print('Time for matching: ', t2 - t1)

    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=None,
                       matchesMask=matchesMask,
                       flags=2)

    result = cv2.drawMatches(template, kp1, target, kp2, good, None, **draw_params)
    plt.imshow(result, 'gray')
    plt.show()


print('Time for matching: ', t2-t1)
