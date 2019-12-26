import numpy as np
import cv2

def getError(a, ref):
    return (a[0] - ref[0]) ** 2 + (a[1] - ref[1]) ** 2 + (a[2] - ref[2]) ** 2

img = cv2.imread('lenna.png', 1)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', img)

cv2.waitKey(0)

hist = {}
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        val = (img[i][j][0] // 8, img[i][j][1] // 8, img[i][j][2] // 8)
        if val in hist:
            hist[val] += 1
        else:
            hist[val] = 1

sortedHist = [(k[0] * 8, k[1] * 8, k[2] * 8) for k in sorted(hist, key=hist.get, reverse=True)]

popHist = sortedHist[:16]
popImg = np.zeros(img.shape, dtype=np.uint8)

memoize = {}
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        val = (img[i][j][0], img[i][j][1], img[i][j][2])
        if val not in memoize:
            bestEst, minError = 0, getError(popHist[0], img[i][j])
            for k in range(len(popHist)):
                if getError(popHist[k], img[i][j]) < minError:
                    minError = getError(popHist[k], img[i][j])
                    bestEst = k

            # memoize[val] = popHist[bestEst][0], popHist[bestEst][1], popHist[bestEst][2]
            memoize[val] = popHist[bestEst]
        
        # popImg[i][j][0], popImg[i][j][1], popImg[i][j][2] = memoize[val][0], memoize[val][1], memoize[val][2]
        popImg[i][j] = memoize[val]
cv2.namedWindow('image2', cv2.WINDOW_NORMAL)
cv2.imshow('image2', popImg)

cv2.imwrite('img1_1_16.png', popImg)

cv2.waitKey(0)
cv2.destroyAllWindows()