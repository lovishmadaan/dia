import numpy as np
import cv2

def getError(a, ref):
    return (a[0] - ref[0]) ** 2 + (a[1] - ref[1]) ** 2 + (a[2] - ref[2]) ** 2

def medianCut(data, n):
    boxes = [getBoundingBox(data)]

    if n != 1:
        boxes = cut(boxes[0])
        while len(boxes) < n:
            largestBoxId = getLargestBoxId(boxes)
            # print(boxes[largestBoxId]['area'], len(boxes))
            splittedBoxes = cut(boxes[largestBoxId])
            boxes.pop(largestBoxId)
            boxes += splittedBoxes

    return getColors(boxes)

def cut(box):
    A, B = [], []
    maxAxis = box['maxDist']
    median = getMedian(box['data'], maxAxis)

    done = []

    for i in range(box['data'].shape[0]):
        if box['data'][i][maxAxis] < median:
            A.append(box['data'][i])
            done.append(1)
        elif box['data'][i][maxAxis] > median:
            B.append(box['data'][i])
            done.append(1)
        else:
            done.append(0)

    for i in range(box['data'].shape[0]):
        if done[i] == 0:
            if 2 * len(A) < box['data'].shape[0]:
                A.append(box['data'][i])
            else:
                B.append(box['data'][i])

    split = []
    if len(A) > 0:
        split.append(getBoundingBox(np.array(A)))
    if len(B) > 0:
        split.append(getBoundingBox(np.array(B)))

    # if len(split) == 1:
    #     print(box)
    #     print(median)
    #     exit()
    return split

def getBoundingBox(data):
    box = { 'data' : data,
    0 : { 'min' : 31, 'max' : 0},
    1 : { 'min' : 31, 'max' : 0},
    2 : { 'min' : 31, 'max' : 0}}

    for i in range(data.shape[0]):
        for j in range(3):
            if data[i][j] < box[j]['min']:
                box[j]['min'] = data[i][j]
            if data[i][j] > box[j]['max']:
                box[j]['max'] = data[i][j]
    
    dist = []
    for j in range(3):
        dist.append(box[j]['max'] - box[j]['min'])
    
    box['area'] = max(dist[0], 1) * max(dist[1], 1) * max(dist[2], 1)
    maxDistance = max(dist);

    for j in range(3):
        if dist[j] == maxDistance:
            box['maxDist'] = j
    
    return box;

def getLargestBoxId(boxes):
    index, largest = 0, 0
    for i in range(len(boxes)):
        if boxes[i]['area'] > largest:
            largest = boxes[i]['area']
            index = i

    return index

def getMedian(data, j):
    hist = [0] * 32

    for i in range(data.shape[0]):
        hist[data[i][j]] += 1

    count = 0
    for i in range(32):
        count += hist[i]
        if count * 2 > len(data):
            return i

    return 31

def getColors(boxes):
    colors = []
    for box in boxes:
        color = (8 * np.mean(box['data'], axis=0)).astype(int)
        colors.append(color)

    return colors

img = cv2.imread('lenna.png', 1)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', np.array(img))

cv2.waitKey(0)

img_data = img.reshape(-1, img.shape[-1])
colors = medianCut((img_data // 8).astype(int), 16)

medCutImg = np.zeros(img.shape, dtype=np.uint8)
dither = True

memoize = {}
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        val = (img[i][j][0], img[i][j][1], img[i][j][2])
        if val not in memoize:
            bestEst, minError = 0, getError(colors[0], img[i][j])
            for k in range(len(colors)):
                if getError(colors[k], img[i][j]) < minError:
                    minError = getError(colors[k], img[i][j])
                    bestEst = k

            memoize[val] = colors[bestEst]
        
        medCutImg[i][j] = memoize[val]
        if dither and i + 1 < img.shape[0] and j + 1 < img.shape[1]:
            error = img[i][j].astype(int) - medCutImg[i][j].astype(int)
            pixels = [img[i][j + 1].astype(int) + ((error * 3) / 8).astype(int), img[i + 1][j].astype(int) + ((error * 3) / 8).astype(int), img[i + 1][j + 1].astype(int) + (error / 4).astype(int)]
            for pixel in pixels:
                for k in range(3):
                    if pixel[k] > 255:
                        pixel[k] = 255
                    elif pixel[k] < 0:
                        pixel[k] = 0
            img[i][j + 1], img[i + 1][j], img[i + 1][j + 1] = pixels[0], pixels[1], pixels[2]

cv2.namedWindow('image2', cv2.WINDOW_NORMAL)
cv2.imshow('image2', medCutImg)

cv2.waitKey(0)

cv2.imwrite('img1_3_16.png', medCutImg)
cv2.destroyAllWindows()