import cv2
import dlib
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull
from scipy.spatial import distance as dist

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

def compute_features(img):
    predictor_path = "predictor.dat"

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    points = []
    dets = detector(img, 1)
    shape = predictor(img, dets[0])
    for i in range(0, 68):
        points.append([shape.part(i).y, shape.part(i).x])

    return points

def draw_points(img, points):
    for point in points:
        cv2.circle(img, (point[1], point[0]), 2, (0, 0, 255), -1)

def bounding_box(points):
    colmin = np.amin(points, axis=0)
    colmax = np.amax(points, axis=0)
    return np.array([[colmin[0], colmin[1]], [colmax[0], colmax[1]]], dtype=int)

# mouse callback function
def add_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(im, (x, y), 2, (0, 0, 255), -1)
        im_points.append((y, x))

def eye_mask(img):
    imgEye = cv2.imread('images/eye_mask.png', -1)

    leftEye = points[LEFT_EYE_POINTS]
    rightEye = points[RIGHT_EYE_POINTS]
    leftCenter = np.mean(leftEye, axis=0).astype(int)
    rightCenter = np.mean(rightEye, axis=0).astype(int)
    leftCenter, rightCenter = np.array(rightCenter), np.array(leftCenter)
    noseCenter = np.mean(points[NOSE_POINTS], axis=0).astype(int)

    t1 = []
    t1.append([leftCenter[1], leftCenter[0]])
    t1.append([rightCenter[1], rightCenter[0]])
    t1.append([noseCenter[1], noseCenter[0]])

    #eye mask
    p1 = np.array([[277, 223], [439, 418]])
    p2 = np.array([[278, 755], [439, 559]])
    p3 = np.array([557, 492])

    c1 = np.mean(p1, axis=0).astype(int)
    c2 = np.mean(p2, axis=0).astype(int)
    t2 = []
    t2.append([c1[1], c1[0]])
    t2.append([c2[1], c2[0]])
    t2.append([p3[1], p3[0]])

    warpMat = cv2.getAffineTransform(np.float32(t2), np.float32(t1))
    dst = np.uint8(cv2.warpAffine(imgEye, warpMat, (img.shape[1], img.shape[0])))

    cv2.imshow('dst', dst)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    alpha = (dst[:, :, 3 : 4]).astype(float) / 255.0
    maskReshaped = dst[:, :, 0 : 3]

    out = np.uint8(alpha * maskReshaped + (1 - alpha) * img)
    cv2.imshow('eye_mask', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return out

def moustache(img):
    imgEye = cv2.imread('images/moustache.png', -1)

    upper = np.array(points[33])
    lower = np.array(points[51])
    left = np.mean(points[[3, 50]], axis=0).astype(int)
    right = np.mean(points[[13, 52]], axis=0).astype(int)

    t1 = []
    t1.append([upper[1], upper[0]])
    t1.append([lower[1], lower[0]])
    # t1.append([right[1], right[0]])
    t1.append([left[1], left[0]])

    p1 = np.array([57, 320])
    p2 = np.array([161, 318])
    p3 = np.array([129, 11])
    p4 = np.array([130, 627])
    t2 = []
    t2.append([p1[1], p1[0]])
    t2.append([p2[1], p2[0]])
    # t2.append([p4[1], p4[0]])
    t2.append([p3[1], p3[0]])

    warpMat = cv2.getAffineTransform(np.float32(t2), np.float32(t1))
    dst = np.uint8(cv2.warpAffine(imgEye, warpMat, (img.shape[1], img.shape[0])))

    # cv2.imshow('dst', dst)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    alpha = (dst[:, :, 3 : 4]).astype(float) / 255.0
    maskReshaped = dst[:, :, 0 : 3]

    out = np.uint8(alpha * maskReshaped + (1 - alpha) * img)
    # cv2.imshow('moustache', out)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return out


def jaws(img):
    imgEye = cv2.imread('images/jaws.png', -1)

    upper = np.array(points[33])
    lower = np.array(points[8]) + np.array([5, 0])
    left = np.array(points[1]) + np.array([0, -5])
    right = np.array(points[15]) + np.array([0, 5])

    t1 = []
    t1.append([upper[1], upper[0]])
    t1.append([lower[1], lower[0]])
    t1.append([left[1], left[0]])
    t1.append([right[1], right[0]])

    #jaws mask
    p1 = np.array([224, 208])
    p2 = np.array([492, 218])
    p3 = np.array([159, 18])
    p4 = np.array([155, 391])
    t2 = []
    t2.append([p1[1], p1[0]])
    t2.append([p2[1], p2[0]])
    t2.append([p3[1], p3[0]])
    t2.append([p4[1], p4[0]])

    warpMat = cv2.getPerspectiveTransform(np.float32(t2), np.float32(t1))
    dst = np.uint8(cv2.warpPerspective(imgEye, warpMat, (img.shape[1], img.shape[0])))

    cv2.imshow('dst', dst)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    alpha = (dst[:, :, 3 : 4]).astype(float) / 255.0
    maskReshaped = dst[:, :, 0 : 3]

    out = np.uint8(alpha * maskReshaped + (1 - alpha) * img)
    cv2.imshow('jaws', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return out

fileName = 'images/c3.jpg'
img = cv2.imread(fileName)

img_copy = np.array(img)

points = np.array(compute_features(img))

draw_points(img_copy, points)
# cv2.imshow('img', img_copy)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

out = moustache(img)
cv2.imwrite('images/muchan.jpg', out)
out = eye_mask(img)
cv2.imwrite('images/eye_mask.jpg', out)
out = jaws(img)
cv2.imwrite('images/jaws.jpg', out)
