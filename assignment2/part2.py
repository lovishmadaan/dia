import cv2
import dlib
import numpy as np
from scipy.spatial import Delaunay
from scipy import sparse
from scipy.sparse import linalg

def seamlessCloning (src, dest, mask):
    src = cv2.imread(src)
    dest = cv2.imread(dest)
    mask = cv2.imread(mask)

   
    maskToCd = []
    CdToId = -1 * np.ones(src.shape[:2], dtype=int)

    
    interior = []  # left, right, top, botton
    idx = 0

    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            if mask[i, j, 0] == 255:
                maskToCd.append([i, j])
                interior.append([
                    i > 0 and mask[i - 1, j, 0] == 255,
                    i < src.shape[0] - 1 and mask[i + 1, j, 0] == 255,
                    j > 0 and mask[i, j - 1, 0] == 255,
                    j < src.shape[1] - 1 and mask[i, j + 1, 0] == 255
                ])
                CdToId[i][j] = idx
                idx += 1

    N = idx
    b = np.zeros((N, 3))
    A = sparse.dok_matrix((N, N), dtype=int)

    for i in range(N):
        # for every pixel in interior and boundary
        A[i, i] = 4
        x, y = maskToCd[i]
        if interior[i][0]:
            A[i, int(CdToId[x - 1, y])] = -1
        if interior[i][1]:
            A[i, int(CdToId[x + 1, y])] = -1
        if interior[i][2]:
            A[i, int(CdToId[x, y - 1])] = -1
        if interior[i][3]:
            A[i, int(CdToId[x, y + 1])] = -1

    for i in range(N):
        flag = 1 - np.array(interior[i], dtype=int)
        x, y = maskToCd[i]
        for j in range(3):
            b[i, j] = 4 * src[x, y, j] - src[x - 1, y, j] - \
                src[x + 1, y, j] - src[x, y - 1, j] - src[x, y + 1, j]
            b[i, j] += flag[0] * dest[x - 1, y, j] + \
                flag[1] * dest[x + 1, y, j] + flag[2] * \
                dest[x, y - 1, j] + \
                flag[3] * dest[x, y + 1, j]

    A = A.tolil()

    x_r = linalg.cg(A, b[:, 0])[0]
    x_g = linalg.cg(A, b[:, 1])[0]
    x_b = linalg.cg(A, b[:, 2])[0]

    newImage = np.array(dest, dtype=np.uint8)

    for i in range(b.shape[0]):
        x, y = maskToCd[i]
        newImage[x, y, 0] = np.clip(x_r[i], 0, 255)
        newImage[x, y, 1] = np.clip(x_g[i], 0, 255)
        newImage[x, y, 2] = np.clip(x_b[i], 0, 255)

    return newImage

def add_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        im, points = param
        cv2.circle(im, (x, y), 4, (0, 0, 255), -1)
        points.append((y, x))

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
        cv2.circle(img, (point[1], point[0]), 4, (0, 0, 255), -1)

def barycentric(triangle, x):
    tri = triangle.astype(np.int64)
    a, b, c = tri[0, :], tri[1, :], tri[2, :]
    v0, v1, v2 = b - a, c - a, x - a
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom;
    w = (d00 * d21 - d01 * d20) / denom;
    u = 1.0 - v - w;
    return np.round(np.array([u, v, w]), 5)

def bounding_box(points):
    colmin = np.amin(points, axis=0)
    colmax = np.amax(points, axis=0)
    return np.array([[colmin[0], colmin[1]], [colmax[0], colmax[1]]], dtype=int)

def draw_box(im, box):
    cv2.line(im, (box[0][1], box[0][0]), (box[0][1], box[1][0]), (255, 0, 0), 1)
    cv2.line(im, (box[0][1], box[0][0]), (box[1][1], box[0][0]), (255, 0, 0), 1)
    cv2.line(im, (box[1][1], box[1][0]), (box[1][1], box[0][0]), (255, 0, 0), 1)
    cv2.line(im, (box[1][1], box[1][0]), (box[0][1], box[1][0]), (255, 0, 0), 1)


def read_im_and_landmarks(fname):
    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
    s = compute_features(im)

    return im, s

# Paste im1 face in im2
def swap_face(im1, landmarks1, im2, landmarks2, box2):
    hull = cv2.convexHull(landmarks2, returnPoints=False)[:, 0]
    points1 = landmarks1[hull]
    points2 = landmarks2[hull]
    
    im_swap = np.array(im2)
    tri = Delaunay(points2).simplices

    for i in range(box2[0][0], box2[1][0] + 1):
        for j in range(box2[0][1], box2[1][1] + 1):
            mp = np.array([i, j])
            tr = -1
            for t in range(len(tri)):
                bar = barycentric(points2[tri[t]], mp)
                if (bar >= 0.0).all() and (bar <= 1.0).all():
                    tr = t
                    break

            if tr >= 0:
                p = (bar @ points1[tri[tr]]).astype(int)
                im_swap[mp[0], mp[1]] = im1[p[0], p[1]]

    return im_swap

def get_face_mask(shape, landmarks):
    points = np.array(landmarks)
    points[:, 0], points[:, 1] = landmarks[:, 1], landmarks[:, 0]
    im = np.zeros(shape[:2], dtype=np.uint8)
    hull = cv2.convexHull(points)
    cv2.fillConvexPoly(im, hull, color=255)
    im = np.array([im, im, im]).transpose((1, 2, 0))

    return im

SCALE_FACTOR = 0.5

LEFT_EYE_POINTS = np.array(list(range(42, 48)), dtype=int)
RIGHT_EYE_POINTS = np.array(list(range(36, 42)), dtype=int)

im1, landmarks1 = read_im_and_landmarks('ted_cruz.jpg')
im2, landmarks2 = read_im_and_landmarks('donald_trump.jpg')

im1_copy = np.array(im1)
im2_copy = np.array(im2)

cv2.namedWindow('image1')
cv2.namedWindow('image2')
cv2.moveWindow('image1', 0, 0)
cv2.moveWindow('image2', im1.shape[1] + 100, 0)

cv2.setMouseCallback('image1', add_points, param=(im1_copy, landmarks1))
cv2.setMouseCallback('image2', add_points, param=(im2_copy, landmarks2))

draw_points(im1_copy, landmarks1)
draw_points(im2_copy, landmarks2)

while True:
    cv2.imshow('image1', im1_copy)
    cv2.imshow('image2', im2_copy)
    key = cv2.waitKey(20)

    if key & 0xFF == 27 or key & 0xFF == 13:
        break

cv2.destroyAllWindows()

landmarks1, landmarks2 = np.array(landmarks1), np.array(landmarks2)
box1, box2 = bounding_box(landmarks1), bounding_box(landmarks2)

draw_box(im1_copy, box1)
draw_box(im2_copy, box2)

while True:
    cv2.imshow('image1', im1_copy)
    cv2.imshow('image2', im2_copy)
    key = cv2.waitKey(20)

    if key & 0xFF == 27 or key & 0xFF == 13:
        break

cv2.destroyAllWindows()

im2_swap = swap_face(im1, landmarks1, im2, landmarks2, box2)

mask = get_face_mask(im2.shape, landmarks2)

center = np.mean(cv2.convexHull(landmarks2)[:, 0], axis=0)

cv2.imwrite('swap.jpg', im2_swap)
cv2.imwrite('mask.jpg', mask)
cv2.imwrite('im2.jpg', im2)
cv2.imwrite('im1.jpg', im1)

output = seamlessCloning('swap.jpg', 'im2.jpg', 'mask.jpg')

mask = cv2.GaussianBlur(np.float64(mask), (11, 11), 0) / 255
output = np.uint8(mask * output + (1 - mask) * im2)

cv2.imwrite('out.jpg', output)

while True:
    cv2.imshow('image1', im1_copy)
    cv2.imshow('image2', im2_copy)
    cv2.imshow('swap', im2_swap)
    cv2.imshow('out', output)
    key = cv2.waitKey(20)

    if key & 0xFF == 27 or key & 0xFF == 13:
        break

cv2.destroyAllWindows()