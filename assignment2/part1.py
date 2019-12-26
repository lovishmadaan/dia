import cv2
import numpy as np
import dlib
from scipy.spatial import Delaunay

def compute_features():
    predictor_path = "predictor.dat"

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    for img, points in [(img1_orig, img1_points), (img2_orig, img2_points)]:
        dets = detector(img, 1)
        shape = predictor(img, dets[0])
        for i in range(0, 68):
            points.append([shape.part(i).y, shape.part(i).x])

def add_boundaries():
    for img, points in [(img1_orig, img1_points), (img2_orig, img2_points)]:
        points.append([0, 0])
        points.append([img.shape[0], 0])
        points.append([0, img.shape[1]])
        points.append([img.shape[0], img.shape[1]])
        points.append([img.shape[0] // 2, 0])
        points.append([img.shape[0] // 2, img.shape[1]])
        points.append([0, img.shape[1] // 2])
        points.append([img.shape[0], img.shape[1] // 2])

def draw_points():
    for img, points in [(img1_copy, img1_points), (img2_copy, img2_points)]:
        for point in points:
            cv2.circle(img, (point[1], point[0]), 4, (0, 0, 255), -1)

# mouse callback function
def add_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        if param == 'image1':
            cv2.circle(img1_copy, (x, y), 4, (0, 0, 255), -1)
            img1_points.append((y, x))
        else:
            cv2.circle(img2_copy, (x, y), 4, (0, 0, 255), -1)
            img2_points.append((y, x))

def barycentric(triangle, x):
    a, b, c = triangle[0, :], triangle[1, :], triangle[2, :]
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

filename1 = 'donald_trump.jpg'
filename2 = 'ted_cruz.jpg'

# Read images
img1_orig = cv2.imread(filename1)
img2_orig = cv2.imread(filename2)

img1_orig = cv2.resize(img1_orig, (0,0), fx=0.5, fy=0.5)
img2_orig = cv2.resize(img2_orig, (0,0), fx=0.5, fy=0.5)

img1_copy = np.array(img1_orig)
img2_copy = np.array(img2_orig)

cv2.namedWindow('image1')
cv2.namedWindow('image2')
cv2.moveWindow('image1', 0, 0)
cv2.moveWindow('image2', img1_orig.shape[1] + 100, 0)

cv2.setMouseCallback('image1', add_points, param='image1')
cv2.setMouseCallback('image2', add_points, param='image2')

img1_points = []
img2_points = []

compute_features()
add_boundaries()
draw_points()

while True:
    cv2.imshow('image1', img1_copy)
    cv2.imshow('image2', img2_copy)
    key = cv2.waitKey(20)

    if key & 0xFF == 27 or key & 0xFF == 13:
        break

cv2.destroyAllWindows()

if(len(img1_points) != len(img2_points)):
    print("Feature points have unequal count")
    exit()

img1_points = np.array(img1_points, dtype=int)
img2_points = np.array(img2_points, dtype=int)

for a in range(1, 10, 1):
    alpha = a * 0.1
    morph_points = np.array(img1_points.shape, dtype=int)
    morph_points = ((1 - alpha) * img1_points + alpha * img2_points).astype(int)

    tri = Delaunay(morph_points).simplices

    temp1, temp2 = np.array(img1_copy), np.array(img2_copy)

    for i in range(tri.shape[0]):
        for img, points in [(temp1, img1_points), (temp2, img2_points)]:
            cv2.line(img, (points[tri[i]][0][1], points[tri[i]][0][0]), (points[tri[i]][1][1], points[tri[i]][1][0]), (255, 0, 0), 1)
            cv2.line(img, (points[tri[i]][1][1], points[tri[i]][1][0]), (points[tri[i]][2][1], points[tri[i]][2][0]), (255, 0, 0), 1)
            cv2.line(img, (points[tri[i]][2][1], points[tri[i]][2][0]), (points[tri[i]][0][1], points[tri[i]][0][0]), (255, 0, 0), 1)

    # while True:
    #     cv2.imshow('image1', temp1)
    #     cv2.imshow('image2', temp2)
    #     key = cv2.waitKey(20)

    #     if key & 0xFF == 27 or key & 0xFF == 13:
    #         break

    # cv2.destroyAllWindows()



    morph_img = np.zeros(img1_orig.shape)

    for i in range(morph_img.shape[0]):
        for j in range(morph_img.shape[1]):
            mp = np.array([i, j])
            tr = 0
            for t in range(len(tri)):
                bar = barycentric(morph_points[tri[t]], mp)
                if (bar >= 0.0).all() and (bar <= 1.0).all():
                    tr = t
                    break

            p1 = (bar @ img1_points[tri[tr]]).astype(int)
            p2 = (bar @ img2_points[tri[tr]]).astype(int)
            morph_img[mp[0], mp[1]] = (1 - alpha) * img1_orig[p1[0], p1[1]] + alpha * img2_orig[p2[0], p2[1]]

    morph_img = morph_img.astype(np.uint8)


    # while True:
    #     cv2.imshow('morph', morph_img)
    #     key = cv2.waitKey(20)

    #     if key & 0xFF == 27 or key & 0xFF == 13:
    #         break

    cv2.imwrite('m'+str(alpha)[:3]+'.jpg', morph_img)
    # cv2.destroyAllWindows()
    print(alpha)




