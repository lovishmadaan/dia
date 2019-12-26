import cv2
import numpy as np
import dlib

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

# mouse callback function
def add_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(im, (x, y), 2, (0, 0, 255), -1)
        im_points.append((y, x))

# Read images
im = cv2.imread('images/mustache.png')
img = cv2.imread('images/hillary_clinton.jpg')
points = compute_features(img)
draw_points(img, points)

print(im.shape)
cv2.namedWindow('image')

cv2.setMouseCallback('image', add_points)

im_points = []

while True:
    cv2.imshow('image', im)
    cv2.imshow('org', img)
    key = cv2.waitKey(20)

    if key & 0xFF == 27 or key & 0xFF == 13:
        break

cv2.destroyAllWindows()
print(im_points)