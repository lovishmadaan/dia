import cv2
import numpy as np
from scipy.ndimage.filters import *

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def dog(img, k=1.6, sigma=1, tau=1):
    img1 = gaussian_filter(img, sigma)
    img2 = gaussian_filter(img, k * sigma)
    return (img1 - tau * img2)

def xdog(img, sigma=0.1, k=1.6, tau=0.9, epsilon=-5.0, phi=2):
    aux = dog(img, sigma=sigma, k=k, tau=tau) / 255
    aux = aux * img
    for i in range(aux.shape[0]):
        for j in range(aux.shape[1]):
            if(aux[i,j] >= epsilon):
                aux[i,j] = 255
            else:
                aux[i,j] = 255 * (1 + sigmoid(phi * (aux[i][j] - epsilon)))
    return aux

img = cv2.imread('img1_3_16.png', 1)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', img)

cv2.waitKey(0)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cart = np.uint8(xdog(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), sigma=1.0, k=1.6, tau = 0.998, epsilon=0.01, phi=10))
# cart = median_filter(cart, 3)
cv2.namedWindow('image2', cv2.WINDOW_NORMAL)
cv2.imshow('image2', cart)

pastel = np.array(hsv)

pastel[:, :, 2] = np.minimum(cart, pastel[:, :, 2])

pastel = cv2.cvtColor(pastel, cv2.COLOR_HSV2BGR)


cv2.namedWindow('image3', cv2.WINDOW_NORMAL)
cv2.imshow('image3', pastel)

cv2.waitKey(0)

cv2.imwrite('xdogsig3_16.png', cart)
cv2.imwrite('pastelsig3_16.png', pastel)
cv2.destroyAllWindows()