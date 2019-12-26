import numpy as np
import cv2
import heapq

KNOWN = 0
INSIDE = 1
BAND = 2

INF = 1.0e6
EPS = 1.0e-6

def solve(i1, j1, i2, j2, T, f):
    row, col = f.shape[0], f.shape[1]
    if i1 < 0 or i1 >= row or j1 < 0 or j1 >= col:
        return INF

    if i2 < 0 or i2 >= row or j2 < 0 or j2 >= col:
        return INF

    sol = INF
    if f[i1, j1] == KNOWN:
        if f[i2, j2] == KNOWN:
            d1, d2 = T[i1, j1], T[i2, j2]
            d = 2.0 - (d1 - d2) * (d1 - d2)
            if d > 0.0:
                r = np.sqrt(d)
                s = (d1 + d2 - r) / 2.0
                if s >= d1 and s >= d2:
                    sol = s
                else:
                    s += r
                    if s >= d1 and s > d2:
                        sol = s
        else:
            sol = 1.0 + T[i1, j1]
    elif f[i2, j2] == KNOWN:
        sol = 1.0 + T[i2, j2] 
    
    return sol

def gradient(i, j, T, f):
    row, col = f.shape[0], f.shape[1]
    d = T[i, j]
    grad = np.zeros((2))
    
    if i - 1 < 0 or i + 1 >= row:
        grad[0] = INF
    else:
        if f[i - 1, j] != INSIDE and f[i + 1, j] != INSIDE:
            grad[0] = (T[i + 1, j] - T[i - 1, j]) / 2.0
        elif f[i - 1, j] != INSIDE:
            grad[0] = T[i, j] - T[i - 1, j]
        elif f[i + 1, j] != INSIDE:
            grad[0] = T[i + 1, j] - T[i, j]

    if j - 1 < 0 or j + 1 >= col:
        grad[1] = INF
    else:
        if f[i, j - 1] != INSIDE and f[i, j + 1] != INSIDE:
            grad[1] = (T[i, j + 1] - T[i, j - 1]) / 2.0
        elif f[i, j - 1] != INSIDE:
            grad[1] = T[i, j] - T[i, j - 1]
        elif f[i, j + 1] != INSIDE:
            grad[1] = T[i, j + 1] - T[i, j]

    return grad

def initialize(mask, epsilon):
    row, col = mask.shape[0], mask.shape[1]
    
    T = -1.0 * INF * np.float32(mask)
    f = np.uint8(mask)
    narrow_band = []

    mask_x, mask_y = np.nonzero(mask)
    for i, j in zip(mask_x, mask_y):
        neighbors = [(i - 1, j), (i, j - 1), (i + 1, j), (i, j + 1)]
        for k, l in neighbors:
            if k < 0 or k >= row or l < 0 or l >= col:
                continue

            if f[k, l] == BAND:
                continue

            if mask[k, l] == 0:
                f[k, l] = BAND
                T[k, l] = 0.0
                heapq.heappush(narrow_band, (0.0, k, l))

    f_inv = np.array(f)
    f_inv[f == KNOWN] = INSIDE
    f_inv[f == INSIDE] = KNOWN
    out_band, max_dist = narrow_band.copy(), 0.0

    while max_dist <= epsilon and out_band:
        _, i, j = heapq.heappop(out_band)
        f_inv[i, j] = KNOWN

        neighbors = [(i - 1, j), (i, j - 1), (i + 1, j), (i, j + 1)]
        for k, l in neighbors:
            if k < 0 or k >= row or l < 0 or l >= col:
                continue

            if f_inv[k, l] == INSIDE:
                f_inv[k, l] = BAND
                T[k, l] = min([solve(k - 1, l, k, l - 1, T, f_inv),
                            solve(k + 1, l, k, l + 1, T, f_inv),
                            solve(k - 1, l, k, l + 1, T, f_inv),
                            solve(k + 1, l, k, l - 1, T, f_inv)])

                max_dist = T[k, l]
                heapq.heappush(out_band, (T[k, l], k, l))


    im = np.zeros(f.shape, dtype=np.uint8)
    im[f == BAND] = 255

    cv2.imshow('Band', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return -1.0 * T, f, narrow_band

def inpaint(i, j, img, T, f, epsilon):
    row, col = f.shape[0], f.shape[1]
    start_i, end_i = max(0, i - epsilon), min(row, i + epsilon + 1)
    start_j, end_j = max(0, j - epsilon), min(col, j + epsilon + 1)

    grad = gradient(i, j, T, f)
    Ia, s = np.zeros((3)), 0.0

    for k in range(start_i, end_i):
        for l in range(start_j, end_j):
            r = np.array([i - k, j - l])
            length_r = np.sqrt(np.dot(r, r))
            if length_r <= epsilon and length_r > 0.0 and f[k, l] != INSIDE:
                dirf = abs(np.dot(r, grad)) / length_r
                if dirf == 0.0:
                    dirf = EPS

                dst = 1.0 / (length_r * length_r)
                
                lev = 1.0 / (1.0 + abs(T[k, l] - T[i, j]))

                w = dirf * dst * lev
                gradI = np.zeros((2, 3))

                # if f[k - 1, l] == KNOWN and f[k + 1, l] == KNOWN:
                #     gradI[0, :] = (img[k + 1, l] - img[k - 1, l]) / 2

                # if f[k, l - l] == KNOWN and f[k, l + 1] == KNOWN:
                #     gradI[1, :] = (img[k, l + 1] - img[k, l - 1]) / 2 

                Ia += w * (np.clip(img[k, l] + np.dot(r, gradI), 0, 255))
                s += w

    img[i, j] = Ia / s

def inpaint_image(img, mask, epsilon=5):
    row, col = img.shape[0], img.shape[1]
    T, f, narrow_band = initialize(mask, epsilon)

    while narrow_band:
        _, i, j = heapq.heappop(narrow_band)
        f[i, j] = KNOWN

        neighbors = [(i - 1, j), (i, j - 1), (i + 1, j), (i, j + 1)]
        for k, l in neighbors:
            if k < 0 or k >= row or l < 0 or l >= col:
                continue

            if f[k, l] != KNOWN:
                T[k, l] = min([solve(k - 1, l, k, l - 1, T, f),
                        solve(k + 1, l, k, l + 1, T, f),
                        solve(k - 1, l, k, l + 1, T, f),
                        solve(k + 1, l, k, l - 1, T, f)])
            
                if f[k, l] == INSIDE:
                    f[k, l] = BAND
                    inpaint(k, l, img, T, f, epsilon)

                heapq.heappush(narrow_band, (T[k, l], k, l))
            
im = cv2.imread('fruits_damaged.jpg')
mask = cv2.imread('fruits_mask.jpg', cv2.IMREAD_GRAYSCALE)
mask[mask > 0] = 1

img = np.array(im)
inpaint_image(img, mask)
cv2.imwrite('fruits_painted.png', img)
cv2.imshow('Orig', im)
cv2.imshow('Painted', img)

cv2.waitKey(0)
cv2.destroyAllWindows()