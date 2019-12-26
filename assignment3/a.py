import numpy as np
import cv2
from quant import quantize

def reduce(im):
	im_blur = cv2.GaussianBlur(im, (5,5), 0)
	return im_blur[::2, ::2]

def expand(im, shape):
	im_exp = np.zeros(shape)
	im_exp[::2, ::2] = im
	return 4 * cv2.GaussianBlur(im_exp, (5, 5), 0)

def gaussPyr(im):
    img = np.array(im, dtype=np.float32)
    pyr = []
    while img.shape[0] >= 10 and img.shape[1] >= 10:
        pyr.append(img)
        img = reduce(img)

    return pyr

def lapPyr(im):
    gauss_pyr = gaussPyr(im)
    pyr = []
    for i in range(len(gauss_pyr) - 1):
        g_im = gauss_pyr[i]
        pyr.append(g_im - expand(gauss_pyr[i + 1], g_im.shape))

    pyr.append(gauss_pyr[-1])
    return pyr

def mergeLapPyr(pyr):
    img = np.array(pyr[-1])
    for i in range(len(pyr) - 2, -1, -1):
        img = expand(img, pyr[i].shape)
        img += pyr[i]
    
    return np.uint8(np.clip(img, 0, 255))

def binning(im, bins):
    im_bin = np.float32(np.clip(im, bins[0], bins[-1]))
    inds = np.digitize(im, bins)

    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            im_bin[i, j] = (bins[inds[i, j] - 1] + bins[inds[i, j]]) // 2

    return im_bin

def compress(im, bin_count, strat='CONST'):
    pyr = lapPyr(im)
    imgs = []
    nbins = bin_count
    for im in pyr[:-1]:
        quant_im = binning(im, np.linspace(-128, 128, nbins))
        imgs.append(quant_im)
        if strat == 'DOUBLE':
            nbins = nbins * 2

    imgs.append(pyr[-1])
    return mergeLapPyr(imgs)

def compress_medCut(im, num_color=16):
    pyr = lapPyr(im)
    imgs = []
    for im in pyr[:-1]:
        im_p = np.uint8(np.clip(im + 128, 0, 255))
        quant_im = np.float32(quantize(im_p, 16))
        imgs.append(quant_im - 128)

    imgs.append(pyr[-1])
    return mergeLapPyr(imgs)


def mosaic(img1, img2):
    pyr1, pyr2 = lapPyr(img1), lapPyr(img2)
    mosaic_pyr = []
    for im1, im2 in zip(pyr1, pyr2):
        rows, cols = im1.shape[0], im2.shape[1]
        mosaic_pyr.append(np.hstack((im1[:, 0 : cols // 2], im2[:, cols // 2 : ])))

    return mergeLapPyr(mosaic_pyr)

## Laplace Pyramid
# filename = "a.png"
# im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

# pyr = lapPyr(im)
# print(len(pyr))

# # for i in range(len(pyr)):
# #     cv2.imshow(str(i), np.uint8(pyr[i]))
# #     cv2.waitKey(0)
        
# # cv2.destroyAllWindows()

# mig = mergeLapPyr(pyr)
# cv2.imshow("merge", mig)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Compress
im = cv2.imread('b.png')
# comp = compress(im, 16, strat='DOUBLE')
comp = compress_medCut(im, 16)
q_im = quantize(im, 16)

diff1 = im - comp
diff2 = im - q_im

print(np.sqrt(np.sum(diff1 * diff1)) * 100 / (im.shape[0] * im.shape[1]))
# print(np.sqrt(np.sum(diff2 * diff2)) * 100 / (im.shape[0] * im.shape[1]))
cv2.imwrite('bcompressed.png', comp)
cv2.imwrite('bquantized.png', q_im)
cv2.imshow('Original', im)
cv2.imshow('Compressed', comp)
cv2.imshow('Quantized', q_im)
cv2.waitKey(0)
cv2.destroyAllWindows()


# # Mosaic
# im1 = cv2.imread('apple.jpg')
# im2 = cv2.imread('orange.jpg')

# rows, cols = im1.shape[0], im1.shape[1]
# trivial = np.hstack((im1[:, 0 : cols // 2], im2[:, cols // 2 : ]))

# cv2.imshow('Trivial', trivial)
# cv2.imshow('Mosaic', mosaic(im1, im2))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
