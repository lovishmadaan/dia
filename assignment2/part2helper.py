import cv2
import numpy as np

def correct_colours(im1, im2):
    blur_amount = 101
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)


    # Avoid divide-by-zero errors.
    im2_blur = im2_blur + 128 * (im2_blur <= 1.0)


    return (im2 * (im1_blur / im2_blur))

im2 = np.float64(cv2.imread('swap.jpg'))
im1 = np.float64(cv2.imread('im2.jpg'))

mask = np.float64(cv2.imread('mask.jpg'))
mask = cv2.GaussianBlur(mask, (11, 11), 0) / 255


final = np.clip(correct_colours(im1, im2), 0.0, 255.0)

final = final * mask + (1 - mask) * im1
final = np.uint8(final)

# out = np.float64(cv2.imread('out.jpg'))
# outf = out * mask + (1 - mask) * im1

# cv2.imwrite('outf.jpg', outf)
cv2.imwrite('final.jpg', final)