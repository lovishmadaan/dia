import cv2
import numpy as np

def display_zoom_in(event, x, y, flags, param):
    global img
    global orig_img
    global zoom_in_img_crop
    global zoom_in_img
    global strata_in
    img = np.copy(orig_img)

    if event == cv2.EVENT_LBUTTONDOWN:
        if strata_in == 'REPLICATION':
            strata_in = 'INTERPOLATION'
            zoom_in_img = zoom_in_img_bil
        else:
            strata_in = 'REPLICATION'
            zoom_in_img = zoom_in_img_rep
    if event == cv2.EVENT_MOUSEMOVE:
        if x+50 < img.shape[1] and x > 50 and y > 50 and y+50 < img.shape[0]:
            zoom_in_img_crop = zoom_in_img[5*y - 250 : 5*y + 250, 5*x - 250 : 5*x + 250]
        cv2.rectangle(img, (x-50, y-50), (x+50, y+50), (0, 0, 0))


def resize_im_in(orig_img, strategy='INTERPOLATION'):
    mat = np.ones((orig_img.shape[0]*5, orig_img.shape[1] * 5, 3), dtype=np.uint8)

    if strategy == 'REPLICATION':
        for row_num in range(orig_img.shape[0]):
            for col_num in range(orig_img.shape[1]):
                mat[row_num*5:row_num*5 + 5, col_num*5:col_num*5+5]=orig_img[row_num][col_num]

    if strategy == 'INTERPOLATION':
        print("Came here")
        for row_num in range(orig_img.shape[0] - 1):
            for col_num in range(orig_img.shape[1] - 1):
                i = np.array([[0], [1], [2], [3], [4]])
                j = np.array([[0], [1], [2], [3], [4]])
                p_i1 = np.dot((1 - i / 5.0), orig_img[row_num : row_num + 1, col_num, :]) + np.dot((i / 5.0), orig_img[row_num: row_num + 1, col_num + 1, :])
                p_i2 = np.dot((1 - i / 5.0), orig_img[row_num + 1: row_num + 2, col_num, :]) + np.dot((i / 5.0), orig_img[row_num + 1: row_num + 2, col_num + 1, :])
                for k in range(3):
                    p = np.dot((1 - j / 5.0), p_i1[:, k : k + 1].T) + np.dot((j / 5.0), p_i2[:, k : k + 1].T)
                    mat[row_num * 5: row_num * 5 + 5, col_num * 5 : col_num * 5 + 5, k] = np.uint8(p)

    return mat

def display_zoom_out(event, x, y, flags, param):
    global img
    global orig_img
    global zoom_out_img_crop
    global zoom_out_img
    global strata_out
    img = np.copy(orig_img)

    if event == cv2.EVENT_LBUTTONDOWN:
        print("Hi")
        if strata_out == 'REPLICATION':
            strata_out = 'INTERPOLATION'
            zoom_out_img = zoom_out_img_avg
        else:
            strata_out = 'REPLICATION'
            zoom_out_img = zoom_out_img_drop
    if event == cv2.EVENT_MOUSEMOVE:
        if x + 150 < img.shape[1] and x > 150 and y > 150 and y + 150 < img.shape[0]:
            pass
            zoom_out_img_crop = zoom_out_img[y // 3 - 50: y//3 + 50, x//3 - 50: x//3 + 50]
        cv2.rectangle(img, (x - 150, y - 150), (x + 150, y + 150), (0, 0, 0))

def resize_im_out(orig_img, strategy='REPLICATION'):
    reduceBy = 3
    mat = np.ones((orig_img.shape[0] // reduceBy, orig_img.shape[1] // reduceBy, 3), dtype=np.uint8)

    if strategy == 'REPLICATION':
        for row_num in range(mat.shape[0]):
            for col_num in range(mat.shape[1]):
                mat[row_num,col_num] = orig_img[reduceBy//2 + row_num*reduceBy][reduceBy//2 + col_num*reduceBy]

    if strategy == 'INTERPOLATION':
        temp_slice = orig_img[0:orig_img.shape[0] - (orig_img.shape[0]%3),0:orig_img.shape[1] - (orig_img.shape[1]%3),:]

        for row_num in range(mat.shape[0]):
            for col_num in range(mat.shape[1]):
                mat[row_num, col_num,0] = temp_slice[row_num*reduceBy : row_num*reduceBy+reduceBy,col_num*reduceBy : col_num*reduceBy+reduceBy,0].mean()
                mat[row_num, col_num, 1] = temp_slice[row_num * reduceBy: row_num * reduceBy + reduceBy,col_num * reduceBy: col_num * reduceBy + reduceBy, 1].mean()
                mat[row_num, col_num, 2] = temp_slice[row_num * reduceBy: row_num * reduceBy + reduceBy,col_num * reduceBy: col_num * reduceBy + reduceBy, 2].mean()

    return mat

orig_img = cv2.imread('lenna.png')
img = np.copy(orig_img)


zoomWin = cv2.namedWindow('zoom')
rootWin = cv2.namedWindow('image')
cv2.moveWindow('image', 0, 0)
cv2.moveWindow('zoom', img.shape[1], 0)

zoom_status = 0
cv2.setMouseCallback('image', display_zoom_in)

strata_in = 'REPLICATION'
strata_out = 'REPLICATION'

zoom_in_img_rep = resize_im_in(orig_img, 'REPLICATION')
zoom_in_img_bil = resize_im_in(orig_img, 'INTERPOLATION')
zoom_out_img_drop = resize_im_out(orig_img, 'REPLICATION')
zoom_out_img_avg = resize_im_out(orig_img, 'INTERPOLATION')
zoom_in_img = zoom_in_img_rep
zoom_out_img = zoom_out_img_drop
zoom_in_img_crop = np.zeros((500, 500, 3), dtype=np.uint8)
zoom_out_img_crop = np.zeros((100, 100, 3), dtype=np.uint8)

while True:
    cv2.imshow('image', img)
    if zoom_status == 0:
        cv2.imshow('zoom', zoom_in_img_crop)
    else:
        cv2.imshow('zoom', zoom_out_img_crop)
    key = cv2.waitKey(20)
    if key & 0xFF == 32:
        if zoom_status == 0:
            cv2.setMouseCallback('image', display_zoom_out)
        else :
            cv2.setMouseCallback('image', display_zoom_in)
        zoom_status = 1 - zoom_status

    if key & 0xFF == 27 or key & 0xFF == 13:
        break
    if key & 0xFF == ord('s'):
        cv2.imwrite('out_'+strata_in+'.jpg',zoom_in_img_crop)
        break
cv2.destroyAllWindows()