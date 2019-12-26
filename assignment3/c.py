import numpy as np
import cv2
from skimage import color
from sklearn.feature_extraction.image import extract_patches_2d
import nmslib

kappa = 2

def transform(A, A_prime, B):
    meu_a, meu_b = np.mean(A), np.mean(B)
    sigma_a, sigma_b = np.std(A), np.std(B)

    return (sigma_b / sigma_a) * (A - meu_a) + meu_b, (sigma_b / sigma_a) * (A_prime - meu_a) + meu_b, B


def get_gaussian_kernel(size, sigma):
    x, y = np.mgrid[-size // 2 + 1 : size // 2 + 1, -size // 2 + 1 : size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()

def get_weights(num_ch, last=False):
    g_sm = np.zeros((3, 3, num_ch))
    g_lg = np.zeros((5, 5, num_ch))

    gauss_sm = get_gaussian_kernel(3, 0.5)
    gauss_lg = get_gaussian_kernel(5, 1)

    for i in range(num_ch):
        g_sm[:, :, i] = gauss_sm
        g_lg[:, :, i] = gauss_lg


    w_sm = (1 / (9 * num_ch)) * g_sm.flatten()
    w_lg = (1 / (25 * num_ch)) * g_lg.flatten()
    w_half = (1 / 12 * num_ch) * g_lg.flatten()[: 12 * num_ch]

    if last:
        return np.hstack([w_lg, w_half])

    return np.hstack([w_sm, w_lg, w_sm, w_half])

def gaussPyr(im, levels):
    img = np.array(im, dtype=np.float32)
    pyr = [img]
    for i in range(levels):
        img = cv2.pyrDown(img)
        if img.ndim == 2:
            pyr.append(img[:, :, np.newaxis])
        else:
            pyr.append(img)

    pyr.reverse()
    return pyr

def preprocess(A, A_prime, B):
    A = color.rgb2yiq(A)[:, :, 0: 1]
    A_prime = color.rgb2yiq(A_prime)[:, :, 0: 1]
    B = color.rgb2yiq(B)[:, :, 0: 1]

    return A, A_prime, B

def get_features(pyr, num_ch, half=False):
    features = []
    pyr_padl = np.pad(pyr[0], ((2, 2), (2, 2), (0, 0)), mode='symmetric')
    
    ft_alone = extract_patches_2d(pyr_padl, (5, 5))
    ft_alone = ft_alone.reshape((ft_alone.shape[0], -1))
    if half:
        ft_alone = ft_alone[:, : 12 * num_ch]

    features.append(ft_alone)

    for i in range(1, len(pyr)):
        row, col = pyr[i].shape[:2]
        pyr_padl = np.pad(pyr[i], ((2, 2), (2, 2), (0, 0)), mode='symmetric')
        pyr_pads = np.pad(pyr[i - 1], ((1, 1), (1, 1), (0, 0)), mode='symmetric')

        ft_l = extract_patches_2d(pyr_padl, (5, 5))
        ft_s = extract_patches_2d(pyr_pads, (3, 3))
        ft_l, ft_s = ft_l.reshape((ft_l.shape[0], -1)), ft_s.reshape((ft_s.shape[0], -1))
        num_ft = (9 + 25) * num_ch
        
        if half:
            ft_l = ft_l[:, : 12 * num_ch]
            num_ft = (9 + 12) * num_ch


        level_ft = np.zeros((row * col, num_ft))
        for x in range(pyr[i].shape[0]):
            for y in range(pyr[i].shape[1]):
                ft_l_xy = np.hstack((ft_s[x // 2 * int(np.ceil(col / 2)) + y // 2], ft_l[x * col + y]))
                level_ft[x * col + y] = ft_l_xy


        features.append(level_ft)

    return features

def coherence_match(data, point, loc, data_size, S):
    row, col = S.shape[:2]
    min_dist, min_i, min_j = float('inf'), -1, -1

    for i in range(max(0, loc[0] - 2), min(row, loc[0] + 3)):
        for j in range(max(0, loc[1] - 2), loc[1] + 1):

            if i == loc[0] and j == loc[1]:
                break

            m_i, m_j = S[i, j, 0] + (loc[0] - i), S[i, j, 1] + (loc[1] - j)

            if m_i >= 0 and m_i < data_size[0] and m_j >= 0 and m_j < data_size[1]:
                A_m_ft = data[m_i * data_size[1] + m_j]
                dist = np.linalg.norm(A_m_ft - point)
                if dist < min_dist:
                    min_dist = dist
                    min_i, min_j = m_i, m_j

    return min_i * data_size[1] + min_j

def image_analogy(A_im, A_prime_im, B_im, process=True):
    if process:
        A, A_prime, B = preprocess(A_im, A_prime_im, B_im)
        A, A_prime, B = transform(A, A_prime, B)
    else:
        A, A_prime, B = A_im, A_prime_im, B_im

    num_ch = A.shape[2]

    levels = int(np.log2(min(A.shape[0], A.shape[1]) / 32))

    A_pyr, A_prime_pyr, B_pyr = gaussPyr(A, levels), gaussPyr(A_prime, levels), gaussPyr(B, levels)

    A_features, B_features = get_features(A_pyr, num_ch), get_features(B_pyr, num_ch)
    A_prime_features = get_features(A_prime_pyr, num_ch, half=True)

    index_time_params = {'M': 15, 'indexThreadQty': 4, 'efConstruction': 100, 'post' : 0}
    query_time_params = {'efSearch': 100}

    weights, weights_last = get_weights(num_ch), get_weights(num_ch, last=True)
    k = kappa * (2 ** -levels)

    B_prime_pyr = []
    for i in range(levels + 1):
        row, col = B_pyr[i].shape[: 2]
        A_row, A_col = A_pyr[i].shape[: 2]
        B_prime_l_pad = np.zeros((row + 4, col + 4, num_ch))
        S = -1 * np.ones((row, col, 2), dtype=int)

        data = np.hstack((A_features[i], A_prime_features[i]))
        print(data.shape)

        index = nmslib.init(method='hnsw', space='l2', data_type=nmslib.DataType.DENSE_VECTOR) 
        index.addDataPointBatch(data)
        index.createIndex(index_time_params)
        index.setQueryTimeParams(query_time_params)
        w = weights_last

        if i != 0:
            B_prime_s_pad = np.pad(B_prime_pyr[i - 1], ((1, 1), (1, 1), (0, 0)), mode='symmetric')
            w = weights

        for x in range(row):
            for y in range(col):
                if i == 0:
                    B_prime_ftp_alone = B_prime_l_pad[x : x + 5, y : y + 5, :].flatten()[: 12 * num_ch]
                    B_ftp = np.hstack((B_features[i][x * col + y], B_prime_ftp_alone))
                else:
                    B_prime_ftp_l = B_prime_l_pad[x : x + 5, y : y + 5, :].flatten()[: 12 * num_ch]
                    B_prime_ftp_s = B_prime_s_pad[x // 2 : x // 2 + 3, y // 2 : y // 2 + 3, :].flatten()
                    B_ftp = np.hstack((B_features[i][x * col + y], B_prime_ftp_s, B_prime_ftp_l))

                ann = index.knnQuery(B_ftp, k=1)[0][0]
                coh = coherence_match(data, B_ftp, (x, y), (A_row, A_col), S)

                if coh < 0:
                    match_i, match_j = ann // A_col, ann % A_col

                else:
                    A_ftp_ann = data[ann]
                    A_ftp_coh = data[coh]

                    d_app = np.linalg.norm((A_ftp_ann - B_ftp) * w)
                    d_coh = np.linalg.norm((A_ftp_coh - B_ftp) * w)

                    if d_coh <= d_app * (1 + k):
                        match_i, match_j = coh // A_col, coh % A_col
                    else:
                        match_i, match_j = ann // A_col, ann % A_col

                B_prime_l_pad[x + 2, y + 2] = A_prime_pyr[i][match_i, match_j]
                S[x, y, 0] = match_i
                S[x, y, 1] = match_j

        B_prime_pyr.append(B_prime_l_pad[2 : 2 + row, 2 : 2 + col])
        k = k * 2
        print('Level', i, 'done')

    if process:
        B_im = color.rgb2yiq(B_im)
        B_yiq = np.dstack((B_prime_pyr[-1][:, :, 0], B_im[:, :, 1], B_im[:, :, 2]))
        return color.yiq2rgb(B_yiq)

    return np.uint8(B_prime_pyr[-1])


A = cv2.imread('watercolor-src.jpg')
A_prime = cv2.imread('watercolor.jpg')
B = cv2.imread('boats.jpg')

B_prime = image_analogy(A, A_prime, B, process=True)

cv2.imwrite('B_prime.jpg', B_prime)
while 2 > 1:
    cv2.imshow('A', A)
    cv2.imshow('A\'', A_prime)
    cv2.imshow('B', B)
    cv2.imshow('B\'', B_prime)
    k = cv2.waitKey(33)
    if k==27:    # Esc key to stop
        break
cv2.destroyAllWindows()
