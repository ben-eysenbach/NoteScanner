
'''
To do:

* use a single matrix multiplication instead of a new one for each pixel

'''


import matplotlib.pyplot as plt
import numpy as np
import Image
import math
import Tkinter
import ImageTk
from corners import BaseImage

def transform(img, matrix, height=300, width=300):
    '''Applies the image transform given by the 3x3 matrix
    to the given image, and returns another image

    Uses bilinear interpolation
    '''
    assert type(img) == np.ndarray
    assert type(matrix) == np.matrix
    assert np.linalg.det(matrix) != 0
    inv_matrix = matrix.I
    # print 'Inv:', inv_matrix
    new_img = np.empty((height, width, 3), dtype='uint8')
    # for i, row in enumerate(new_img):
    #     for j, cell in enumerate(row):
    for r in range(height):
        for c in range(width):
            pt = [c, r]
            # cell = get_pt(pt, img, inv_matrix)
            new_img[r, c] = get_pt(pt, img, inv_matrix)
            # new_img[r, c] = img[r, c]
    return new_img

def get_pt(pt, img, inv_matrix):
    '''Finds the corresponding pixel in the original image

    Pt should be a tuple/list of the form (x, y)

    Images are indexed (row, col), where row increases from top to bottom'''
    # print 'Pt:',pt
    matrix_pt = np.matrix(list(pt) + [1]).T
    exact_pt = inv_matrix * matrix_pt
    x_shift, y_shift, _ = exact_pt % 1
    floor_x, floor_y, _ = np.floor(exact_pt).astype(int)
    ceil_x, ceil_y, _ = np.ceil(exact_pt).astype(int)
    try:
        ul_pix = img[floor_y, floor_x]
    except:
        ul_pix = np.array([0,0,0])

    try:
        ur_pix = img[floor_y, ceil_x]
    except:
        ur_pix = np.array([0,0,0])

    try:
        ll_pix = img[ceil_y, floor_x]
    except:
        ll_pix = np.array([0,0,0])

    try:
        lr_pix = img[ceil_y, ceil_x]
    except:
        lr_pix = np.array([0,0,0])



    return (1. - x_shift) * (1. - y_shift) * ul_pix + \
           (x_shift) * (1. - y_shift) * ur_pix + \
           (1. - x_shift) * (y_shift) * ll_pix + \
           (x_shift) * (y_shift) * lr_pix

def get_perspective_matrix(ul, ur, ll, lr, height=300, width=300):
    '''Takes 4 points as (x,y) tuples/lists and returns the
    corresponding matrix, M.
    Letting R denote the matrix of corners of the transformed image and
    X denote the corners on the untransformed image:

    M * X = R
    M * X * X.T = R * X.T
    M = (R * X.T) * (X * X.T).I

    '''

    R = [[0, width, 0, width],
         [0, 0, height, height],
         [1, 1, 1, 1]]

    X = np.matrix([ul, ur, ll, lr]).T
    X = np.vstack([X, np.ones((1,4))])


    return (R * X.T) * (X * X.T).I



if __name__ == '__main__':
    # img = np.asarray(Image.open('flag.jpg'))[::5, ::5, :]
    img = Image.open('rmb1_small.jpg')
    base_image = BaseImage(img)
    corners = base_image.corners
    matrix = get_perspective_matrix(*corners)
    # print matrix
    # matrix = 2 * np.matrix(np.identity(3))
    # print get_perspective_matrix([0,0], [200,0], [100,100], [300,100])
    new_img = transform(np.asarray(img), matrix)
    plt.imshow(new_img)
    plt.show()
