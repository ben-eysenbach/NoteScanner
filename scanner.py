import cv2
import numpy as np
import math
import sys
import os
import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.ndimage.filters import uniform_filter

def threshold(img):
    '''Converts the given image to greyscale

    TODO:
    * only find mask0
    * try multiple k values
    * compare with Otsu's Method of binarization. (Implemented in my QR project)

    '''

    # Cluster colors into 2 groups
    colors = img.reshape(img.size / 3, 3)
    k_means = KMeans(2)
    k_means.fit(colors)
    width, height, depth = img.shape
    labels = k_means.labels_
    mask0 = (labels == 0).reshape((width, height))
    mask1 = (labels == 1).reshape((width, height))
    sorted_masks = sorted([mask0, mask1], key=np.sum, reverse=True)
    notes_mask = sorted_masks[0]
    # background_mask = sorted_masks[1]



    # color1 = np.mean(colors[mask1], axis=0)
    # color2 = np.mean(colors[mask2], axis=0)
    # print color1, color2
    # print img[mask1]


    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(notes_mask)
    plt.show()

    return notes_mask.astype('uint8')
    # return gray.astype('uint8')

def get_corners(img):
    '''Finds the bounding rectangle of the most likely rectangle
    in the given greyscale image, and returns its corners'''
    gray = threshold(img)
    contours, hierachy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_contour = sorted(contours, key=lambda c:c.size, reverse=True)[0]
    approx = cv2.approxPolyDP(best_contour, 0.01*cv2.arcLength(best_contour, True), True)
    # print 'Length:', len(approx)


    print 'Got %d contours' % len(contours)
    # print 'C1:', contours[0]
    # print 'Shape:', contours[0].shape
    # size = [c.size for c in contours]
    # plt.hist(size, bins=20)
    # plt.show()
    red = cv2.cv.CV_RGB(255,0,0)
    cv2.drawContours(img, [best_contour], -1, red, 2)
    # cv2.drawContours(img, [approx], -1, red, 2)
    plt.imshow(img)
    plt.show()
    # cv2.imshow('Frame', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    corners = approx.flatten().reshape(len(approx), 2)
    sorted_corners = sorted(corners, key=np.sum)
    print 'Corners:', sorted_corners
    return sorted_corners

def get_shape(corners):
    '''Returns the shape of the image to be extracted'''
    tl, tr, bl, br = corners
    top = tr - tl
    print 'top', top
    right = tr - br
    print 'right', right
    bottom = br - bl
    print 'bottom', bottom
    left = tl - bl
    print 'left', left
    top_dist = np.sqrt(np.dot(top, top))
    right_dist = np.sqrt(np.dot(right, right))
    bottom_dist = np.sqrt(np.dot(bottom, bottom))
    left_dist = np.sqrt(np.dot(left, left))

    rows = int((left_dist + right_dist) / 2.)
    cols = int((top_dist + bottom_dist) / 2.)
    print 'Rows:', rows
    print 'Cols:', cols
    return rows, cols


def crop_to_rect(img, corners):
    '''Extracts the portion of the given color image within the bounding
    rectangle specified by corners


    TODO:
    * correct size
    * filename
    '''
    assert len(corners) == 4
    rows, cols = get_shape(corners)


    pts1 = np.float32(corners)
    pts2 = np.float32([[0,0],[rows,0],[0,cols],[rows,cols]])

    M = cv2.getPerspectiveTransform(pts1,pts2)

    # Shape as col, rows

    dst = cv2.warpPerspective(img, M, (rows, cols))
    cv2.imwrite('output.jpg', dst)
    plt.imshow(dst)
    plt.show()


def get_image(filename):
    '''Returns the image with the given filename'''
    return cv2.imread(filename)[::10,::10,:].astype('uint8')

if __name__ == '__main__':
    img = get_image('rmb1.jpg')
    img = uniform_filter(img, size=10)
    gray = threshold(img)
    corners = get_corners(img)
    crop_to_rect(img, corners)
