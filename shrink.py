import numpy as np
import Image
import sys
def shrink(filename, factor=10):
    '''Shrinks a given image by a given factor'''
    img = Image.open(filename)
    np_img = np.asarray(img)[::factor, ::factor, :]
    small_img = Image.fromarray(np_img)
    base, ending = filename.split('.')
    small_img.save(base + '_small.' + ending)

if __name__ == '__main__':
    filename = sys.argv[1]
    if len(sys.argv) > 2:
        factor = int(sys.argv[2])
    else:
        factor = 10
    shrink(filename, factor)
