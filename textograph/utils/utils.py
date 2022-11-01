import cv2
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np

def resize(img, dims):
    return cv2.resize(img, dims)
def imread(uri):
    return iio.imread(uri)
    
def plot_images_with_titles(images, titles, divide_by_255 = True, figsize=(10,10)):
    
    cols = min(8, len(images))
    rows = (len(images) // cols) + 1
    
    fig, ax = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    for i, (image, title) in enumerate(zip(images, titles)):
        row = i // cols
        col = i % cols
        if divide_by_255:
            image = image/255.
        ax[row,col].imshow(image)
        ax[row,col].set_title(title)
        ax[row,col].axis('off')

    plt.show()


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) 

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]