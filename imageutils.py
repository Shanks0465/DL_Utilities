import imutils
from matplotlib import image
from skimage.transform import resize, rescale
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import img_as_ubyte
import random

def shuffle(samples):
    shuffled=random.sample(samples, len(samples))
    return shuffled

def hr_lr_generator(samples,path,batch_size=10):
    num_samples = len(samples)
    while True:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            X_train = []
            y_train = []
            for sample_file in batch_samples:
                img_data = image.imread(path+sample_file)
                image_resized = resize(img_data, (256, 256))
                output = cv2.resize(image_resized, dsize)
                lr=cv2.resize(output, (256,256))
                X_train.append(img_as_ubyte(lr))
                y_train.append(img_as_ubyte(image_resized))
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            yield X_train, y_train

def create_mask(img):
  mask = np.full((256,256,3), 0, np.uint8)
  for _ in range(np.random.randint(1, 10)):
    x1, x2 = np.random.randint(1, 256), np.random.randint(1, 256)
    y1, y2 = np.random.randint(1, 256), np.random.randint(1, 256)
    thickness = 7
    cv2.line(mask,(x1,y1),(x2,y2),(255,255,255),thickness)
  masked_image = img.copy()
  image_resized = resize(masked_image, (256, 256))
  return img_as_ubyte(image_resized), mask


def inpaint_generator(samples,batch_size=10):
    num_samples = len(samples)
    while True:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            X_train = []
            y_train = []
            for sample_file in batch_samples:
                img_data = image.imread('/content/drive/My Drive/DIV2K_train_HR/'+sample_file)
                image_resized = resize(img_data, (256, 256))
                img,mask=create_mask(image_resized)
                masked=cv2.bitwise_or(img,mask)
                X_train.append(masked)
                y_train.append(img)
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            yield X_train, y_train

def get_iou(bb1, bb2):
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def sliding_window(image, step, ws):
    # slide a window across the image
    for y in range(0, image.shape[0] - ws[1], step):
        for x in range(0, image.shape[1] - ws[0], step):
            # yield the current window
            yield (x, y, image[y:y + ws[1], x:x + ws[0]])


def image_pyramid(image, scale=1.5,minSize=(224, 224)):
    # yield the original image
    yield image

    # keep looping over the image pyramid
    while True:
        # compute the dimensions of the next image in the pyramid
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)

        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break

        # yield the next image in the pyramid
        yield image
