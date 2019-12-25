import random
from glob import glob
import numpy as np
import cv2
import tensorflow as tf

def build_dataset(x_dir,y_dir,to_categorical=True):
    # 读取数据和标签
    print("------read data------")
    xs = []
    ys = []

    # 拿到图像数据路径，方便后续读取
    x_img_list = glob('{}/*.jpg'.format(x_dir))
    x_img_list.sort()
    y_img_list = glob('{}/*.png'.format(y_dir))
    y_img_list.sort()
    rand_idx = np.arange(len(x_img_list))
    # np.random.shuffle(rand_idx)
    # 遍历读取数据
    count = 0
    for i_dx in rand_idx:
        x_path = x_img_list[i_dx]
        y_path = y_img_list[i_dx]
        # 读取图像数据
        x = cv2.imread(x_path)
        # x = cv2.resize(x, (512, 512))
        x = resize_input(x,do_subtract=True)
        xs.append(x)
        # 读取标签
        y = cv2.imread(y_path)
        # y = cv2.resize(y, (512, 512))
        y = resize_input(y, do_subtract=False)
        y = y.max(axis=2) # shape (h,w,3) to shape (h,w)
        # y = np.expand_dims(y,-1) # shape (h,w) to shape (h,w,1)
        if to_categorical:
            from keras.utils.np_utils import to_categorical
            y_categorical = to_categorical(y, num_classes=21)
            ys.append(y_categorical)
        else:
            y = np.expand_dims(y,-1)
            ys.append(y)

        # for debug
        # count = count+1
        # if count==320:
        #      break

    xs = np.array(xs)
    ys = np.array(ys,dtype=np.int)
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(xs, ys, test_size=0.3, random_state=42)
    print('finish read data')
    return x_train, x_test, y_train, y_test
    # return xs,xs,ys,ys


def resize_input(image, trained_image_width=512, do_subtract=False,mean_subtraction_value = 127.5):
    # trained_image_width to max dimension of images from training dataset
    w, h, _ = image.shape
    ratio = float(trained_image_width) / np.max([w, h])  # 按比例缩放，这样可以保持变化后图像的长宽比不变
    from PIL import Image
    resized_image = np.array(Image.fromarray(image.astype('uint8')).resize((int(ratio * h), int(ratio * w))))
    if do_subtract:
        # apply normalization for trained dataset images
        resized_image = (resized_image / mean_subtraction_value) - 1.

    # pad array to square image to match training images
    pad_x = int(trained_image_width - resized_image.shape[0])
    pad_y = int(trained_image_width - resized_image.shape[1])
    resized_image = np.pad(resized_image, ((0, pad_x), (0, pad_y), (0, 0)), mode='constant')
    return resized_image
