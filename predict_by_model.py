from model import Deeplabv3
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import tensorflow as tf
import os
import argparse
import keras
import keras.backend.tensorflow_backend as KTF
from data import dataset_builder


def parse_args():
    parser = argparse.ArgumentParser()
    # gpu config
    parser.add_argument('--gpu_devices', type=str, default="1",
                        help='select gpus, use in "CUDA_VISIBLE_DEVICES"')
    # data config
    parser.add_argument('--dataset_name', type=str, default='cifar10',
                        help='classic dataset name, ["cifar10","to be added"]')

    # model config
    parser.add_argument('--backbone', type=str, default='Xception',
                        help="backbone,['Xception', 'MobileNetv2']")
    parser.add_argument('--pre_weight', type=str, default='pascal_voc',
                        help="['pascal_voc','cityscapes']")
    parser.add_argument('--num_classes', type=int, default=21,
                        help='num of classes, base on dataset')

    # training config
    # for compile
    parser.add_argument('--loss_func', type=str, default='categorical_crossentropy',
                        help='loss function')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='optimizer,[Adam,momentum]')
    parser.add_argument('--metrics', type=str, default='accuracy',
                        help='metrics')
    # for fit
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size, default=64')
    parser.add_argument('--epochs', type=int, default=20,
                        help='epochs, default=20')

    # callbacks config
    parser.add_argument('--chpt_dir', type=str, default='/ssd/maxiaowen/deep_lab_v3/check_points',
                        help='where saved the checkpoints')
    parser.add_argument('--tblog_dir', type=str, default='/ssd/maxiaowen/deep_lab_v3/tensorboard_log',
                        help='where saved the tensorboard log')

    return parser.parse_args()

def predict_and_display(image_path,weights='pascal_voc'):
    # Generates labels using most basic setup.  Supports various image sizes.  Returns image labels in same format
    # as original image.  Normalization matches MobileNetV2
    trained_image_width = 512
    mean_subtraction_value = 127.5
    image = np.array(Image.open(image_path))

    # trained_image_width to max dimension of images from training dataset
    w, h, _ = image.shape
    ratio = float(trained_image_width) / np.max([w, h])
    resized_image = np.array(Image.fromarray(image.astype('uint8')).resize((int(ratio * h), int(ratio * w))))

    # apply normalization for trained dataset images
    resized_image = (resized_image / mean_subtraction_value) - 1.

    # pad array to square image to match training images
    pad_x = int(trained_image_width - resized_image.shape[0])
    pad_y = int(trained_image_width - resized_image.shape[1])
    resized_image = np.pad(resized_image, ((0, pad_x), (0, pad_y), (0, 0)), mode='constant')

    # make prediction
    deeplab_model = Deeplabv3(weights=weights)
    res = deeplab_model.predict(np.expand_dims(resized_image, 0))
    labels = np.argmax(res.squeeze(), -1)

    # remove padding and trained_image_width back to original image
    if pad_x > 0:
        labels = labels[:-pad_x]
    if pad_y > 0:
        labels = labels[:, :-pad_y]
    labels = np.array(Image.fromarray(labels.astype('uint8')).resize((h, w)))

    plt.imshow(labels)
    plt.waitforbuttonpress()


if __name__=="__main__":
    args = parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # 按照PCI_BUS_ID顺序从0开始排列GPU设备
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices  # GPU ID，可通过命令行命令 nvidia-smi 来查看当前
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # 仅使用CPU

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True  # 不一定全部占满显存, 按需分配
    tf_config.gpu_options.per_process_gpu_memory_fraction = 1  # 占用比例，如0.3 ; 代码或配置层面设置了对显存占用百分比阈值，但在实际运行中如果达到了这个阈值，程序有需要的话还是会突破这个阈值。换而言之如果跑在一个大数据集上还是会用到更多的显存。
    sess = tf.Session(config=tf_config)
    KTF.set_session(sess)

    image_path = '/ssd/zhangyiyang/data/VOCdevkit/VOC2012/JPEGImages/2011_003238.jpg'
    predict_and_display(image_path,args.pre_weight)