import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as misc
import os
import glob
import os
import glob
import shutil
from scipy import io as scipy_io
# from skimage import io as skimage_io

def label_to_rgb_Voc(label_mask, plot=False):
    """
    功能：
        标签（类）转RGB，用于显示
    参数:
        label_mask (np.ndarray): (M,N)维度的含类别信息的矩阵.
        plot (bool, optional): 是否绘制图例.

    结果:
        (np.ndarray, optional): 解码后的色彩图.
    """
    label_colours = get_pascal_labels()
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, 21):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb

def get_pascal_labels():
    """
    Pascal VOC各类别对应的色彩标签

    结果:
        (21, 3)矩阵，含各类别对应的三通道数值信息
    """
    return np.asarray(
        [
            [0, 0, 0],
            [128, 0, 0],
            [0, 128, 0],
            [128, 128, 0],
            [0, 0, 128],
            [128, 0, 128],
            [0, 128, 128],
            [128, 128, 128],
            [64, 0, 0],
            [192, 0, 0],
            [64, 128, 0],
            [192, 128, 0],
            [64, 0, 128],
            [192, 0, 128],
            [64, 128, 128],
            [192, 128, 128],
            [0, 64, 0],
            [128, 64, 0],
            [0, 192, 0],
            [128, 192, 0],
            [0, 64, 128],
        ]
    )

def rbg_to_label_Voc(mask):
    """
    功能：
        将label转换为对应的类别信息
    参数:
        mask (np.ndarray): 原始label信息.
    返回值:
        (np.ndarray): 含色彩信息的label.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(get_pascal_labels()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask

def prepare_img_infolder():
    # RGB图路径,其中只有部分用于分割。这部分中分为训练集、验证集
    raw_all_input_path = "/ssd/zhangyiyang/data/VOCdevkit/VOC2012/JPEGImages"
    segimg_id_file = "/ssd/zhangyiyang/data/VOCdevkit/VOC2012/ImageSets/Segmentation/trainval.txt"
    # 从所有原始图像中抽取用于分割的图像
    file = open(segimg_id_file)
    lines = file.read()
    segimg_id_lst = lines.split('\n')
    raw_seg_input_dir = "/ssd/maxiaowen/deep_lab_v3/dataset/VOC/input"
    RGB_label_image_dir = "/ssd/zhangyiyang/data/VOCdevkit/VOC2012/SegmentationClass"
    Y_label_image_dir = "/ssd/maxiaowen/deep_lab_v3/dataset/VOC/truth"
    # 创建路径
    if os.path.exists(raw_seg_input_dir) is not True:
        os.mkdir(raw_seg_input_dir)
    if os.path.exists(Y_label_image_dir) is not True:
        os.mkdir(Y_label_image_dir)
    raw_img_names = glob.glob(os.path.join(raw_all_input_path, "*.jpg"))
    for filename_index, single_pascal_filename in enumerate(raw_img_names):
        _filename = single_pascal_filename.split('/')[-1].split('.')[0]
        if _filename in segimg_id_lst:
            shutil.copy(single_pascal_filename, raw_seg_input_dir)
    # 遍历RGB图并转为灰度格式
    for _filename in os.listdir(RGB_label_image_dir):
        lbl_path = os.path.join(RGB_label_image_dir, _filename)
        lbl = rbg_to_label_Voc(misc.imread(lbl_path))
        lbl = misc.toimage(lbl, high=lbl.max(), low=lbl.min())
        misc.imsave(os.path.join(Y_label_image_dir, _filename), lbl)
    print('convert RGB to labelY:Done!')

    # # 遍历灰度图并解码
    # for img in os.listdir(aug_path):
    #     img_path = os.path.join(aug_path, img)
    #     img1 = misc.imread(img_path)
    #     decoded = decode_segmap(img1)
    #     misc.imsave(os.path.join(out_path, img), decoded)
    # print("Done!")
    # # 遍历RGB图并转为灰度格式
    # for ii in os.listdir(aug_path):
    #     lbl_path = os.path.join(aug_path, ii)
    #     lbl = decode_segmap(misc.imread(lbl_path))
    #     lbl = misc.toimage(lbl, high=lbl.max(), low=lbl.min())
    #     misc.imsave(os.path.join(out_path, ii), lbl)
    # print('Done!')

# def max_min_normalize(data):
#     # 数据预处理：像素值max-min标准化
#     data = np.array(data, dtype="float") / 255.0
#     return data
#
# def subtract_pixel_mean(data):
#     # 均值标准化
#     print('test')
#     return data



if __name__=="__main__":
    prepare_img_infolder()