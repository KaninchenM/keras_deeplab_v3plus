from model import Deeplabv3
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import tensorflow as tf
import os
import argparse
import keras.backend.tensorflow_backend as KTF
from data import dataset_builder


def parse_args():
    parser = argparse.ArgumentParser()
    # gpu config
    parser.add_argument('--gpu_devices', type=str, default="0",
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
    parser.add_argument('--metrics', type=str, default='categorical_accuracys',
                        help="metrics,['accuracy','categorical_accuracy']")
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




def validate_pre_model(args,pre_weights='pascal_voc',num_classes=21):
    deeplab_model = Deeplabv3(weights=pre_weights)
    deeplab_model.summary()
    x_dir = "/ssd/maxiaowen/deep_lab_v3/dataset/VOC/input"
    y_dir = "/ssd/maxiaowen/deep_lab_v3/dataset/VOC/truth"
    _, x_test, _, y_test = dataset_builder.build_dataset(x_dir, y_dir)

    # 原本打算用以下写法，但发现model接受的x_test和我给的样子不一样，而且作者明确说他没有做过Pascal Voc数据集的验证
    # 故自己写了IoU的计算+用的prediction
    # todo:注意，predict只需要加载有预训练权重的model；而evaluate因为是“评价”需要知道评价的指标，所以明确给出optimizer和loss_func的
    metrics = ['accuracy','binary_accuracy','categorical_accuracy']
    # metrics.append(args.metrics)
    deeplab_model.compile(args.optimizer, args.loss_func,metrics=metrics)
    scores = deeplab_model.evaluate(x_test, y_test, verbose=1)
    # print(scores)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    hist = np.zeros((num_classes, num_classes))
    for i in range(len(x_test)):
        _x = x_test[i] # x's shape:(512,512,3)
        _truth_y = y_test[i].argmax(axis=-1) # shape (512,512,21) to shape (512,512)
        # _truth_y = y_test[i]
        res = deeplab_model.predict(np.expand_dims(_x, 0))  # _x:shape (h,w,c) to shape (1,h,w,c); res's shape(1,512,512,21)
        # res = deeplab_model.predict(_x)
        _pred_labels = np.argmax(res.squeeze(), -1) # squeeze makes res's shape (1,512,512,21) to (512,512,21), and after argmax, shape becomes (512,512)
        # print('validating:'+str(i+1)+'/'+str(len(x_test)))
        if len(_truth_y.flatten()) != len(_pred_labels.flatten()):  # 如果图像分割结果与标签的大小不一样，这张图片就不计算
            continue
        else:
            hist += fast_hist(_truth_y.flatten(), _pred_labels.flatten(), num_classes)  # 对一张图片计算n*n的hist矩阵，并累加
    mIoUs = per_class_iu(hist)  # 计算所有验证集图片的逐类别mIoU值
    IoUs_without_background = 0.000
    import math
    count = 0
    for ind_class in range(num_classes):  # 逐类别输出一下mIoU值
        print('===> class ' + str(ind_class) + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
        if ind_class!=0 and (not math.isnan(mIoUs[ind_class])):
            IoUs_without_background = IoUs_without_background+mIoUs[ind_class]
            count = count+1
    # print('===> mIoU: ' + str(round(np.nanmean(mIoUs), 2)))  # 在所有验证集图像上求所有类别平均的mIoU值，计算时忽略NaN值
    print('===> mIoU(without background): ' + str(round(IoUs_without_background/count*100, 2))+'%')
    return mIoUs


def fast_hist(true_y, pred_y, n):
    '''
	n是类别数目
	'''
    k = (true_y >= 0) & (true_y < n) #k是一个一维bool数组，形状(H×W,)；目的是找出标签中需要计算的类别（去掉了背景即去掉label=0） k=0或1
    # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    return np.bincount(n * true_y[k].astype(int) + pred_y[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    '''
	分别为每个类别计算mIoU，hist的形状(n, n)
	'''
    # hist.sum(0)=按列相加  hist.sum(1)按行相加
    # 矩阵的对角线上的值组成的一维数组/矩阵的所有元素之和，返回值形状(n,)
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))




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
    validate_pre_model(args, pre_weights='pascal_voc', num_classes=21)