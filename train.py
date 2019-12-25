from model import Deeplabv3
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import tensorflow as tf
import keras
import os
import argparse
import keras.backend.tensorflow_backend as KTF
from data import dataset_builder
from utils import metric_utils,loss_utils


def parse_args():
    parser = argparse.ArgumentParser()
    # gpu config
    parser.add_argument('--gpu_devices', type=str, default="0",
                        help='select gpus, use in "CUDA_VISIBLE_DEVICES"')
    # data config
    parser.add_argument('--dataset_name', type=str, default='pascal',
                        help='classic dataset name, ["pascal","cityscape"]')

    # model config
    parser.add_argument('--model_type', type=str, default="", help='')
    parser.add_argument('--backend_type', type=str, default="", help='')
    parser.add_argument('--backbone', type=str, default='Xception',
                        help="backbone,['Xception', 'MobileNetv2']")
    parser.add_argument('--pre_weight', type=str, default='pascal_voc',
                        help="['pascal_voc','cityscapes']")
    parser.add_argument('--num_classes', type=int, default=21,
                        help='num of classes, base on dataset')

    # training config
    # for compile
    parser.add_argument('--loss_func', type=str, default='sparse_keras',
                        help="loss function,['categorical','categorical_mask','sparse_tf','sparse_keras']")
    parser.add_argument('--ignore_label', type=int, default=-1,
                        help='calculate loss ignore which label')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='optimizer,[Adam,momentum]')
    parser.add_argument('--metrics', type=str, default='mIoU',
                        help="metrics,['mIoU','categorical_accuracy']")
    parser.add_argument('--base_lr',
                        type=float, default=7e-4,
                        help='base learning rate,e.g 1e-3 for Adam')
    # for fit
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size, default=64')
    parser.add_argument('--epochs', type=int, default=50,
                        help='epochs, default=20')

    # callbacks config
    parser.add_argument('--chpt_dir', type=str, default='/ssd/maxiaowen/deep_lab_v3/check_points',
                        help='where saved the checkpoints')
    parser.add_argument('--tblog_dir', type=str, default='/ssd/maxiaowen/deep_lab_v3/tensorboard_log',
                        help='where saved the tensorboard log')
    parser.add_argument('--save_model_dir', type=str, default='/ssd/maxiaowen/deep_lab_v3/save_model',
                        help='where saved the trained model')

    return parser.parse_args()


def _get_model_dir_name(args):
    model_dir_name = 'model-{}-{}-lr_{}-{}'.format(
            args.dataset_name,  # dataset
            args.loss_func,
            args.optimizer,
            args.base_lr,)
    return model_dir_name


def train_model(args):
    # deeplab_model = Deeplabv3(weights=args.pre_weight)
    # deeplab_model = Deeplabv3(weights=None)
    deeplab_model = Deeplabv3(weights='pascal_voc')
    deeplab_model.summary()
    x_dir = "/ssd/maxiaowen/deep_lab_v3/dataset/VOC/input"
    y_dir = "/ssd/maxiaowen/deep_lab_v3/dataset/VOC/truth"
    loss_func = args.loss_func
    ignore_label = args.ignore_label
    base_lr = args.base_lr
    optimizer = args.optimizer
    if base_lr!=1e-3:
        optimizer = keras.optimizers.Adam(lr=base_lr, epsilon=1e-8, decay=1e-6)
    if loss_func=='categorical':
        x_train, x_test, y_train, y_test = dataset_builder.build_dataset(x_dir, y_dir)
        if ignore_label<0: # means no ignore label
            deeplab_model.compile(optimizer, loss=args.loss_func,
                                  metrics=[metric_utils.categorical_accuracy_without_background])
        else:
            deeplab_model.compile(optimizer, loss=loss_utils.categorical_crossentropy_without_background,
                                  metrics=[metric_utils.categorical_accuracy_without_background])
    else :
        x_train, x_test, y_train, y_test = dataset_builder.build_dataset(x_dir, y_dir, to_categorical=False)
        if loss_func=='sparse_tf':
            deeplab_model.compile(optimizer,loss=loss_utils.sparse_cross_entropy_loss,metrics=[metric_utils.mean_iou])
        else:
            deeplab_model.compile(optimizer, loss='sparse_categorical_crossentropy',
                                  metrics=[metric_utils.mean_iou])

    model_dir_name = _get_model_dir_name(args)
    print('finish compile')
    chpt_dir = args.chpt_dir
    tblog_dir = args.tblog_dir
    callbacks = []
    import use_callbacks
    checkpoint = use_callbacks.build_checkpoint(chpt_dir,model_dir_name)
    callbacks.append(checkpoint)
    tensorboard = use_callbacks.build_tensorboard(tblog_dir,model_dir_name)
    callbacks.append(tensorboard)

    print('begin fit')
    deeplab_model.fit(x_train, y_train,
              batch_size=args.batch_size,
              epochs=args.epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=callbacks)
    scores = deeplab_model.evaluate(x_test, y_test, verbose=1)
    print('loss:')
    print(scores[0])
    print('accu')
    print(scores[1])

    if not os.path.isdir(args.save_model_dir):
        os.makedirs(args.save_model_dir)
    print('save model in: ' + args.save_model_dir)
    model_file_name = 'model_' + str(args.loss_func)+str(ignore_label)+'_epoch'+str(args.epochs)+'.h'
    model_filepath = os.path.join(args.save_model_dir, model_file_name)
    deeplab_model.save(model_filepath)


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
    train_model(args)