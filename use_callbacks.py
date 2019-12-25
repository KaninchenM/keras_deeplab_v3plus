# coding=utf-8
import keras
import os

def build_history():
    # use callback.history
    history = keras.callbacks.History()
    return history

def build_checkpoint(save_checkpoint_dir="",file_name="",monitor='val_acc',save_best_only=False):
    # use callback.checkpoint
    if save_checkpoint_dir=='':
        save_checkpoint_dir = os.path.join(os.getcwd(), 'check_points')
    if not os.path.isdir(save_checkpoint_dir):
        os.makedirs(save_checkpoint_dir)
    print('save checkpoint in: ' + save_checkpoint_dir)
    if file_name=='':
        file_name = 'model-{epoch:02d}-{loss:.4f}.h5'
    checkpoint_filepath = os.path.join(save_checkpoint_dir, file_name)
    checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_filepath, monitor=monitor, verbose=1,
                                                 save_best_only=save_best_only, save_weights_only=False, mode='auto', period=1)
    return checkpoint

def build_tensorboard(save_tensorboard_log_dir='',file_name=''):
    # use tensorboard
    if save_tensorboard_log_dir=='':
        save_tensorboard_log_dir = os.path.join(os.getcwd(), 'tensorboard_log')
    if not os.path.isdir(save_tensorboard_log_dir):
        os.makedirs(save_tensorboard_log_dir)
    print('save tensorboard log in: ' + save_tensorboard_log_dir)
    if file_name=='':
        file_name = 'model-{epoch:02d}-{loss:.4f}.h5'
    tensorboardLog_filepath = os.path.join(save_tensorboard_log_dir, file_name)
    print(tensorboardLog_filepath)
    tensorboard = keras.callbacks.TensorBoard(log_dir=tensorboardLog_filepath)
    return tensorboard

def build_earlystopping(monitor='val_acc', min_delta=0, patience=5):
    # use callback earlystop
    earlystop = keras.callbacks.EarlyStopping(monitor=monitor, min_delta=min_delta, patience=patience, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
    return earlystop


def build_classic_callbacks(use_history=True,use_checkpoint=True,use_tensorboard=True,use_earlystop=False):
    callbacks = []
    if use_history:
        history = build_history()
        callbacks.append(history)
    if use_checkpoint:
        checkpoint = build_checkpoint()
        callbacks.append(checkpoint)
    if use_tensorboard:
        tensorboard = build_tensorboard()
        callbacks.append(tensorboard)
    if use_earlystop:
        earlystop = build_earlystopping()
        callbacks.append(earlystop)

    return callbacks

