import tensorflow as tf
import keras
def categorical_crossentropy_without_background(y_true, y_pred,ignore=0,return_mask=False):
    print('do customized loss')
    print('y_true')
    print(y_true)
    print('y_pred')
    print(y_pred)
    equal_info = tf.equal(tf.argmax(y_true, -1), ignore)
    mask = tf.where(equal_info, tf.zeros_like(equal_info, dtype=tf.float32, name='zeros_like'),
                    tf.ones_like(equal_info, dtype=tf.float32, name='ones_like'))
    # print('mask')
    # print(mask)
    # # y_true_masked = y_true[np.where(np.argmax(y_true, -1) != ignore)]
    # # y_true_masked = tf.gather_nd(y_true, tf.where(tf.argmax(y_true, -1) != ignore))
    # mask = tf.where(tf.not_equal(tf.argmax(y_true, -1), ignore))
    # y_true_masked = tf.gather_nd(y_true, mask)
    # # y_pred_masked = y_pred[np.where(np.argmax(y_pred, -1) != ignore)]
    # # y_pred_masked = tf.gather_nd(y_pred, tf.where(tf.argmax(y_pred, -1) != ignore))
    # # y_pred_masked = tf.gather_nd(y_pred, tf.where(tf.equal(tf.argmax(y_pred, -1), ignore)))
    # y_pred_masked = tf.gather_nd(y_pred,mask)
    loss_raw = keras.losses.categorical_crossentropy(y_true, y_pred)
    loss_mask = loss_raw * mask
    if return_mask:
        return loss_mask,mask
    else:
        return loss_mask

def sparse_cross_entropy_loss(y_true, y_pred, num_classes=21):
    logits = y_pred
    labels = tf.cast(y_true, tf.int32)

    # base results
    labels = tf.squeeze(labels, axis=-1)  # reduce the channel dimension.
    logits_by_num_classes = tf.reshape(logits, [-1, num_classes])
    labels_flat = tf.reshape(labels, [-1, ])
    valid_indices = tf.cast(labels_flat <= num_classes - 1, tf.int32)
    valid_logits = tf.dynamic_partition(
        logits_by_num_classes, valid_indices, num_partitions=2)[1]
    valid_labels = tf.dynamic_partition(
        labels_flat, valid_indices, num_partitions=2)[1]

    # get loss
    cross_entropy = tf.compat.v1.losses.sparse_softmax_cross_entropy(
        logits=valid_logits, labels=valid_labels)
    return cross_entropy