import tensorflow as tf
import keras

def categorical_accuracy_without_background(y_true, y_pred,ignore=0):
    print('do customized accuracy')
    equal_info = tf.equal(tf.argmax(y_true, -1), ignore)
    mask = tf.where(equal_info, tf.zeros_like(equal_info, dtype=tf.float32, name='zeros_like'),
                    tf.ones_like(equal_info, dtype=tf.float32, name='ones_like'))
    # # y_true_masked = y_true[np.where(np.argmax(y_true, -1) != ignore)]
    # # y_true_masked = tf.gather_nd(y_true, tf.where(tf.argmax(y_true, -1) != ignore))
    # mask = tf.where(tf.not_equal(tf.argmax(y_true, -1), ignore))
    # y_true_masked = tf.gather_nd(y_true, mask)
    # # y_pred_masked = y_pred[np.where(np.argmax(y_pred, -1) != ignore)]
    # # y_pred_masked = tf.gather_nd(y_pred, tf.where(tf.argmax(y_pred, -1) != ignore))
    # # y_pred_masked = tf.gather_nd(y_pred, tf.where(tf.equal(tf.argmax(y_pred, -1), ignore)))
    # y_pred_masked = tf.gather_nd(y_pred, mask)
    accuracy_raw = keras.metrics.categorical_accuracy(y_true, y_pred)
    accuracy_mask = accuracy_raw * mask
    return accuracy_mask

def mean_iou(y_true, y_pred, num_classes=21):
    logits = y_pred
    labels = tf.cast(y_true, tf.int32)

    pred_classes = tf.expand_dims(
        tf.argmax(logits, axis=3, output_type=tf.int32), axis=3)
    labels = tf.squeeze(labels, axis=-1)  # reduce the channel dimension.
    labels_flat = tf.reshape(labels, [-1, ])
    valid_indices = tf.cast(labels_flat <= num_classes - 1, tf.int32)
    valid_labels = tf.dynamic_partition(
        labels_flat, valid_indices, num_partitions=2)[1]
    preds_flat = tf.reshape(pred_classes, [-1, ])
    valid_preds = tf.dynamic_partition(
        preds_flat, valid_indices, num_partitions=2)[1]

    cm = tf.math.confusion_matrix(
        valid_labels, valid_preds, num_classes
    )

    def compute_mean_iou(total_cm):
        sum_over_row = tf.cast(
            tf.reduce_sum(total_cm, 0), tf.float32)
        sum_over_col = tf.cast(
            tf.reduce_sum(total_cm, 1), tf.float32)
        cm_diag = tf.cast(tf.diag_part(total_cm), tf.float32)
        denominator = sum_over_row + sum_over_col - cm_diag

        # The mean is only computed over classes that appear in the
        # label or prediction tensor. If the denominator is 0, we need to
        # ignore the class.
        num_valid_entries = tf.reduce_sum(
            tf.cast(
                tf.not_equal(denominator, 0), dtype=tf.float32))

        # If the value of the denominator is 0, set it to 1 to avoid
        # zero division.
        denominator = tf.where(
            tf.greater(denominator, 0), denominator,
            tf.ones_like(denominator))
        iou = tf.div(cm_diag, denominator)

        # If the number of valid entries is 0 (no classes) we return 0.
        result = tf.where(
            tf.greater(num_valid_entries, 0),
            tf.reduce_sum(iou, name='mean_iou') / num_valid_entries, 0)
        return result

    score = compute_mean_iou(cm)

    return score