import tensorflow as tf
from keras import backend as K

##################################
# Losses
##################################
def rpn_loss_regr(num_anchors, lambda_rpn_regr = 1.0, epsilon = 1e-4):
    """Loss function for rpn regression
    Args:
        num_anchors: number of anchors (9 in here)
    Returns:
        Smooth L1 loss function 
            0.5*x*x (if x_abs < 1)
            x_abx - 0.5 (otherwise)
        lambda * sum(IsPositive * ((x_bool)(0.5*x*x) + (1 - x_bool)(x_abx-0.5)) ) / N
    """
    def rpn_loss_regr_fixed_num(y_true, y_pred):
        """
        y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=1), y_rpn_regr], axis=1) -> POS Labels + regression coordinates
        print(y_rpn_regr.shape) # (1, 37, 56, 72) y_true
        """
        # x is the difference between true value and predicted value
        if K.image_data_format() == "channels_last":
            x = y_true[:, :, :, 4 * num_anchors:] - y_pred
        else:
            x = y_true[:, 4 * num_anchors:, :, :] - y_pred

        # absolute value of x
        x_abs = K.abs(x)

        # If x_abs <= 1.0, x_bool = 1
        x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32)

        if K.image_data_format() == "channels_last":
            return lambda_rpn_regr * K.sum(y_true[:, :, :, :4 * num_anchors] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :, :4 * num_anchors])
        return lambda_rpn_regr * K.sum(y_true[:, :4 * num_anchors, :, :] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :4 * num_anchors, :, :])

    return rpn_loss_regr_fixed_num

def rpn_loss_cls(num_anchors, lambda_rpn_class = 1.0, epsilon = 1e-4):
    """Loss function for rpn classification
    Args:
        num_anchors: number of anchors (9 in here)
        y_true[:, :, :, :9] = y_is_box_valid: [0,1,0,0,0,0,0,1,0] means only the second and the eighth box is valid which contains pos or neg anchor => isValid
        y_true[:, :, :, 9:] = y_rpn_overlap: [0,1,0,0,0,0,0,0,0] means the second box is pos and eighth box is negative 
    Returns:
        lambda * sum((binary_crossentropy(isValid*y_pred,y_true))) / N # ignore neutral
    """
    def rpn_loss_cls_fixed_num(y_true, y_pred):
        """
        y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=1)
        print(y_rpn_cls.shape) # (1, 37, 56, 18)
        """
        if K.image_data_format() == "channels_last":
            return lambda_rpn_class * K.sum(y_true[:, :, :, :num_anchors] * K.binary_crossentropy(y_true[:, :, :, num_anchors:], y_pred[:, :, :, :])) / K.sum(epsilon + y_true[:, :, :, :num_anchors])
        
        # different shape (1, 18, 37, 56)
        return lambda_rpn_class * K.sum(y_true[:, :num_anchors, :, :] * K.binary_crossentropy(y_true[:, num_anchors:, :, :], y_pred[:, :, :, :])) / K.sum(epsilon + y_true[:, :num_anchors, :, :])

    return rpn_loss_cls_fixed_num

def class_loss_regr(num_classes, lambda_cls_regr = 1.0, epsilon = 1e-4):
    def class_loss_regr_fixed_num(y_true, y_pred):
        x = y_true[:, :, 4*num_classes:] - y_pred
        x_abs = K.abs(x)
        x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
        return lambda_cls_regr * K.sum(y_true[:, :, :4*num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :4*num_classes])
    return class_loss_regr_fixed_num

def class_loss_cls(lambda_cls_class = 1.0):
    def class_loss_cls_fix_num(y_true, y_pred):
        return lambda_cls_class * K.mean(K.categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))
    return class_loss_cls_fix_num

