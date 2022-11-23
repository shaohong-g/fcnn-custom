import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Conv2D, Input, TimeDistributed, Flatten, Dense
from keras.optimizers import Adam



##################################
# Models - base
##################################

def freeze_model(model, start=0, end=None):
    """Feeze models based on start:end (inclusive:exclusive) layers

    Parameters: 
    ----------
    model : object
        model object from tensorflow
    start : int, optional
        first layer's index to be frozen. (Default = 0)
    end : int, optional
        last layer's index to be frozen (excluding) (Default = None)
    """
    if end is None:
        end = len(model.layers)
    for x in model.layers[start:end]:
        x.trainable = False 


def partial_base_model(model_name = "VGG19", input_size = (None, None, 3), weights = "imagenet", start=0, end=None, preview = False):
    """Get pretrained model

    Parameters
    ----------
    model_name : string
        SOTA architecture name
    input_size : tuple, optional
        height, width, depth (Default = (None, None, 3) )
    weights : string, optional
        weight used in the pretrained model (Default = imagenet)
    start : int, optional
        first layer's index to be frozen. None if you do not want to freeze model (Default = 0)
    end : int, optional
        last layer's index to be frozen (excluding) (Default = None)
    preview : booleon, optional
        option to show model summary (Default = False)

    Returns
    -------
    base_model.layers[layer_idx].output , base_model.input

    Raises
    ------
    Exception
        No base model can be found!
    """
    model_input = Input(shape=input_size, name = "Img_input")
    
    layer_idx = -1
    if model_name == "VGG19":
        from keras.applications.vgg19 import VGG19

        base_model = VGG19(weights=weights, include_top=False, input_tensor=model_input)
        layer_idx = -2
    elif model_name == "VGG16":
        from keras.applications.vgg16 import VGG16
        base_model = VGG16(weights=weights, include_top=False, input_tensor=model_input)
        layer_idx = -2
    else:
        raise Exception("No base model can be found!")

    if start is not None:
        freeze_model(base_model, start, end)

    if preview: 
        # print(base_model.layers[0].get_config())
        base_model.summary()
        print(f"Freeze layers from {start} - {end}")
        print(base_model.input)
        print(base_model.layers[layer_idx].output)

    return base_model.layers[layer_idx].output , base_model.input # get the layer before max_pooling


##################################
# Models - RPN
##################################

def rpn_layer(base_layers, num_anchors, model_name = "VGG19"):
    """Concat rpn_layer to base_layer

    Parameters
    ----------
    base_layer : object
        last layer of the base_model
    num_anchors : int
        Number of anchors (anchor size X anchor ratio)

    Returns
    -------
    [x_class, x_regr, base_layers]
    """
    #cnn_used for creating feature maps: vgg, num_anchors: 9
    if model_name in ["VGG19", "VGG16"]:
        filters = 512
    else:
        raise Exception("Update filters in rpn_layer function!")
    x = Conv2D(filters, (3, 3), padding='same', activation='relu', name = "rpn_conv1")(base_layers)
    # print(x.shape)
    
    #classification layer: num_anchors (9) channels for 0, 1 sigmoid activation output
    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', name = "rpn_result_classification")(x)
    # print(x_class.shape)
    
    #regression layer: num_anchors*4 (36) channels for computing the regression of bboxes
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', name = "rpn_result_regression")(x)
    # print(x_regr.shape)
    return [x_class, x_regr, base_layers] #classification of object(0 or 1),compute bounding boxes, base layers vgg


##################################
# Models - Classifier
##################################
class RoiPoolingConv(tf.keras.layers.Layer):
    '''ROI pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    
    e.g. out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])
    # Arguments
        pool_size: int
            Size of pooling region to use. pool_size = 7 will result in a 7x7 region.
        num_rois: number of regions of interest to be used
    # Input shape
        list of two 4D tensors [X_img,X_roi] with shape:
        X_img:
        `(1, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(1, rows, cols, channels)` if dim_ordering='tf'.
        X_roi:
        `(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
    # Output shape
        3D tensor with shape:
        `(1, num_rois, pool_size, pool_size, channels)`
    '''
    def __init__(self, pool_size, num_rois, **kwargs):

        self.dim_ordering = K.image_data_format()
        assert self.dim_ordering == 'channels_last', 'dim_ordering must be "channels_last". K.common.image_dim_ordering() is depreciated - Keras v2'

        self.pool_size = pool_size
        self.num_rois = num_rois

        super(RoiPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.dim_ordering == 'channels_first':
            self.nb_channels = input_shape[0][1]
        elif self.dim_ordering == 'channels_last':
            self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        """Optional"""
        if self.dim_ordering == 'channels_first':
            return None, self.num_rois, self.nb_channels, self.pool_size, self.pool_size
        else:
            return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    def call(self, x):

        assert(len(x) == 2), 'There must be two inputs: [base_layers, input_rois] specified.'

        img = x[0]  # `(1, rows, cols, channels)` tf
        rois = x[1] # `(1,num_rois,4)` list of rois, with ordering (x,y,w,h)

        input_shape = K.shape(img) # tensor shape

        outputs = []

        for roi_idx in range(self.num_rois):

            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]
            
            num_pool_regions = self.pool_size

            #NOTE: the RoiPooling implementation differs between theano and tensorflow due to the lack of a resize op
            # in theano. The theano implementation is much less efficient and leads to long compile times

            if self.dim_ordering == 'channels_first':
                row_length = w / float(self.pool_size)
                col_length = h / float(self.pool_size)
                for jy in range(num_pool_regions):
                    for ix in range(num_pool_regions):
                        x1 = x + ix * row_length
                        x2 = x1 + row_length
                        y1 = y + jy * col_length
                        y2 = y1 + col_length

                        x1 = K.cast(x1, 'int32')
                        x2 = K.cast(x2, 'int32')
                        y1 = K.cast(y1, 'int32')
                        y2 = K.cast(y2, 'int32')

                        x2 = x1 + K.maximum(1,x2-x1)
                        y2 = y1 + K.maximum(1,y2-y1)
                        
                        new_shape = [input_shape[0], input_shape[1],
                                     y2 - y1, x2 - x1]

                        x_crop = img[:, :, y1:y2, x1:x2]
                        xm = K.reshape(x_crop, new_shape)
                        pooled_val = K.max(xm, axis=(2, 3))
                        outputs.append(pooled_val)

            elif self.dim_ordering == 'channels_last':
                x = K.cast(x, 'int32')
                y = K.cast(y, 'int32')
                w = K.cast(w, 'int32')
                h = K.cast(h, 'int32')

                rs = tf.image.resize(img[:, y:y+h, x:x+w, :], (self.pool_size, self.pool_size)) # fix pooling size for each ROIs (resize ROI (portion of the image) to fixed pool size)
                outputs.append(rs)

        final_output = K.concatenate(outputs, axis=0) # (number of ROIs, rows, cols, channels)
        final_output = K.reshape(final_output, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels)) # add another dimension (Image index) to final_output

        if self.dim_ordering == 'channels_first':
            final_output = K.permute_dimensions(final_output, (0, 1, 4, 2, 3))
        else:
            final_output = K.permute_dimensions(final_output, (0, 1, 2, 3, 4))

        return final_output
    
    
    def get_config(self):
        config = {'pool_size': self.pool_size,
                  'num_rois': self.num_rois}
        base_config = super(RoiPoolingConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def classifier_layer(base_layers, num_rois, nb_classes, input_rois_shape=(None, 4), pooling_regions = 7):
    """ROI Pooling and dense layers

    Parameters
    ----------
    base_layer : object
        last layer of the base_model
    num_rois : int
        Number of rois to be considered
    nb_classes : int
        Number of classes including BG
    input_rois_shape : tuple
        Shape of ROIs input which is transformed from output of RPN model (Default = (None, 4))
    pooling_regions : int
        pooling shape (x,x) (Default = 7)
    
    Returns
    -------
    [out_class, out_regr] : list
        shape=(1, num_rois, 2), shape=(1, num_rois, 4)
    """
    roi_input = Input(shape=input_rois_shape, name='ROI_Inputs')
    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, roi_input])

    out = TimeDistributed(Flatten(name='flatten'))(out_roi_pool)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc1'))(out)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc2'))(out)

    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out) # Classifier
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out) # regression coordinates

    return [out_class, out_regr] , roi_input

##################################
# Models - All
##################################

def show_save_model(model, save_plot = False, graphviz_bin_dir = r"C:/Program Files/Graphviz/bin/"):
    # model.summary()
    assert os.path.exists(graphviz_bin_dir), "Graphviz Bin file does not exists. (see instructions at https://graphviz.gitlab.io/download/)"
    if save_plot:
        # conda install graphviz
        # conda install pydot
        # conda install pydotplus
        from keras.utils import plot_model
        os.environ["PATH"] += os.pathsep + graphviz_bin_dir
        plot_model(model, to_file=f'{save_plot}', show_shapes=True)

def get_faster_rcnn_model(C = None, save_plot = False, base_SOTA = "VGG19", start = 0, end = None, training = True):
    """Get model

    Parameters
    ----------
    C : Config
        img_input_shape, num_anchors, num_rois, roi_input_shape, pooling_regions
    save_plot : string, optional
        Save model architecture (FINAL model) to save_plot.png (Default = False)
    start : int, optional
        first layer's index to be frozen. None if you do not want to freeze model (Default = 0)
    end : int, optional
        last layer's index to be frozen (excluding) (Default = None)
    training : Boolean, optional
        True if you want to train model (Default = True)

    Returns
    -------
    model_rpn, model_classifier, model_all
    """
    # Configuration
    assert hasattr(C, 'img_input_shape'), f"Configuration does not contain the attribute: img_input_shape"
    assert hasattr(C, 'num_anchors'), f"Configuration does not contain the attribute: num_anchors"
    assert hasattr(C, 'num_rois'), f"Configuration does not contain the attribute: num_rois"
    assert hasattr(C, 'roi_input_shape'), f"Configuration does not contain the attribute: roi_input_shape"
    assert hasattr(C, 'pooling_regions'), f"Configuration does not contain the attribute: pooling_regions"
    
    base_layer, base_input = partial_base_model(model_name=base_SOTA, input_size = C.img_input_shape, start = start, end = end)

    rpn = rpn_layer(base_layer, C.num_anchors)

    nb_classes = len(C.classes) # classes include BG
    current_folder = os.path.dirname(os.path.realpath(__file__))
    if training:
        classifier, roi_input = classifier_layer(base_layer, C.num_rois, nb_classes, input_rois_shape=C.roi_input_shape, pooling_regions = C.pooling_regions)
        model_rpn = Model(base_input, rpn[:2])
        model_classifier = Model([base_input, roi_input], classifier)

        model_all = Model([base_input, roi_input], rpn[:2] + classifier)

        if save_plot:
            os.makedirs(os.path.join(current_folder, base_SOTA), exist_ok=True)
            show_save_model(model_rpn, save_plot = os.path.join(current_folder, base_SOTA, "train_model_rpn.png"))
            show_save_model(model_classifier, save_plot = os.path.join(current_folder, base_SOTA, "train_model_classifier.png"))
            show_save_model(model_all, save_plot = os.path.join(current_folder, base_SOTA, "train_model_all.png"))

        return model_rpn, model_classifier, model_all

    else:
        if base_SOTA in ["VGG19", "VGG16"]:
            num_features = 512
        else:
            raise Exception(f"{base_SOTA} is not configured for num_features")
        classifier_input = Input(shape=(None, None, num_features), name="feature_map_input")
        classifier, roi_input = classifier_layer(classifier_input, C.num_rois, nb_classes, input_rois_shape=C.roi_input_shape, pooling_regions = C.pooling_regions)
        model_rpn = Model(base_input, rpn)
        model_classifier = Model([classifier_input, roi_input], classifier)

        if save_plot:
            os.makedirs(os.path.join(current_folder, base_SOTA), exist_ok=True)
            show_save_model(model_rpn, save_plot = os.path.join(current_folder, base_SOTA, "test_model_rpn.png"))
            show_save_model(model_classifier, save_plot = os.path.join(current_folder, base_SOTA, "test_model_classifier.png"))
        return model_rpn, model_classifier


if __name__ == "__main__": # Run at cwd: custom-fcnn/
    import sys
    sys.path.insert(0, os.getcwd())
    from config.config import config as C

    ################# Check base model (backbone) - determine no. of layers to be frozen #################
    # model_name = "VGG16"
    # x,y = partial_base_model(model_name = model_name, preview = True)

    ################ Get training/testing models and save plot #################
    # model_rpn, model_classifier, model_all = get_faster_rcnn_model(C = C, save_plot = True, base_SOTA = "VGG16", start = 0, end = None, training = True)
    # model_rpn, model_classifier = get_faster_rcnn_model(C = C, save_plot = True, base_SOTA = "VGG16", start = 0, end = None, training = False)

    ################# model with optimizers #################
    # from losses import rpn_loss_regr, rpn_loss_cls, class_loss_regr, class_loss_cls
    # optimizer_rpn = Adam(learning_rate=C.optimizer_lr)
    # optimizer_classifier = Adam(learning_rate=C.optimizer_lr)

    # model_rpn.compile(optimizer=optimizer_rpn, loss=[rpn_loss_cls(C.num_anchors, lambda_rpn_class = C.lambda_rpn_class, epsilon = C.epsilon), rpn_loss_regr(C.num_anchors, lambda_rpn_regr = C.lambda_rpn_regr, epsilon = C.epsilon)])
    # model_classifier.compile(optimizer=optimizer_classifier, loss=[class_loss_cls(lambda_cls_class = C.lambda_cls_class), class_loss_regr(len(C.classes)-1, lambda_cls_regr = C.lambda_cls_regr, epsilon = C.epsilon)], metrics={'dense_class_{}'.format(len(C.classes)): 'accuracy'})
    # # model_all.compile(optimizer='sgd', loss='mae') # just to get the weights for the whole model, no training is involved for this model
    
    # print(model_rpn.optimizer.get_weights())

    pass

