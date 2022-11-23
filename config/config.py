
class config():

    classes = ["Motorcycle", "Car", "Truck", "bg"] # include BG

    # Inputs shape
    img_input_shape = (None, None, 3) 
    roi_input_shape = (None, 4)

    # Anchors
    anchor_box_sizes = [64, 128, 256] #[128, 256, 512] # anchor box scales # 8, 16, 32, 64, 
    anchor_box_ratios = [[1, 1], [1, 2], [2, 1]] # anchor box ratios # , [1, 3], [3, 1]
    num_anchors = len(anchor_box_sizes) * len(anchor_box_ratios)

    # Model - ALL
    epsilon = 1e-4  # loss
    optimizer_lr = 1e-5 #5 # learning rate

    # Model - RPN
    lambda_rpn_regr = 1.0     # Losses
    lambda_rpn_class = 1.0     # Losses
    
    # Model - Classifier
    num_rois = 300 # number of roi regions to be considered for roi pooling output
    pooling_regions = 7 # ROI pooling output P x P shape
    lambda_cls_regr = 1.0
    lambda_cls_class = 1.0

    # Augment (Train)
    augment = {
        'horizontal_flip':True, 'vertical_flip': True, 'rotate': True
    }
    IM_SIZE = 600 # Input size shortest side (according to paper)

    # RPN GT
    rpn_stride = 16 # Depending on your architecture: for e.g. how many max pooling layers etc
    bbox_labels = ["neg", "pos", "neutral"]
    rpn_min_overlap_iou = 0.3 # neg below this threshold
    rpn_max_overlap_iou = 0.7 # pos above this threshold

    num_regions = 256 # number of rpn regions (Ground truth) to be considered
    img_scaling_factor = 1.0
    preprocess_input_import = "imagenet" # Determine which preprocess_input method to import from Keras. (Default = None)


    # File Directories
    train_file = "./dataset/open-images-v6/train_labels.json"
    validation_file = "./dataset/open-images-v6/validation_labels.json"


    # Training
    num_epochs = 300
    epoch_length = 1000 # number of images (batch size) # 32892 6235 231
    nms_max_ROIs = 500 # Maximum number of region of interest (will be further diminish to num_rois)
    nms_overlap_thresh = 0.9 # Objects which have above said threshold to be removed to avoid choosing the same object for non_max_suppression
    classifier_regr_std = [8.0, 8.0, 4.0, 4.0] # [8.0, 8.0, 4.0, 4.0] 
    classifier_min_overlap_iou = 0.3 # neg below this threshold
    classifier_max_overlap = 0.7  # pos above this threshold


    # GPU = False
    # STEP = 0 # 0- train together 1- train rpn, 2- train classifier, 3- finetune rpn, 4- finetune classifier
    # verbose = True 
    # checkpoint = True
    
    # model_weights = "model/weights/model_all.h5" # "model/weights/model_all.h5" 32800/
    # model_weights_checkpoint = "model/weights/model_all_checkpoint.h5"
    # model_weights_rpn = "model/weights/model_rpn_pos.h5"
    # model_weights_classifier = "model/weights/model_classifier.h5"
    # rpn_optimizer_file = 'model/weights/optimizer_rpn.pkl'
    # classifier_optimizer_file = 'model/weights/optimizer_detector.pkl'

    # # Tensorboard for logging
    # log_path = "./logs"




if "__main__" == __name__:
    print(type(config))
    print(isinstance(None, object))
    print(hasattr(config, 'GPU'))
