import os
import cv2
import json
import copy
import random
import numpy as np

from keras import backend as K
from utilis.general import calculate_iou 

def get_data(LABEL_JSON, format = None):
    """ Get data

    Parameters
    ----------
    LABEL_JSON : str
        File path relative to the folder that you run
    format : str, optional
        Takes in one of the value [None, multi_label, multi_class] (Default is None) - check code comments
        
    Returns
    -------
    data_json : {'image': 'C:\\Users\\gansh\\Documents\\work\\custom-fcnn\\dataset\\open-images-v6\\train\\images\\0000048549557964.jpg', 'classes': ['Car'], 'bbox': [['0000048549557964', 
199.04, 552.96, 334.08, 582.399744, 'Car'], ['0000048549557964', 445.44, 536.319744, 743.04, 725.1202559999999, 'Car'], ['0000048549557964', 723.2, 526.719744, 959.36, 664.3199999999999, 'Car'], ['0000048549557964', 848.64, 531.84, 1023.36, 697.599744, 'Car']]}
    """
    label_path = os.path.realpath(LABEL_JSON)
    assert os.path.exists(label_path), f"{label_path} does not exists"
    print("Getting data from", label_path)

    with open(label_path, 'r') as f:
        data_json = json.load(f)

    if format is None:
        return data_json
    elif format == "multi_label": # Get images with more than one bounding box
        return list(filter(lambda x: len(x["bbox"]) > 1, data_json))
    elif format == "multi_class": # Get images with more than 1 class
        return list(filter(lambda x: len(x["classes"]) > 1, data_json))

##############################
# Data Generator
##############################
def augment(img_data, config = None):
    """Augment images

    Parameters
    ----------
    img_data : dictionary
        Dictionary objects containing image and bounding box (absolute coordinates) data.
        Sample format:
        {'image': './open-images-v6/train/data/0000b9fcba019d36.jpg', 
        'bbox': [
            ['./open-images-v6/train/data/0000b9fcba019d36.jpg', 168.96, 206.079744, 925.44, 766.719744, 'Dog'],
            ['Imagefile', x1, y1, x2, y2, label]
        ]}
    config : dict, optional
        Configuration for augmentation. Check sample for the respective keys. (default = None - No augementation)
        Sample format: 
            {'horizontal_flip'=True, 'vertical_flip': False, 'rotate': True}

    Returns
    -------
    img_data_aug :
        img_data format with modified bounding box coordinates, height and width
    img : 
        modified image vector (H, W, C)
    """

    if config is not None:
        assert 'horizontal_flip' in config 
        assert 'vertical_flip' in config 
        assert 'rotate' in config
    
    img_data_aug = copy.deepcopy(img_data)
    
    img = cv2.imread(img_data_aug['image'])

    if config is not None:
        height, width = img.shape[:2] # rows, cols 

        if config['horizontal_flip'] and np.random.randint(0, 2) == 0: # random flip will occur if True
            # print("Horizontal Flip: True")
            img = cv2.flip(img, 1)
            for bbox in img_data_aug['bbox']:
                x1 = bbox[1]
                x2 = bbox[3]
                bbox[3] = width - x1
                bbox[1] = width - x2
        
        if config['vertical_flip'] and np.random.randint(0, 2) == 0: # random flip will occur if True
            # print("Vertical_flip Flip: True")
            img = cv2.flip(img, 0)
            for bbox in img_data_aug['bbox']:
                y1 = bbox[2]
                y2 = bbox[4]
                bbox[4] = height - y1
                bbox[2] = height - y2

        if config['rotate']: # random rotation will occur if True
            angle = np.random.choice([0,90,180,270],1)[0]
            
            # print(f"Rotation Angle: {angle}")
            if angle == 270:
                img = np.transpose(img, (1,0,2))
                img = cv2.flip(img, 0)
            elif angle == 180:
                img = cv2.flip(img, -1)
            elif angle == 90:
                img = np.transpose(img, (1,0,2))
                img = cv2.flip(img, 1)
            elif angle == 0:
                pass

            for bbox in img_data_aug['bbox']:
                x1 = bbox[1]
                x2 = bbox[3]
                y1 = bbox[2]
                y2 = bbox[4]
                if angle == 270:
                    bbox[1] = y1
                    bbox[3] = y2
                    bbox[2] = width - x2
                    bbox[4] = width - x1
                elif angle == 180:
                    bbox[3] = width - x1
                    bbox[1] = width - x2
                    bbox[4] = height - y1
                    bbox[2] = height - y2
                elif angle == 90:
                    bbox[1] = height - y2
                    bbox[3] = height - y1
                    bbox[2] = x1
                    bbox[4] = x2
                elif angle == 0:
                    pass

    img_data_aug['width'] = img.shape[1]
    img_data_aug['height'] = img.shape[0]
    return img_data_aug, img

def get_new_img_size(height, width, min_size=600):
    """ Get image height and weight with the shortest side being min_size

    Parameters
    ----------
    height : float
    width : float
    min_size: int, optional
        default = 600

    Returns
    -------
    new_height, new_width, multiplier
    """
    multiplier_h = min_size / height
    multiplier_w = min_size / width
    multiplier = [multiplier_h, multiplier_w]
    return int(height * multiplier_h), int(width * multiplier_w), multiplier

def get_achorb_gt(input_size, img_data_aug, multiplier, C):
    """Get ground truth for anchors in relative to the bounding box ground truth 

    Parameters
    ----------
    input_size : list
        Input size (Feed Model) of image. Format: [height, weight]
    img_data_aug : dict
        Dictionary objects containing image and bounding box (absolute coordinates) data.
        Sample format:
        {'image': './open-images-v6/train/data/0000b9fcba019d36.jpg', 
        'bbox': [['./open-images-v6/train/data/0000b9fcba019d36.jpg', 168.96, 206.079744, 925.44, 766.719744, 'Dog'],['Imagefile', x1, y1, x2, y2, label]],
        'height': 123,
        'width': 123}
    img : list
        Modified image vector
    multiplier : float
        size factor base on input image over original image size
    C : class
        Configuration class
    checkpoint : int, optional
        Debug codes (Default = None)

    Returns
    -------
    y_rpn_overlap : list
        (h, w, 9) Truthy value at each anchor (if there is an overlapping positive region ("pos"))
    y_is_box_valid: list
        (h, w, 9) Truthy value at each anchor (if there is an overlapping box region ("pos" and "neg"))
    y_rpn_regr: list
        (h, w, 36),  (tx, ty, tw, th) at each anchor
    """

    # Configuration
    rpn_stride = C.rpn_stride # Depending on your architecture: for e.g. how many max pooling layers etc
    anchor_box_sizes = C.anchor_box_sizes # anchor box scales
    anchor_box_ratios = C.anchor_box_ratios # anchor box ratios
    num_anchors = C.num_anchors
    bbox_labels = C.bbox_labels
    rpn_min_overlap_iou = C.rpn_min_overlap_iou
    rpn_max_overlap_iou = C.rpn_max_overlap_iou

    resized_height, resized_width = input_size[0], input_size[1]
    output_height, output_width = int(resized_height / rpn_stride), int(resized_width / rpn_stride) # (h, w)

    # initialise empty output objectives
    y_rpn_overlap = np.zeros((output_height, output_width, num_anchors)) # (h, w, 9) - Truthy value at each anchor (if there is an overlapping positive region ("pos"))
    y_is_box_valid = np.zeros((output_height, output_width, num_anchors)) # (h, w, 9) - Truthy value at each anchor (if there is an overlapping box region ("pos" and "neg"))
    y_rpn_regr = np.zeros((output_height, output_width, num_anchors * 4)) # (h, w, 36) - (tx, ty, tw, th) at each anchor

    num_bbox = len(img_data_aug['bbox'])

    num_anchors_for_bbox = np.zeros(num_bbox).astype(int) # number of anchors (based on max overlap iou) for each bounding box
    best_anchor_for_bbox = -1 * np.ones((num_bbox, 4)).astype(int) # best [ix, iy, anchor_size_index, anchor_ratio_index] best anchor config for each bounding box
    best_iou_for_bbox = np.zeros(num_bbox).astype(np.float32) # best iou for each bounding box
    best_x_for_bbox = np.zeros((num_bbox, 4)).astype(int) # best [x1_anchor_bbox, y1_anchor_bbox, x2_anchor_bbox,  y2_anchor_bbox] anchor size (Input) for each bounding box
    best_dx_for_bbox = np.zeros((num_bbox, 4)).astype(np.float32) # best [tx, ty, tw, th] for each bounding box
    anchor_index = -1

    # get the Gound Truth box coordinates, and resize (Input size) to account for image resizing
    gt_box = np.zeros((num_bbox, 4)) # shape = (num_bbox, bbox_coordinates)
    for index, bbox in enumerate(img_data_aug['bbox']):
        gt_box[index, 0] = bbox[1] * multiplier[1] #x1
        gt_box[index, 1] = bbox[2] * multiplier[0] #y1
        gt_box[index, 2] = bbox[3] * multiplier[1] #x2
        gt_box[index, 3] = bbox[4] * multiplier[0] #y2

    
    # rpn ground truth
    for anchor_size_index in range(len(anchor_box_sizes)):
        for anchor_ratio_index in range(len(anchor_box_ratios)): 
            anchor_x = anchor_box_sizes[anchor_size_index] * anchor_box_ratios[anchor_ratio_index][0]
            anchor_y = anchor_box_sizes[anchor_size_index] * anchor_box_ratios[anchor_ratio_index][1]
            anchor_index = anchor_size_index * (len(anchor_box_ratios)) + anchor_ratio_index  # anchor_index += 1 

            for ix in range(output_width):
                # x-coordinates of the current anchor box (reference to Input width) => rpn_stride * midpoint +/- half of anchor
                # Reason for not taking reference to output width is so that the bounding box can capture smaller objects
                x1_anchor_bbox = rpn_stride * (ix + 0.5) - anchor_x / 2 
                x2_anchor_bbox = rpn_stride * (ix + 0.5) + anchor_x / 2
                
                # ignore boxes that go across image boundaries
                if x1_anchor_bbox < 0 or x2_anchor_bbox > resized_width:
                    continue
                
                
                for iy in range(output_height):
                    # y-coordinates of the current anchor box (reference to Input Height) 
                    y1_anchor_bbox = rpn_stride * (iy + 0.5) - anchor_y / 2
                    y2_anchor_bbox = rpn_stride * (iy + 0.5) + anchor_y / 2
                    # ignore boxes that go across image boundaries
                    if y1_anchor_bbox < 0 or y2_anchor_bbox > resized_height:
                        continue
                    
                    # Best IOU for the (x,y) coord at the current anchor
                    best_iou_current_loc = 0.0
                    bbox_type = bbox_labels[0]
                    
                    # anchor_index = anchor_ratio_index + len(anchor_box_sizes) * anchor_size_index # unique from (0-8 inclusive) for each anchor_size and anchor_ratio combination 
                    
                    for bbox_index in range(num_bbox): # for each bounding boxes 
                        current_iou = calculate_iou([gt_box[bbox_index, 0], gt_box[bbox_index, 1], gt_box[bbox_index, 2], gt_box[bbox_index, 3]], [x1_anchor_bbox, y1_anchor_bbox, x2_anchor_bbox,  y2_anchor_bbox])
                        
                        # calculate the regression targets if they will be needed
                        # authors used a linear scale for training the center and the log-space for training the height and width.
                        if current_iou > best_iou_for_bbox[bbox_index] or current_iou >= rpn_max_overlap_iou:
                            # Centre point of gt box and anchor box
                            cx_gt = (gt_box[bbox_index, 0] + gt_box[bbox_index, 2]) / 2.0
                            cy_gt = (gt_box[bbox_index, 1] + gt_box[bbox_index, 3]) / 2.0
                            cx_ab = (x1_anchor_bbox + x2_anchor_bbox)/2.0
                            cy_ab = (y1_anchor_bbox + y2_anchor_bbox)/2.0
                            
                            # displacement of the center relative to current anchor
                            tx = (cx_gt - cx_ab) / (x2_anchor_bbox - x1_anchor_bbox)
                            ty = (cy_gt - cy_ab) / (y2_anchor_bbox - y1_anchor_bbox)
                            # log space for difference in width and height
                            tw = np.log((gt_box[bbox_index, 2] - gt_box[bbox_index, 0]) / (x2_anchor_bbox - x1_anchor_bbox))
                            th = np.log((gt_box[bbox_index, 3] - gt_box[bbox_index, 1]) / (y2_anchor_bbox - y1_anchor_bbox))
                        
                        
                            # all GT boxes should be mapped to an anchor box, so we keep track of which anchor box is the best
                            if current_iou > best_iou_for_bbox[bbox_index]:
                                best_anchor_for_bbox[bbox_index] = [ix, iy, anchor_size_index, anchor_ratio_index]
                                best_iou_for_bbox[bbox_index] = current_iou
                                best_x_for_bbox[bbox_index,:] = [x1_anchor_bbox, y1_anchor_bbox, x2_anchor_bbox,  y2_anchor_bbox]
                                best_dx_for_bbox[bbox_index,:] = [tx, ty, tw, th]

                            # we set the anchor to positive if the IOU is >0.7 (it does not matter if there was another better box, it just indicates overlap)
                            if current_iou >= rpn_max_overlap_iou:
                                bbox_type = bbox_labels[1] # "pos"
                                num_anchors_for_bbox[bbox_index] += 1
                                
                                # we update the regression layer target if this IOU is the best for the current (x,y) and anchor position
                                if current_iou > best_iou_current_loc:
                                    best_iou_current_loc = current_iou
                                    best_regr = (tx, ty, tw, th)
                                    
                                    start = 4 * (anchor_index)
                                    y_rpn_regr[iy, ix, start:start+4] = best_regr 
                                    
                        # if the IOU is >0.3 and <0.7, it is ambiguous and no included in the objective
                        if rpn_min_overlap_iou < current_iou < rpn_max_overlap_iou:
                            # gray zone between neg and pos
                            if bbox_type != bbox_labels[1]:
                                bbox_type = bbox_labels[2]
                        
                    # turn on or off outputs depending on IOUs
                    if bbox_type == bbox_labels[0]:
                        # print(y_is_box_valid.shape, iy, ix, anchor_index)
                        y_is_box_valid[iy, ix, anchor_index] = 1 # (37, 56, 9)
                        y_rpn_overlap[iy, ix, anchor_index] = 0
                    elif bbox_type == bbox_labels[2]: # 'neutral'
                        y_is_box_valid[iy, ix, anchor_index] = 0
                        y_rpn_overlap[iy, ix, anchor_index] = 0
                    elif bbox_type == bbox_labels[1]: # 'pos'
                        y_is_box_valid[iy, ix, anchor_index] = 1
                        y_rpn_overlap[iy, ix, anchor_index] = 1
            
    # we ensure that every bbox has at least one positive RPN region (Take the best value (even if it lower than maxoverlapiou) for the bounding box index which does not have an anchor)
    for idx in range(num_anchors_for_bbox.shape[0]): # tuple(number of bounding box,)
        if num_anchors_for_bbox[idx] == 0: # no box with an IOU greater than rpn_max_overlap_iou ...
            if best_anchor_for_bbox[idx, 0] == -1: # no anchor for the bounding box (should not be the case) -> should raise Exception (BUT it will not have if bounding box cannot detect(too big))
                continue
                
            anchor_index = best_anchor_for_bbox[idx, 3] + len(anchor_box_ratios) * best_anchor_for_bbox[idx, 2] # anchor_index = anchor_size_index * (len(anchor_box_ratios)) + anchor_ratio_index
            y_is_box_valid[best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,0], anchor_index] = 1
            y_rpn_overlap[best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,0], anchor_index] = 1
            start = 4 * (anchor_index)
            y_rpn_regr[best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,0], start:start+4] = best_dx_for_bbox[idx, :]
    
    return y_rpn_overlap, y_is_box_valid, y_rpn_regr

def transform_image_vector(img, preprocess_input_import, img_scaling_factor):
    # Transform Image Vector, Zero-center by mean pixel, and preprocess image
    if preprocess_input_import == "imagenet":
        from keras.applications.imagenet_utils import preprocess_input
        img = preprocess_input(img)

    # # Can be substitute by preprocess_input module in keras
    # x_img = copy.deepcopy(img)
    # img_channel_mean = [103.939, 116.779, 123.68]
    # x_img = x_img[:, :, (2, 1, 0)]  # BGR -> RGB
    # x_img = x_img.astype(np.float32)
    # x_img[:, :, 0] -= img_channel_mean[0]
    # x_img[:, :, 1] -= img_channel_mean[1]
    # x_img[:, :, 2] -= img_channel_mean[2]

    img /= img_scaling_factor

    img = np.transpose(img, (2, 0, 1)) # (1024, 683, 3) -> (3, 1024, 683)
    img = np.expand_dims(img, axis=0)

    return img

def transform_objective(y_rpn_overlap, y_is_box_valid, y_rpn_regr, num_regions = 256):
    """
    y_rpn_overlap = np.zeros((output_height, output_width, num_anchors)) # (37, 56, 9) - Truthy value at each anchor (if there is an overlapping positive region ("pos"))
    y_is_box_valid = np.zeros((output_height, output_width, num_anchors)) # (37, 56, 9) - Truthy value at each anchor (if there is an overlapping box region ("pos" and "neg"))
    y_rpn_regr = np.zeros((output_height, output_width, num_anchors * 4)) # (37, 56, 36) - (tx, ty, tw, th) at each anchor
    """
    # Transform shape
    y_rpn_overlap = np.transpose(y_rpn_overlap, (2, 0, 1)) # (output_height, output_width, num_anchors) -> (num_anchors, output_height-y, output_width -x)
    y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis=0) # increase dimension (sample, num_anchors, output_height-y, output_width -x)

    y_is_box_valid = np.transpose(y_is_box_valid, (2, 0, 1))
    y_is_box_valid = np.expand_dims(y_is_box_valid, axis=0)

    y_rpn_regr = np.transpose(y_rpn_regr, (2, 0, 1))
    y_rpn_regr = np.expand_dims(y_rpn_regr, axis=0)

    pos_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1))
    neg_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1))

    num_pos = len(pos_locs[0]) # Number of possible positive anchors
    
    # one issue is that the RPN has many more negative than positive regions, so we turn off some of the negative regions. We also limit it to 256 regions. -> Output layer before classification and regression layers in rpn
    # Limit number of positive region(anchors) to num_regions/2 or less, take the excess off by random sampling
    if len(pos_locs[0]) > num_regions/2:
        val_locs = random.sample(range(len(pos_locs[0])), len(pos_locs[0]) - int(num_regions/2)) # mask for excess pos_loc
        y_is_box_valid[0, pos_locs[0][val_locs], pos_locs[1][val_locs], pos_locs[2][val_locs]] = 0
        num_pos = int(num_regions/2)

    # Limit number of negative region(anchors) to num_regions/2, take the excess off by random sampling (only take into account of num_regions)
    if len(neg_locs[0]) + num_pos > num_regions:
        val_locs = random.sample(range(len(neg_locs[0])), len(neg_locs[0]) - num_pos) # + num_pos - num_regions Changes from - num_pos:  so that y_is_box_valid = 256
        y_is_box_valid[0, neg_locs[0][val_locs], neg_locs[1][val_locs], neg_locs[2][val_locs]] = 0

    y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=1) # concatenate via index 1 (1,9,56,37)*2 -> (1, 18, 56, 37)
    y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=1), y_rpn_regr], axis=1) # (1, 9, 56, 37), (1, 36, 56, 37) -> (1, 72, 56, 37), each regression loss will reference to the respective anchor

    std_scaling = 4.0 # we are repeating y_rpn_overlap 4 times, scale it during losses calculation
    y_rpn_regr[:, y_rpn_regr.shape[1]//2:, :, :] *= std_scaling

    return y_rpn_cls, y_rpn_regr

def transform_output(y_rpn_overlap, y_is_box_valid, y_rpn_regr, img_data_aug, img, input_size, C):
    """Transform shape of output (objective and img) to suit model
    
    Parameters
    ----------
    y_rpn_overlap : list
        (h, w, 9) Truthy value at each anchor (if there is an overlapping positive region ("pos"))
    y_is_box_valid : list
        (h, w, 9) Truthy value at each anchor (if there is an overlapping box region ("pos" and "neg"))
    y_rpn_regr: list
        (h, w, 36) (tx, ty, tw, th) at each anchor
    img_data_aug : dict
        Dictionary objects containing image and bounding box (absolute coordinates) data.
        Sample format:
            {'image': './open-images-v6/train/data/0000b9fcba019d36.jpg', 
            'bbox': [['./open-images-v6/train/data/0000b9fcba019d36.jpg', 168.96, 206.079744, 925.44, 766.719744, 'Dog'], ['Imagefile', x1, y1, x2, y2, label]],
            'height': 123,
            'width': 123}
    img : list
        Modified image vector
    input_size : list
        [resized_height, resized_width]
    C : class
        Configuration class that contains preprocess_input_import and img_scaling_factor
    checkpoint : Bool, optional
        Debug codes (Default = False)


    Returns
    -------
    np.copy(img): (h,w, anchors)
    [np.copy(y_rpn_cls), np.copy(y_rpn_regr)]: [(h,w, anchors * 2), (h,w, anchors* 4 * 2)]
    img_data_aug: original
    """

    # Transform shape for objective 
    y_rpn_cls, y_rpn_regr = transform_objective(y_rpn_overlap, y_is_box_valid, y_rpn_regr, C.num_regions)

    # Transform image
    img = cv2.resize(img, dsize=(input_size[1], input_size[0]), interpolation=cv2.INTER_CUBIC)
    img = transform_image_vector(img, C.preprocess_input_import, C.img_scaling_factor)

    
    if K.image_data_format() == "channels_last":
        img = np.transpose(img, (0, 2, 3, 1))  # (3, 1024, 683) -> (1024, 683, 3)
        y_rpn_cls = np.transpose(y_rpn_cls, (0, 2, 3, 1))
        y_rpn_regr = np.transpose(y_rpn_regr, (0, 2, 3, 1))

    return np.copy(img), [np.copy(y_rpn_cls), np.copy(y_rpn_regr)], img_data_aug

def data_generator(train_data, C , seed = 64):
    """ Data generator for training
    
    Required Packages: 

    Parameters
    ----------
    train_data : list
        List of dictionaries that contains image path and bounding boxes coordinates. Format:
        {'image': './open-images-v6/train/data/0000b9fcba019d36.jpg', 
        'bbox': [
            ['./open-images-v6/train/data/0000b9fcba019d36.jpg', 168.96, 206.079744, 925.44, 766.719744, 'Dog'],
            ['Imagefile', x1, y1, x2, y2, label]
        ]}
    C : class
        Configuration class.
    
    Returns
    ---------
    img, groundtruth, img_data_aug
    """
    import random

    random.Random(seed).shuffle(train_data) # random.shuffle(train_data)

    for each_image in train_data:
        try:
            # Augment data
            img_data_aug, img = augment(img_data=each_image, config = C.augment)
            
            # Scale dimension to C.IM_SIZE (Default = 600) for input to Neural Network
            resized_height, resized_width, multiplier = get_new_img_size(img_data_aug["height"], img_data_aug["width"],  C.IM_SIZE)

            # Get achorbox groundtruth, img vector (transform_image_vector) and img_data_aug (no change)
            y_rpn_overlap, y_is_box_valid, y_rpn_regr = get_achorb_gt(input_size = [resized_height, resized_width], img_data_aug = img_data_aug, multiplier = multiplier, C = C)
            
            img, groundtruth, img_data_aug = transform_output(y_rpn_overlap, y_is_box_valid, y_rpn_regr, img_data_aug, img, input_size = [resized_height, resized_width], C=C)
            
            yield img, groundtruth, img_data_aug # np.copy(img), [np.copy(y_rpn_cls), np.copy(y_rpn_regr)], img_data_aug

        except Exception as e:
            raise Exception(f"Error in data_generator") from e
