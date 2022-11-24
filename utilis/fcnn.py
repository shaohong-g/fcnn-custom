import copy, os
import cv2
import random
import contextlib
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K

from utilis.dataloader import get_new_img_size
from utilis.general import calculate_iou 

def apply_regr_np(X, T):
    """Determine the ground-Truth coordinates x1,y1,w,h using T(regre_layer)
    
    Parameters
    ----------
    X : np.array
        (x1,y1,w,h) for every anchor_boxes at every image (height, width)
    T : np.array
        From regr_layer: (4,h,w) tx,ty,tw,th at dim 0

    Returns
    -------
    np.stack([x1, y1, w1, h1])
        Bounding box coordinate at x1,y1
    """
    try:
        x = X[0, :, :]
        y = X[1, :, :]
        w = X[2, :, :]
        h = X[3, :, :]
        
        tx = T[0, :, :]
        ty = T[1, :, :]
        tw = T[2, :, :]
        th = T[3, :, :]

        cx = x + w/2.
        cy = y + h/2.
        cx1 = tx * w + cx # center_x_true
        cy1 = ty * h + cy # center_y_true

        # prevent overflow
        with np.errstate(all='raise'):
            try:
                w1 = np.exp(tw.astype(np.float64)) * w # width_true
                h1 = np.exp(th.astype(np.float64)) * h # height_true
            except Exception:
                print("Value too small to invoke exponential")
                w1 = 0.0
                h1 = 0.0

        x1 = cx1 - w1/2. # x1_true
        y1 = cy1 - h1/2. # y1_true

        x1 = np.round(x1)
        y1 = np.round(y1)
        w1 = np.round(w1)
        h1 = np.round(h1)
        return np.stack([x1, y1, w1, h1])
    except Exception as e:
        print(e)
        return X

def non_max_suppression_fast(boxes, probs, overlap_thresh=0.9, max_boxes=300):
    """Determine the ground-Truth coordinates x1,y1,w,h using T(regre_layer)
    
    Code used from here: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/

    Parameters
    ----------
    boxes : np.array
        (x1,y1,w,h) for every anchor_boxes at every image (height, width)
    probs : np.array
        From regr_layer: (4,h,w) tx,ty,tw,th at dim 0
    overlap_thresh : int, optional
        Objects which have above said threshold to be removed to avoid choosing the same object (Default = 0.9)
    max_boxes : int, optional
        Maximum number of region of interests taken into consideration (Default = 300)

    Returns
    -------
    boxes
        (x1,y1,w,h) of all chosen ROI. Length is based on max_boxes
    pick 
        Probability of an object exists in respect to boxes
    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    np.testing.assert_array_less(x1, x2)
    np.testing.assert_array_less(y1, y2)

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # calculate the areas
    area = (x2 - x1) * (y2 - y1)

    # sort the bounding boxes based on classification probability (most probable to have an object)
    idxs = np.argsort(probs)

    # keep looping while some indexes still remain in the indexes list or when we have chosen max_boxes number of boxes
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the intersection
        xx1_int = np.maximum(x1[i], x1[idxs[:last]])
        yy1_int = np.maximum(y1[i], y1[idxs[:last]])
        xx2_int = np.minimum(x2[i], x2[idxs[:last]])
        yy2_int = np.minimum(y2[i], y2[idxs[:last]])

        ww_int = np.maximum(0, xx2_int - xx1_int)
        hh_int = np.maximum(0, yy2_int - yy1_int)

        area_int = ww_int * hh_int

        # find the union
        area_union = area[i] + area[idxs[:last]] - area_int

        # compute the ratio of overlap,  IOU
        overlap = area_int/(area_union + 1e-6) 

        # delete all indexes from the index list that have IOU more than the threshhold (high IOU means most likely they are detecting the same object)
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))
        
        if len(pick) >= max_boxes:
            break
    
    # return only the bounding boxes that were picked using the integer data type
    boxes = boxes[pick].astype("int")
    probs = probs[pick]
    return boxes, probs

def rpn_to_roi(rpn_layer, regr_layer, C, use_regr=True):
    """ Convert RPN layer to ROI bboxes
    
    rpn_layer: classification (1,h,w,9), regr_layer: (1,h,w,36) -> ROI output (max_boxes, 4)

    1. Calculate x1,y1,w,h for every anchor_boxes at every image (height, width) 
        1.1. Apply regressors via regr_layer to obtain the exact bounding box dimension
        1.2. Ensure that w and h are minimally 1 (at least a 1x1 square box)
    2. Change (x1,y1,w,h) to (x1,y1,x2,y2) (x2 = x1+w, y2 = y2+h)
        2.1. Ensure that x1, y1 are minimally 0 and x2, y2 are maximally img_w-1, img_h-1 (Not out of bound)
    3. Transpose (x1,y1,x2,x2)(4, h, w, layers) to (None, 4) shape as all_boxes
    4. Transpose rpn_layer (1,h,w,9) as (None,) shape as all_probs
    5. Remove all invalid bounding boxes ((x1 - x2 >= 0) | (y1 - y2 >= 0))
    5. Apply non_max_suppression_fast to obtain the top 'max_boxes' boxes. Output: (max_boxes, 4)

    Parameters
    ----------
    rpn_layer : np.array
        classification output from rpn_model (1,h,w,9)
    regr_layer : np.array
        regressor output from rpn_model (1,h,w,36)
    C : object
        Attributes: anchor_box_sizes, anchor_box_ratios, rpn_stride, nms_overlap_thresh, nms_max_ROIs
    use_regr : bool, optional
        Apply regressor to evalute bounding box dimension (Default = True)

    Returns
    -------
    (max_boxes, 4)
        max_boxes, (x1,y1,x2,x2) for region of interest
    """
    regr_layer = regr_layer / 4.0 # Do this if you scaled it in transform_objective function
    
    anchor_sizes = C.anchor_box_sizes
    anchor_ratios = C.anchor_box_ratios

    if K.image_data_format() == "channels_last":
        rows, cols = rpn_layer.shape[1:3] # height, width
        bbox = np.zeros((4, rpn_layer.shape[1], rpn_layer.shape[2], rpn_layer.shape[3]))
    else:
        rows, cols = rpn_layer.shape[2:]
        bbox = np.zeros((4, rpn_layer.shape[2], rpn_layer.shape[3], rpn_layer.shape[1]))

    curr_layer = 0
    for anchor_size in anchor_sizes:
        for anchor_ratio in anchor_ratios:
            # Get anchor relative to output (C.anchor_size is based on the input)
            anchor_x = (anchor_size * anchor_ratio[0])/C.rpn_stride # Output anchor_x (width)
            anchor_y = (anchor_size * anchor_ratio[1])/C.rpn_stride # output anchor_y (height)

            X, Y = np.meshgrid(np.arange(cols), np.arange(rows)) # list of grid[cols][rows] - simulate image size representation

            bbox[0, :, :, curr_layer] = X - anchor_x/2 # x1
            bbox[1, :, :, curr_layer] = Y - anchor_y/2 # y1
            bbox[2, :, :, curr_layer] = anchor_x # width
            bbox[3, :, :, curr_layer] = anchor_y # height
            
            if use_regr: # Get the exact bounding box dimension using bounding-box regressors
                # Manipulate shape: channel, height (row), width (cols)
                if K.image_data_format() == "channels_last": 
                    regr = regr_layer[0, :, :, 4 * curr_layer:4 * curr_layer + 4]  # reduce 1 dim as well
                    regr = np.transpose(regr, (2, 0, 1))
                else:
                    regr = regr_layer[0, 4 * curr_layer:4 * curr_layer + 4, :, :] 

                bbox[:, :, :, curr_layer] = apply_regr_np(bbox[:, :, :, curr_layer], regr)

            bbox[2, :, :, curr_layer] = np.maximum(1, bbox[2, :, :, curr_layer]) # min width = 1
            bbox[3, :, :, curr_layer] = np.maximum(1, bbox[3, :, :, curr_layer]) # min height = 1
            bbox[2, :, :, curr_layer] += bbox[0, :, :, curr_layer] # x2
            bbox[3, :, :, curr_layer] += bbox[1, :, :, curr_layer] # y2

            bbox[0, :, :, curr_layer] = np.maximum(0, bbox[0, :, :, curr_layer]) # min x1 = 0 (cannot escape outside of image boundary)
            bbox[1, :, :, curr_layer] = np.maximum(0, bbox[1, :, :, curr_layer]) # min y1 = 0 (cannot escape outside of image boundary)
            bbox[2, :, :, curr_layer] = np.minimum(cols-1, bbox[2, :, :, curr_layer]) # x2 cannot exceed img_width (-1 to get the last element)
            bbox[3, :, :, curr_layer] = np.minimum(rows-1, bbox[3, :, :, curr_layer]) # y2 cannot exceed img_height (-1 to get the last element)

            curr_layer += 1

    all_boxes = np.reshape(bbox.transpose((0, 3, 1,2)), (4, -1)).transpose((1, 0)) # (None, 4) <- x1,y1,x2,y2
    all_probs = rpn_layer.transpose((0, 3, 1, 2)).reshape((-1)) # (None,)

    x1 = all_boxes[:, 0]
    y1 = all_boxes[:, 1]
    x2 = all_boxes[:, 2]
    y2 = all_boxes[:, 3]

    idxs = np.where((x1 - x2 >= 0) | (y1 - y2 >= 0))

    all_boxes = np.delete(all_boxes, idxs, 0)
    all_probs = np.delete(all_probs, idxs, 0)

    assert all_boxes.shape[0] == all_probs.shape[0]

    result = non_max_suppression_fast(all_boxes, all_probs, overlap_thresh=C.nms_overlap_thresh, max_boxes=C.nms_max_ROIs)[0]
    return result # (max_boxes, 4)


def process_rois(R, img_data, C):
    """ Process ROIs for input to classifier_model

    Parameters
    ----------
    R : np.array
        (None, 4)
    img_data : np.array
       Sample format:
        [{
        'image': './open-images-v6/train/data/0000b9fcba019d36.jpg', 
        'bbox': [
            ['./open-images-v6/train/data/0000b9fcba019d36.jpg', 168.96, 206.079744, 925.44, 766.719744, 'Dog'],
            ['Imagefile', x1, y1, x2, y2, label]
            ]
        }]
    C : object
        Attributes: IM_SIZE, classifier_min_overlap_iou, classifier_max_overlap, classifier_regr_std
    classes : list
        classes including BG
        
    Returns
    -------
    np.expand_dims(X, axis=0) 
        X2 - [x1, y1, w, h] for classifier_model
    np.expand_dims(Y1, axis=0) 
        Y2_1 [1, valid_bbox, labels] labels for classifier_model
    np.expand_dims(Y2, axis=0)
        Y2_2 [1, valid_bbox, 4 (duplicate label) x len(non-bg labels) + 4 (regressor) x len(non-bg labels)] regressor for classifier_model
    IoUs
        For bebug purposes
    """

    classes = C.classes
    bboxes, width, height = img_data['bbox'], img_data['width'], img_data['height']

    # get image dimensions for resizing and find bbox_true for output image dimension
    resized_height, resized_width, multiplier = get_new_img_size(height, width, C.IM_SIZE) # input height and input width
    gta = np.zeros((len(bboxes), 4))
    for bbox_num, bbox in enumerate(bboxes):
        gta[bbox_num, 0] = int(round(bbox[1] * multiplier[1] / C.rpn_stride)) # x1_true for output
        gta[bbox_num, 1] = int(round(bbox[2] * multiplier[0] / C.rpn_stride)) # y1_true for output
        gta[bbox_num, 2] = int(round(bbox[3] * multiplier[1] / C.rpn_stride)) # x2_true for output
        gta[bbox_num, 3] = int(round(bbox[4] * multiplier[0] / C.rpn_stride)) # y2_true for output


    x_roi = []
    y_class_num = []
    y_class_regr_coords = []
    y_class_regr_label = []
    IoUs = [] # for debugging only

    for ix in range(R.shape[0]): # (anchor_boxes, h, w) combination
        (x1, y1, x2, y2) = R[ix, :]
        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x2))
        y2 = int(round(y2))


        # Find which bounding box best suited for x1,y1,x2,y2 coordinates
        best_iou = 0.0
        best_bbox = -1
        for bbox_num in range(len(bboxes)):
            curr_iou = calculate_iou([gta[bbox_num, 0], gta[bbox_num, 1], gta[bbox_num, 2], gta[bbox_num, 3]], [x1, y1, x2, y2])
            if curr_iou > best_iou:
                best_iou = curr_iou
                best_bbox = bbox_num

        if best_iou < C.classifier_min_overlap_iou:
            continue
        else:
            w = x2 - x1
            h = y2 - y1
            x_roi.append([x1, y1, w, h])
            IoUs.append(best_iou)

            if C.classifier_min_overlap_iou <= best_iou < C.classifier_max_overlap:
                # hard negative example
                cls_name = 'bg'
            elif C.classifier_max_overlap <= best_iou:
                cls_name = bboxes[best_bbox][-1]  # label

                # calculate regressor
                cxg = (gta[best_bbox, 0] + gta[best_bbox, 2]) / 2.0
                cyg = (gta[best_bbox, 1] + gta[best_bbox, 3]) / 2.0

                cx = x1 + w / 2.0
                cy = y1 + h / 2.0

                tx = (cxg - cx) / float(w)
                ty = (cyg - cy) / float(h)
                tw = np.log((gta[best_bbox, 2] - gta[best_bbox, 0]) / float(w))
                th = np.log((gta[best_bbox, 3] - gta[best_bbox, 1]) / float(h))
            else:
                print('roi = {}'.format(best_iou))
                raise RuntimeError

        class_num = classes.index(cls_name) # class_num = class_mapping[cls_name]
        class_label = len(classes) * [0.0]
        class_label[class_num] = 1.0
        y_class_num.append(copy.deepcopy(class_label))

        # Y_True regressor for classes other than bg
        coords = [0.0] * 4 * (len(classes) - 1)
        labels = [0.0] * 4 * (len(classes) - 1)
        if cls_name != 'bg':
            label_pos = 4 * class_num
            sx, sy, sw, sh = C.classifier_regr_std # 4
            coords[label_pos:4+label_pos] = [sx*tx, sy*ty, sw*tw, sh *th]
            labels[label_pos:4+label_pos] = [1.0, 1.0, 1.0, 1.0]
            y_class_regr_coords.append(copy.deepcopy(coords))
            y_class_regr_label.append(copy.deepcopy(labels))
        else:
            y_class_regr_coords.append(copy.deepcopy(coords))
            y_class_regr_label.append(copy.deepcopy(labels))

    if len(x_roi) == 0:
        return None, None, None, None

    X = np.array(x_roi)
    Y1 = np.array(y_class_num)
    Y2 = np.concatenate([np.array(y_class_regr_label),np.array(y_class_regr_coords)],axis=1)

    return np.expand_dims(X, axis=0), np.expand_dims(Y1, axis=0), np.expand_dims(Y2, axis=0), IoUs
