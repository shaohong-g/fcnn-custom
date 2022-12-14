import os, math, time, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import traceback
import tensorflow as tf
# tf.config.run_functions_eagerly(True)
# print(tf.config.list_physical_devices('GPU'))

from config.config import config as C
from model.model import get_faster_rcnn_model
from utilis.general import get_logger, get_ious, NumpyEncoder
from utilis.dataloader import get_data, get_new_img_size, transform_image_vector
from utilis.fcnn import non_max_suppression_fast, fcnn_get_results
from utilis.metrics import ap_per_class, ConfusionMatrix
# from metrics.BoundingBox import BoundingBox
# from metrics.BoundingBoxes import BoundingBoxes
# from metrics.utils import *

def plot_bboxes(image, bboxes, gt_bbox, multiplier, out_file, classes):
    color_array = [(255,0,0), (0,0,255), (0,255,0), (0, 0, 0), (255,255,255)] # red, blue, green, black, white
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    text_scale = image.shape[1] / 1000

    for x1, y1, x2, y2, prob, label in bboxes:

        cv2.rectangle(image, (x1,y1), (x2,y2), color_array[0], 2)
        textLabel = f'{classes[label]}: {round(100*prob,2)}%'
        (retval, baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_SIMPLEX,text_scale,1)

        textOrg = (min(x1,x2) , min(y1,y2) + retval[1] )
        cv2.rectangle(image, (min(x1,x2), min(y1,y2)), (min(x1,x2) + retval[0] , min(y1,y2) + retval[1]+baseLine), color_array[-1], -1)
        cv2.putText(image, textLabel, textOrg, cv2.FONT_HERSHEY_SIMPLEX, text_scale, color_array[-2], 1)

    for i in range(len(gt_bbox)):
        x1 = int(gt_bbox[i][1] / multiplier[1])
        y1 = int(gt_bbox[i][2] / multiplier[0])
        x2 = int(gt_bbox[i][3] / multiplier[1])
        y2 = int(gt_bbox[i][4] / multiplier[0])
        cv2.rectangle(image, (x1,y1), (x2,y2), color_array[2], 2)
        textLabel = f'{gt_bbox[i][5]}: GT'
        (retval, baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_SIMPLEX,text_scale,1)

        textOrg = (max(x1,x2)- retval[0] , max(y1,y2)-baseLine )
        cv2.rectangle(image, (max(x1,x2) - retval[0], max(y1,y2) - retval[1]-baseLine), (max(x1,x2) , max(y1,y2)), color_array[-2], -1)
        cv2.putText(image, textLabel, textOrg, cv2.FONT_HERSHEY_SIMPLEX, text_scale, color_array[-1], 1)
    
    cv2.imwrite(out_file, image)



def test(save_dir, model_weight, logger, annotations, C, save_txt= None, plots= True, v5_metric = False, train = False, sample_rate = 100):
    """
    
    """
    assert os.path.exists(annotations)
    assert os.path.exists(model_weight)

    # Get models
    model_rpn, model_classifier = get_faster_rcnn_model(C = C, save_plot = False, base_SOTA = C.backbone, start = 0, end = None, training = False)
    model_rpn.load_weights(model_weight, by_name=True) 
    model_classifier.load_weights(model_weight, by_name=True) #C.model_weights

    model_rpn.compile(optimizer='sgd', loss='mse')
    model_classifier.compile(optimizer='sgd', loss='mse')
    logger.info(F"Loaded weights from: {model_weight}")

    # testing_data and classes
    testing_data = get_data(annotations)
    classes = C.classes
    names = {i: classes[i] for i in range(len(classes))}
    names.pop(len(classes)-1, None) # pop background
    nc = len(classes) - 1 # without background
    seen = 0
    if plots: confusion_matrix = ConfusionMatrix(nc=nc, conf=C.bbox_threshold, iou_thres=0.5)
    stats = []
    if save_txt: plabel = []
    start = time.time()

    # Test
    total_counter = len(testing_data)
    for counter, data in enumerate(testing_data):
        image = data["image"]
        gt_bbox = data["bbox"]
        assert os.path.exists(image), f"Image does not exists - {image}"

        start_image = time.time()
        # Process Image
        img_original = cv2.imread(image)
        resized_height, resized_width, multiplier = get_new_img_size(img_original.shape[0], img_original.shape[1],  C.IM_SIZE)
        img = cv2.resize(img_original, dsize=(resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
        img = transform_image_vector(img, C.preprocess_input_import, C.img_scaling_factor)
        img = np.transpose(img, (0, 2, 3, 1))

        # Process bbox (Ground Truth) x1,y1,x2,y2
        tcls = []
        for i in range(len(gt_bbox)):
            gt_bbox[i][1] = gt_bbox[i][1] * multiplier[1]
            gt_bbox[i][2] = gt_bbox[i][2] * multiplier[0]
            gt_bbox[i][3] = gt_bbox[i][3] * multiplier[1]
            gt_bbox[i][4] = gt_bbox[i][4] * multiplier[0]
            tcls.append(classes.index(gt_bbox[i][5]))

        # Fit to models
        bboxes, probs = fcnn_get_results(img, model_rpn, model_classifier, classes, C)

        if len(bboxes) == 0:
            logger.info(f"{counter}/{total_counter} No bounding box found in {image}")
            continue
        
        # NMS and get all detections
        seen += 1
        all_dets = []
        if save_txt: plabel.append({"image": image}) # save_txt_list = []

        iouv = np.linspace(0.5, 0.95, 10)
        correct = [] # np.zeros((pred_count, niou), dtype = bool)
        conf = []
        pcls = []
        detected = []


        for key in bboxes.keys(): # each label
            bbox = np.array(bboxes[key])
            new_boxes, new_probs = non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh= C.nms_dets)

            conf.extend(new_probs)
            pcls.extend([classes.index(key)] * len(new_probs))
            for jk in range(new_boxes.shape[0]):
                # Calculate real coordinates on original image 
                (x1, y1, x2, y2) = new_boxes[jk,:] 
                real_x1, real_y1, real_x2, real_y2 = int(x1 / multiplier[1]), int(y1 / multiplier[0]), int(x2 / multiplier[1]), int(y2 / multiplier[0]) 
                real_x1 = np.maximum(0, real_x1) # min x1 = 0 (cannot escape outside of image boundary)
                real_y1 = np.maximum(0, real_y1) # min y1 = 0 (cannot escape outside of image boundary)
                real_x2 = np.minimum(img_original.shape[1], real_x2) # x2 cannot exceed img_original_width (-1 to get the last element)
                real_y2 = np.minimum(img_original.shape[0], real_y2) # y2 cannot exceed img_original_height (-1 to get the last element)
                all_dets.append([real_x1, real_y1, real_x2, real_y2, new_probs[jk], classes.index(key)])

                # Calculate True Positive
                correct.append(np.array([False] * len(iouv)))
                ious, idx_arr = get_ious(ground_truth_arr = gt_bbox, pred = [real_x1, real_y1, real_x2, real_y2])
                for idx in idx_arr:
                    if len(detected) == len(gt_bbox):
                        break # all labels in image have been detected
                    # Not detected, iou > 0.5 (default) and pred class == true class
                    if idx not in detected and ious[idx] >= iouv[0] and key == gt_bbox[idx][5]:
                        detected.append(idx)
                        correct[-1] = ious[idx] >= iouv
                        break

                # if save_txt:
                #     save_txt_list.append(f"{key} {real_x1} {real_y1} {real_x2} {real_y2}")

        assert len(conf) == len(pcls) == len(correct)

        # if save_txt:
        #     os.makedirs(os.path.join(save_dir, "plabels"), exist_ok=True)
        #     with open(os.path.join(save_dir, "plabels", f"{os.path.basename(image).split('.')[0]}.txt"),'w') as f:
        #         f.write("\n".join(save_txt_list))
        if save_txt: plabel[-1]["plabels"] = all_dets
        if plots:
            confusion_matrix.process_batch(detections = np.array(all_dets), labels= np.array(gt_bbox), classes = classes)
            os.makedirs(os.path.join(save_dir, "sample"), exist_ok=True)
            if (counter + 1) % sample_rate == 0:
                plot_bboxes(img_original, all_dets, gt_bbox, multiplier, os.path.join(save_dir, "sample", os.path.basename(image)), classes)

        stats.append([np.array(correct), np.array(conf), np.array(pcls), np.array(tcls)])

        logger.info(f"Tested {counter + 1}/{total_counter}: {image} - {len(conf)} objects, Elapsed Time: {time.time() - start_image:.2f}s")

    # Statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    p, r, f1, mp, mr, map50, map = 0., 0., 0., 0., 0., 0., 0.
    if len(stats) and stats[0].any(): # must have correct labels
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, v5_metric=v5_metric, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = np.zeros(1)

    # Print results
    logger.info(('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95'))
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
    logger.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if not train and nc > 1 and len(stats) and stats[0].any():
        for i, c in enumerate(ap_class):
            logger.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))
    
    if save_txt:
        with open(os.path.join(save_dir, "plabels.json"), 'w') as f:
            json.dump(plabel, f, indent=4, cls= NumpyEncoder)
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        

    logger.info(f"Testing Completed! Fail to detect: {total_counter - counter -1} images,  Elapsed Time: {time.time() - start:.2f}s")
    


if __name__ == "__main__":
    """
    python test.py --device -1 --save-txt --name openimages --weights ./runs/train/exp1/model.h5 --annotations ./dataset/open-images-v6/validation_labels.json --hyp ./runs/train/exp1/hyp.json --sample-rate 10
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Resume testing as per name')
    parser.add_argument('--save-txt', action='store_true', help='Save Labels in txt format')
    parser.add_argument('--name', type=str, default='exp', help='Experiment name')
    parser.add_argument('--device', type=int, default='-1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu (-1)')
    parser.add_argument('--weights', type=str, default='./model.h5', help='File location of where the weight file is saved (relative to this file)')
    parser.add_argument('--annotations', type=str, required=True, help='File location of where the annotation file is saved (relative to this file)')
    parser.add_argument('--hyp', type=str, required=True, help='File location of where the config (hyp.json) file is saved (relative to this file)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Confidence level for classifier')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--sample-rate', type=int, default=100, help='Save samples')
    parser.add_argument('--logfile', type=str, default='test.log', help='Log file')
    opt = parser.parse_args()

    # Create save_dir
    save_parent = os.path.join(os.path.dirname(__file__), "runs", "test")
    os.makedirs(save_parent, exist_ok=True)
    parent_folders = next(os.walk(save_parent))[1] # all folders in parent
    save_count = sum([x.startswith(opt.name) for x in parent_folders]) + 1 if opt.name in parent_folders and not opt.resume else ''
    save_dir = os.path.join(save_parent, f"{opt.name}{save_count}")
    os.makedirs(save_dir, exist_ok=True)

    # Logger
    logfile = os.path.join(save_dir, opt.logfile)
    logger = get_logger(production=False, fixed_logfile=logfile)
    logger.info(f"{vars(opt)}") # log the settings

    # Model Weights, annotations
    model_weight = os.path.realpath(os.path.join(os.path.realpath(os.path.dirname(__file__)),opt.weights))
    annotations = os.path.realpath(os.path.join(os.path.realpath(os.path.dirname(__file__)),opt.annotations))

    # Hyperparameters
    hyp = os.path.realpath(os.path.join(os.path.realpath(os.path.dirname(__file__)),opt.hyp))
    assert os.path.exists(hyp)
    with open(hyp, 'r') as f:
        content = json.load(f)
    for key, value in content.items():
        setattr(C, key, value)
    C.bbox_threshold = opt.conf_thres
    C.nms_dets = opt.iou_thres

    # GPU/CPU (up to 4 gpus)
    assert opt.device in [-1,0,1,2,3], "Please check device argument. Valid values: [-1,0,1,2,3]"
    
    try:
        if opt.device != -1:
            gpus= tf.config.list_physical_devices('GPU')
            gpu = gpus[opt.device]
            tf.config.experimental.set_memory_growth(gpu,True)
            tf.config.experimental.set_visible_devices(gpu, 'GPU')
            
            logger.info(tf.config.list_physical_devices('GPU'))
            logger.info(tf.config.list_logical_devices('GPU'))
            logger.info(f"using GPU {opt.device}- {gpu}")
            device = "/device:GPU:0" # set visible devices alr
        else:
            device = "/CPU:0"
            logger.info("Using /CPU:0")
        
        with tf.device(device):
            test(save_dir, model_weight, logger, annotations, C, save_txt= opt.save_txt, plots= True, v5_metric = False, train = False, sample_rate = opt.sample_rate)
    except Exception as e:
        logger.info(traceback.format_exc())
