import os, math, time, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import traceback
# tf.config.run_functions_eagerly(True)
# print(tf.config.list_physical_devices('GPU'))

from config.config import config as C
from model.model import get_faster_rcnn_model
from utilis.general import get_logger
from utilis.dataloader import get_new_img_size, transform_image_vector
from utilis.fcnn import  non_max_suppression_fast, fcnn_get_results
# from metrics.BoundingBox import BoundingBox
# from metrics.BoundingBoxes import BoundingBoxes
# from metrics.utils import *

def get_bboxes(frame, C, model_rpn, model_classifier, classes, save_txt, filename, color_arr):
    resized_height, resized_width, multiplier = get_new_img_size(frame.shape[0], frame.shape[1],  C.IM_SIZE)
    img = cv2.resize(frame, dsize=(resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
    img = transform_image_vector(img, C.preprocess_input_import, C.img_scaling_factor)
    img = np.transpose(img, (0, 2, 3, 1))
    seen = 0

    # cv2 settings
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    text_scale = frame.shape[1] / 1000

    # Fit to models
    bboxes, probs = fcnn_get_results(img, model_rpn, model_classifier, classes, C)

    if len(bboxes) == 0:
        return frame, seen
    
    # NMS and get all detections
    if save_txt: save_txt_list = []

    
    for key in bboxes.keys(): # each label
        bbox = np.array(bboxes[key])
        new_boxes, new_probs = non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh= C.nms_dets)
        for jk in range(new_boxes.shape[0]):
            # Calculate real coordinates on original image 
            (x1, y1, x2, y2) = new_boxes[jk,:] 
            real_x1, real_y1, real_x2, real_y2 = int(x1 / multiplier[1]), int(y1 / multiplier[0]), int(x2 / multiplier[1]), int(y2 / multiplier[0]) 
            real_x1 = np.maximum(0, real_x1) # min x1 = 0 (cannot escape outside of image boundary)
            real_y1 = np.maximum(0, real_y1) # min y1 = 0 (cannot escape outside of image boundary)
            real_x2 = np.minimum(frame.shape[1], real_x2) # x2 cannot exceed frame_width (-1 to get the last element)
            real_y2 = np.minimum(frame.shape[0], real_y2) # y2 cannot exceed frame_height (-1 to get the last element)


            # CV2
            cv2.rectangle(frame, (real_x1,real_y1), (real_x2,real_y2), color_arr[0], 2)
            textLabel = f'{key}: {round(100*new_probs[jk], 2)}%'
            (retval, baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_SIMPLEX,text_scale,1)
            textOrg = (min(real_x1,real_x2) , min(real_y1,real_y2) + retval[1] )
            cv2.rectangle(frame, (min(real_x1,real_x2), min(real_y1,real_y2)), (min(real_x1,real_x2) + retval[0] , min(real_y1,real_y2) + retval[1]+baseLine), color_arr[-1], -1)
            cv2.putText(frame, textLabel, textOrg, cv2.FONT_HERSHEY_SIMPLEX, text_scale, color_arr[-2], 1)

            if save_txt:
                save_txt_list.append(f"{real_x1} {real_y1} {real_x2} {real_y2} {new_probs[jk]} {key} ")
            seen += 1

    if save_txt:
        os.makedirs(os.path.join(save_dir, "plabels"), exist_ok=True)
        with open(os.path.join(save_dir, "plabels", f"{os.path.basename(filename).split('.')[0]}.txt"),'w') as f:
            f.write("\n".join(save_txt_list))

    return frame, seen

    


def test(source, save_dir, model_weight, logger, C, save_txt= None, acceptable_extensions_img = ['.png', '.jpg'], acceptable_extensions_vid=['.mp4', '.avi'], color_arr=[(255,0,0), (0,0,255), (0,255,0), (0, 0, 0), (255,255,255)]):
    """
    # accept video format: avi, mp4
    # accept image format: jpg, png
    # other acceptable format: folder / txt file

    Video Frame Processing : https://github.com/akash-agni/Real-Time-Object-Detection/blob/main/Drone_Human_Detection_Model.py
    color_array = [(255,0,0), (0,0,255), (0,255,0), (0, 0, 0), (255,255,255)] # red, blue, green, black, white
    """
    assert os.path.exists(model_weight)
    assert os.path.exists(source), "Folder/image/video does not exist: {source}"

    # Get models
    model_rpn, model_classifier = get_faster_rcnn_model(C = C, save_plot = False, base_SOTA = C.backbone, start = 0, end = None, training = False)
    model_rpn.load_weights(model_weight, by_name=True) 
    model_classifier.load_weights(model_weight, by_name=True) #C.model_weights

    model_rpn.compile(optimizer='sgd', loss='mse')
    model_classifier.compile(optimizer='sgd', loss='mse')
    logger.info(F"Loaded weights from: {model_weight}")

    # detecting data and classes
    acceptable_extensions = tuple(acceptable_extensions_img + acceptable_extensions_vid)
    if os.path.isdir(source):
        detect_data = list(filter(lambda x: x.endswith(acceptable_extensions), os.listdir(source) ))
        detect_data = [os.path.join(source, x) for x in detect_data]
    elif source.endswith(".txt"): # txt file
        with open(source, 'r') as f:
            detect_data = f.readlines()
            detect_data = [x.strip() for x in detect_data] # remove any trailling spaces
        detect_data = list(filter(lambda x: x.endswith(acceptable_extensions), detect_data ))
    elif source.endswith(acceptable_extensions):
        detect_data = [source]
    else:
        raise Exception(f"Extension is not acceptable. Accepted extensions: {acceptable_extensions}")

    classes = C.classes 
    start = time.time()

    # Test
    logger.info("Start Detecting...")
    total_counter = len(detect_data)
    mean_img = 0
    fps_img = 0
    seen = 0
    for counter, data in enumerate(detect_data):
        if not os.path.exists(data):
            logger.info(f"Image/video does not exists - {data}")
            continue

        out_file = os.path.join(save_dir, os.path.basename(data))
        isimage = True
        if data.endswith(tuple(acceptable_extensions_vid)):
            isimage = False
        
        # Video processing
        if not isimage: 
            cap = cv2.VideoCapture(data)
            assert cap.isOpened()

            # Video stats
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            v_fps = cap.get(cv2.CAP_PROP_FPS)
            tfc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            logger.info(f"Video Stats: {w}w {h}h {v_fps}fps {tfc}frames")

            four_cc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_file, four_cc, v_fps, (w, h))

            count = 0
            total_fps = 0
            mean = 0
            while True:
                count += 1
                start_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    break
                frame, obj = get_bboxes(frame, C, model_rpn, model_classifier, classes, save_txt, data, color_arr)
                fps = 1/np.round(time.time() - start_time, 3)
                total_fps += fps
                
                if count % 10 == 0:
                    mean = round(total_fps / count, 3)

                logger.info(f"Frames {count}/{tfc}: {fps} fps, {mean} mfps, detect: {obj} obj!")
                out.write(frame)
            cap.release()
        else:
            seen += 1
            start_detect = time.time()
            # Process Image
            img_original = cv2.imread(data)
            img, obj = get_bboxes(img_original, C, model_rpn, model_classifier, classes, save_txt, data, color_arr)
            elapsed_time = np.round(time.time() - start_detect, 3)
            fps = round(1/elapsed_time, 3)
            fps_img += fps

            if seen % 10 == 0:
                mean_img = round(fps_img / seen, 3)

            cv2.imwrite(out_file, img)
            logger.info(f"Tested {counter + 1}/{total_counter}: {obj} objects, Elapsed Time: {elapsed_time}s, {fps} fps, {mean_img} mfps, {out_file}")


    logger.info(f"Testing Completed! Elapsed Time: {time.time() - start:.2f}s")
    


if __name__ == "__main__":
    """
    python detect.py --device -1 --name openimages --weights ./runs/train/exp1/model.h5 --hyp ./runs/train/exp1/hyp.json --source ./dataset/open-images-v6/test/images
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-txt', action='store_true', help='Save Labels in txt format')
    parser.add_argument('--name', type=str, default='exp', help='Experiment name')
    parser.add_argument('--source', type=str, required=True, help='Folder/txt file/images/video (relative to this file)')
    parser.add_argument('--device', type=int, default='-1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu (-1)')
    parser.add_argument('--weights', type=str, default='./model.h5', help='File location of where the weight file is saved (relative to this file)')
    parser.add_argument('--hyp', type=str, required=True, help='File location of where the config (hyp.json) file is saved (relative to this file)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Confidence level for classifier')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--logfile', type=str, default='detect.log', help='Log file')
    opt = parser.parse_args()

    # Create save_dir
    save_parent = os.path.join(os.path.dirname(__file__), "runs", "detect")
    os.makedirs(save_parent, exist_ok=True)
    parent_folders = next(os.walk(save_parent))[1] # all folders in parent
    save_count = sum([x.startswith(opt.name) for x in parent_folders]) + 1 if opt.name in parent_folders else ''
    save_dir = os.path.join(save_parent, f"{opt.name}{save_count}")
    os.makedirs(save_dir, exist_ok=True)

    # Logger
    logfile = os.path.join(save_dir, opt.logfile)
    logger = get_logger(production=False, fixed_logfile=logfile)
    logger.info(f"{vars(opt)}") # log the settings

    # Model Weights
    model_weight = os.path.realpath(os.path.join(os.path.realpath(os.path.dirname(__file__)),opt.weights))
    source = os.path.realpath(os.path.join(os.path.realpath(os.path.dirname(__file__)),opt.source))

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
            test(source, save_dir, model_weight, logger, C, save_txt= opt.save_txt, acceptable_extensions_img = ['.png', '.jpg'], acceptable_extensions_vid=['.mp4', '.avi'], color_arr=[(255,0,0), (0,0,255), (0,255,0), (0, 0, 0), (255,255,255)])

    except Exception as e:
        logger.info(traceback.format_exc())