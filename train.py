import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import time
import json, yaml
import tensorflow as tf
import numpy as np
from datetime import datetime
from tqdm import tqdm
import pickle
import argparse
import traceback

from keras.optimizers import Adam
from keras import backend as K
# from keras.callbacks import TensorBoard
# from keras.utils import generic_utils

from config.config import config as C
from model.model import get_faster_rcnn_model
from model.losses import rpn_loss_cls, rpn_loss_regr, class_loss_cls, class_loss_regr
from utilis.dataloader import get_data, data_generator
from utilis.fcnn import rpn_to_roi, process_rois
from utilis.general import get_logger, NumpyEncoder, std_out_err_redirect_tqdm, save_or_load_model


def save_results(c, add_arr,save_record, yml_file, json_file):

    if not hasattr(c, 'epoch_left'):
        c.epoch_left = c.num_epochs
    c.epoch_left -= add_arr[0]
    c.train_step = add_arr[1]
    c.best_loss = add_arr[2]

    if add_arr[3] is not None:
        c.best_loss_epoch = add_arr[3]

    save_dict = {i: vars(c)[i] for i in vars(c) if not i.startswith('__')}

    with open(json_file, 'w') as f:
        json.dump(save_record, f, indent=4, cls= NumpyEncoder)
    with open(yml_file, 'w') as f:
        json.dump(save_dict, f, indent=4, cls= NumpyEncoder)

    # with open(yml_file, 'w') as file:
    #     documents = yaml.dump(save_dict, file)
    

def train(C, save_dir = None, logger= None, resume = False):
    logger.info("Setting up Training models and dataloader...")
    # File and directories
    train_file = os.path.realpath(C.train_file)
    validation_file = os.path.realpath(C.validation_file)
    assert os.path.exists(train_file), f"{train_file} does not exist!"
    assert os.path.exists(validation_file), f"{validation_file} does not exist!"
    assert save_dir is not None and os.path.exists(save_dir), "Invalid save_dir"
    assert logger is not None, "Please initialize logger"


    # Setup for training
    training_data = get_data(train_file)
    classes = C.classes

    # Model
    model_rpn, model_classifier, model_all = get_faster_rcnn_model(C = C, save_plot = False, base_SOTA = C.backbone, start = None, end = None, training = True) #, save_plot='model_all' # train last 6 layers
    
    optimizer_rpn = Adam(learning_rate=C.optimizer_lr)
    optimizer_classifier = Adam(learning_rate=C.optimizer_lr)

    if resume:
        save_or_load_model(model = model_rpn, folder = os.path.join(save_dir, "weights/last"), model_weights_file = "model.h5", optimizer_file = "rpn_optimizer.npy", state = "load", optimizer = optimizer_rpn)
        save_or_load_model(model = model_classifier, folder = os.path.join(save_dir, "weights/last"), model_weights_file = "model.h5", optimizer_file = "classifier_optimizer.npy", state = "load", optimizer = optimizer_classifier)

    model_rpn.compile(optimizer=optimizer_rpn, loss=[rpn_loss_cls(C.num_anchors, lambda_rpn_class = C.lambda_rpn_class, epsilon = C.epsilon), rpn_loss_regr(C.num_anchors, lambda_rpn_regr = C.lambda_rpn_regr, epsilon = C.epsilon)])
    model_classifier.compile(optimizer=optimizer_classifier, loss=[class_loss_cls(lambda_cls_class = C.lambda_cls_class), class_loss_regr(len(C.classes)-1, lambda_cls_regr = C.lambda_cls_regr, epsilon = C.epsilon)], metrics={'dense_class_{}'.format(len(C.classes)): 'accuracy'})
    model_all.compile(optimizer='sgd', loss='mae') # just to get the weights for the whole model, no training is involved for this model
    

    # Training
    num_epochs = C.num_epochs if not hasattr(C, 'epoch_left') else C.epoch_left
    epoch_length = C.epoch_length # aka batch size, number of images per epoch
    iter_num = 0 # counter for epoch length
    train_step = 0 if not hasattr(C, 'train_step') else C.train_step
    assert epoch_length <= len(training_data), "Epoch Length must be less than len(training-data)!"

    rpn_accuracy_for_epoch = [] # number of positive samples per image per epoch
    losses_for_epoch = [] 

    best_loss = np.Inf if not hasattr(C, 'best_loss') else C.best_loss
    best_loss_epoch = np.Inf if not hasattr(C, 'best_loss_epoch') else C.best_loss_epoch
    start = time.time() # training start time
    save_record = []
    best_loss_folder = os.path.join(save_dir, "weights/best_batch")
    best_loss_epoch_folder = os.path.join(save_dir, "weights/best_epoch")
    last_loss_folder = os.path.join(save_dir, "weights/last") # change in line 67 as well (resume weights)

    with std_out_err_redirect_tqdm( logger ) as outputstream:
        for i in range(num_epochs):
            
            logger.info(f"Start training epoch from image idx:{train_step}...")
            data_gen_train = data_generator(training_data, C, train_step) # Generator for training data
            
            pbar = tqdm(total=epoch_length, unit='images', file=outputstream, dynamic_ncols=True) #, position=0, leave=True
            losses = np.zeros((epoch_length, 5)) # 4 losses and 1 accuracy for each epoch
            rpn_accuracy_rpn_monitor = [] # number of positive samples per image per batch
            save_record.append({"epoch": i+1, "train": []})
            start_batch = time.time() # start time per batch

            while True:
                ##################################
                # Training
                ##################################
                pbar.set_description(f"Epoch {i + 1}/{num_epochs}-Batch {iter_num + 1}/{epoch_length}-Image {train_step+1}/{len(training_data)}")
                X1, Y1, img_data = next(data_gen_train)

                loss_rpn = model_rpn.train_on_batch(X1, Y1) # return Total_losses, loss_1 (rpn_loss_cls), loss_2 (rpn_reg_loss) ..
                P_rpn = model_rpn.predict_on_batch(X1) # P_rpn[0] = classification (1,x,x,9), P_rpn[1] = regression coordinates (1,x,x,36)
                
                # note: process_rois converts from (x1,y1,x2,y2) to (x,y,w,h) format
                rois = rpn_to_roi(P_rpn[0], P_rpn[1], C, use_regr=True)  # (None, 4) x1,y2,x1,y2
                X2, Y2_1, Y2_2, IouS = process_rois(rois, img_data, C)

                if X2 is None:
                    logger.info(f"Skip- no ROIs, {img_data['image']}")
                    rpn_accuracy_rpn_monitor.append(0) # metric -rpn_count (batch)
                    # losses = np.delete(losses, -1, 0) 
                    train_step += 1
                    continue

                # sampling positive/negative samples # last index represent bg class
                neg_samples = np.where(Y2_1[0, :, -1] == 1) 
                pos_samples = np.where(Y2_1[0, :, -1] == 0)

                neg_samples = neg_samples[0] if len(neg_samples) > 0 else []
                pos_samples = pos_samples[0] if len(pos_samples) > 0 else [] 

                if C.num_rois > 1:
                    if len(pos_samples) < C.num_rois//2:
                        selected_pos_samples = pos_samples.tolist()
                    else:
                        if len(pos_samples) > 0:
                            selected_pos_samples = np.random.choice(pos_samples, C.num_rois//2, replace=False).tolist()
                        else:
                            selected_pos_samples = []
                    try:
                        if len(neg_samples) > 0:
                            selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=False).tolist()
                        else:
                            selected_neg_samples = []
                    except:
                        if len(neg_samples) > 0:
                            selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=True).tolist()
                        else:
                            selected_neg_samples = []
                    sel_samples = selected_pos_samples + selected_neg_samples
                else:
                    # in the extreme case where num_rois = 1, we pick a random pos or neg sample
                    selected_pos_samples = pos_samples.tolist()
                    selected_neg_samples = neg_samples.tolist()
                    if np.random.randint(0, 2):
                        sel_samples = np.random.choice(neg_samples)
                    else:
                        sel_samples = np.random.choice(pos_samples) 

                if len(sel_samples) != C.num_rois: # ignore as it will cause error for model_classifier
                    logger.debug(f"total length: {len(sel_samples)}, pos: {len(selected_pos_samples)}, neg: {len(selected_neg_samples)}")
                    rpn_accuracy_rpn_monitor.append(0)
                    train_step += 1
                    continue

                rpn_accuracy_rpn_monitor.append((len(selected_pos_samples))) # metric -rpn_count (batch)
                loss_class = model_classifier.train_on_batch( [X1, X2[:, sel_samples, :]], [Y2_1[:, sel_samples, :], Y2_2[:, sel_samples, :]])
                

                ##################################
                # Save Record and evaluate
                ##################################
                losses[iter_num, 0], losses[iter_num, 1], losses[iter_num, 2], losses[iter_num, 3] = loss_rpn[1], loss_rpn[2], loss_class[1], loss_class[2]
                losses[iter_num, 4] = loss_class[3] # accuracy
                iter_num += 1
                train_step += 1
                pbar.set_postfix_str(f"pos_samples={len(selected_pos_samples)}, rpn_loss=[{np.mean(losses[:iter_num, 0]):.5f},{np.mean(losses[:iter_num, 1]):.5f}], Detector_loss=[{np.mean(losses[:iter_num, 2]):.5f},{np.mean(losses[:iter_num, 3]):.5f}]")
                pbar.update()

                if iter_num == epoch_length or train_step == len(training_data):
                    class_acc = np.mean(losses[:, 4])
                    loss_rpn_cls = np.mean(losses[:, 0])
                    loss_rpn_regr = np.mean(losses[:, 1])
                    loss_class_cls = np.mean(losses[:, 2])
                    loss_class_regr = np.mean(losses[:, 3])
                    curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                    losses_for_epoch.append(curr_loss)

                    rpn_accuracy_batch = sum(rpn_accuracy_rpn_monitor) / len(rpn_accuracy_rpn_monitor)
                    rpn_accuracy_for_epoch.append(rpn_accuracy_batch)
                    
                    if best_loss > curr_loss:
                        best_loss = curr_loss
                        save_or_load_model([model_rpn, model_classifier, model_all], best_loss_folder, "model.h5", ["rpn_optimizer.npy", "classifier_optimizer.npy"], state = "save", optimizer = None)
                    save_or_load_model([model_rpn, model_classifier, model_all], last_loss_folder, "model.h5", ["rpn_optimizer.npy", "classifier_optimizer.npy"], state = "save", optimizer = None)

                    # save batch data
                    batch_data = {
                        "batch": len(save_record[-1]["train"]) + 1,
                        "loss_rpn_cls": loss_rpn_cls,
                        "loss_rpn_regr": loss_rpn_regr,
                        "loss_class_cls": loss_class_cls,
                        "loss_class_regr": loss_class_regr,
                        "loss_total": curr_loss,
                        "class_acc": class_acc,
                        "rpn_accuracy_batch": rpn_accuracy_batch,
                        "Elapsed_time": f"{time.time() - start_batch:.2f}"
                    }
                    save_record[-1]["train"].append(batch_data)
                    # save_record[-1]["rpn_accuracy_for_epoch"] = rpn_accuracy_for_epoch
                    # save_record[-1]["losses_for_epoch"] = losses_for_epoch
                    save_record[-1]["best_loss"] = best_loss

                    save_results(C, [i, train_step, best_loss, None],save_record, os.path.join(save_dir, "hyp.json"), os.path.join(save_dir, "results.json"))

                    logger.info(json.dumps(batch_data, cls = NumpyEncoder, indent=4))
                    logger.info(f"Best loss: {best_loss}")
                    # logger.debug(f"Memory {train_step}-{tf.config.experimental.get_memory_info('GPU:0')}")


                    losses = np.zeros((epoch_length, 5))
                    rpn_accuracy_rpn_monitor = []
                    iter_num = 0
                    start_batch = time.time()

                    if train_step == len(training_data):
                        break
                    
            train_step = 0
            save_record[-1]["mean_overlapping_bboxes"] = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
            save_record[-1]["loss"] = float(sum(losses_for_epoch)) / len(losses_for_epoch)

            if best_loss_epoch > save_record[-1]["loss"]:
                best_loss_epoch = save_record[-1]["loss"]
                save_or_load_model([model_rpn, model_classifier, model_all], best_loss_epoch_folder, "model.h5", ["rpn_optimizer.npy", "classifier_optimizer.npy"], state = "save", optimizer = None)

            logger.info(f"Epoch {i+1} loss: {save_record[-1]['loss']}")
            logger.info(f"Epoch {i+1} mean_overlapping_bboxes: {save_record[-1]['mean_overlapping_bboxes']}")

            save_results(C, [i + 1, train_step, best_loss, best_loss_epoch],save_record, os.path.join(save_dir, "hyp.json"), os.path.join(save_dir, "results.json"))

    logger.info(f"Training Completed! Elapsed Time: {time.time() - start:.2f}s")

            
            

        



    




if __name__ == "__main__":
    """
    Sample runs:
    1. python train.py --name test --device -1 --logfile train.log
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Resume training as per name')
    parser.add_argument('--name', type=str, default='exp', help='Experiment name')
    parser.add_argument('--device', type=int, default='-1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu (-1)')
    parser.add_argument('--epoch-length', type=int, help='AKA batch size')
    parser.add_argument('--num-epochs', type=int, help='Number of epoches')
    parser.add_argument('--logfile', type=str, default='train.log', help='Log file')
    opt = parser.parse_args()

    # Create save_dir
    save_parent = os.path.join(os.path.dirname(__file__), "runs", "train")
    os.makedirs(save_parent, exist_ok=True)
    parent_folders = next(os.walk(save_parent))[1] # all folders in parent
    save_count = sum([x.startswith(opt.name) for x in parent_folders]) + 1 if opt.name in parent_folders and not opt.resume else ''
    save_dir = os.path.join(save_parent, f"{opt.name}{save_count}")
    os.makedirs(save_dir, exist_ok=True)

    # Logger
    logfile = os.path.join(save_dir, opt.logfile)
    logger = get_logger(production=False, fixed_logfile=logfile)
    logger.info(f"{vars(opt)}") # log the settings

    # Resume
    if opt.resume and os.path.exists(os.path.join(save_dir, "hyp.json")):
        with open(os.path.join(save_dir, "hyp.json"), 'r') as f:
            content = json.load(f)
        for key, value in content.items():
            setattr(C, key, value)
        logger.info(f"Resume config: {os.path.join(save_dir, 'hyp.json')}")
    elif opt.resume:
        raise Exception("This is a complete new run! Cannot resume")

    # epoch length and num epochs
    if opt.num_epochs is not None: C.num_epochs = opt.num_epochs
    if opt.epoch_length is not None: C.epoch_length = opt.epoch_length


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
            train(C, save_dir = save_dir, logger= logger, resume = opt.resume)

    except Exception as e:
        logger.info(traceback.format_exc())
