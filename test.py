import os, math, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# tf.config.run_functions_eagerly(True)
print(tf.config.list_physical_devices('GPU'))

from config import config as C

def test();
    

class Evaluation():
    def __init__(self, C, model_name) -> None:
        # Initialize configuration
        assert hasattr(C, 'color_array'), "color_array not found in Object C"
        assert hasattr(C, 'classes'), "classes not found in Object C"
        assert hasattr(C, 'nms_dets'), "nms_dets not found in Object C"
        assert len(C.color_array) == len(C.classes) -1, "Color array and classes must be of a same size"
        self.C = C
        self.classes = C.classes

        if model_name == "fcnn":
            assert hasattr(self.C, 'bbox_threshold'), "bbox_threshold not found in Object C"
            assert hasattr(self.C, 'nms_max_ROIs'), "nms_max_ROIs not found in Object C"
            assert hasattr(self.C, 'IM_SIZE'), "IM_SIZE not found in Object C"
            assert hasattr(self.C, 'preprocess_input_import'), "preprocess_input_import not found in Object C"
            assert hasattr(self.C, 'img_scaling_factor'), "img_scaling_factor not found in Object C"
            assert hasattr(self.C, 'anchor_box_sizes'), "anchor_box_sizes not found in Object C"
            assert hasattr(self.C, 'anchor_box_ratios'), "anchor_box_ratios not found in Object C"
            assert hasattr(self.C, 'rpn_stride'), "rpn_stride not found in Object C"
            assert hasattr(self.C, 'nms_overlap_thresh'), "nms_overlap_thresh not found in Object C"
            assert hasattr(self.C, 'num_rois'), "num_rois not found in Object C"
            assert hasattr(self.C, 'classifier_regr_std'), "classifier_regr_std not found in Object C"
            assert hasattr(self.C, 'model_weights'), "model_weights not found in Object C"
            assert hasattr(self.C, 'img_input_shape'), "img_input_shape not found in Object C"
            assert hasattr(self.C, 'num_anchors'), "num_anchors not found in Object C"
            assert hasattr(self.C, 'roi_input_shape'), "roi_input_shape not found in Object C"
            assert hasattr(self.C, 'pooling_regions'), "pooling_regions not found in Object C"
            model_rpn, model_classifier = self.get_models(name = "fcnn")
            self.model_rpn = model_rpn
            self.model_classifier = model_classifier

    def run_image_offline(self, data):
        from data.data import get_new_img_size, transform_image_vector
        from utilis import check_image, rpn_to_roi, check_image_bbox, process_rois, non_max_suppression_fast
        
        for image in data:
            self.all_dets = []
            image = self.C.image_parent_dir + image
            assert os.path.exists(image), f"Image does not exists - {image}"
            start = time.time()

            # Process Image
            img = cv2.imread(image)
            resized_height, resized_width, multiplier = get_new_img_size(img.shape[0], img.shape[1],  self.C.IM_SIZE)
            img = cv2.resize(img, dsize=(resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
            img = transform_image_vector(img, self.C.preprocess_input_import, self.C.img_scaling_factor)
            img = np.transpose(img, (0, 2, 3, 1))

            bboxes, probs = self.fcnn_get_results(img, rpn_to_roi)

            if len(bboxes) == 0:
                print(f"No bounding box found in {image}")
                continue
                
            self.show_result(image, multiplier, bboxes, probs, non_max_suppression_fast)
            print('Elapsed time = {}'.format(time.time() - start))

    def get_models(self, name):
        if name == "fcnn":
            from model.model import get_faster_rcnn_model, rpn_loss_cls, rpn_loss_regr, class_loss_cls, class_loss_regr
            
            model_rpn, model_classifier = get_faster_rcnn_model(nb_classes = len(self.C.classes), C = self.C, start= 0, end = None, training = False) 

            model_rpn.load_weights(self.C.model_weights, by_name=True) 
            model_rpn.compile(optimizer='sgd', loss='mse')

            model_classifier.load_weights(self.C.model_weights, by_name=True) #C.model_weights
            model_classifier.compile(optimizer='sgd', loss='mse')

            # model_rpn.compile(optimizer='sgd', loss=[rpn_loss_cls(self.C.num_anchors, lambda_rpn_class = self.C.lambda_rpn_class, epsilon = self.C.epsilon), rpn_loss_regr(self.C.num_anchors, lambda_rpn_regr = self.C.lambda_rpn_regr, epsilon = self.C.epsilon)])
            # model_classifier.compile(optimizer='sgd', loss=[class_loss_cls(lambda_cls_class = self.C.lambda_cls_class), class_loss_regr(len(self.classes)-1, lambda_cls_regr = self.C.lambda_cls_regr, epsilon = self.C.epsilon)], metrics={'dense_class_{}'.format(len(self.classes)): 'accuracy'})

            print(F"Loaded weights from: {self.C.model_weights}")
            return model_rpn, model_classifier
        raise Exception(f"{name} is not a valid model!")

    def fcnn_get_results(self, img, rpn_to_roi):
        # First stage detector for RPN
        Y1, Y2, fmap = self.model_rpn.predict_on_batch(img)
        R = rpn_to_roi(Y1, Y2, self.C, use_regr=True)  # (nms_max_ROIs, 4)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        # Second stage detector for classification
        bboxes = {}
        probs = {}
        for jk in range(R.shape[0]//self.C.num_rois + 1):
            ROIs = np.expand_dims(R[self.C.num_rois*jk:self.C.num_rois*(jk+1), :], axis=0) # (1, 300, 4)
            
            if ROIs.shape[1] == 0:
                break

            if jk == R.shape[0]//self.C.num_rois:
                #pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0],self.C.num_rois,curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded
            
            [P_cls, P_regr] = self.model_classifier.predict_on_batch([fmap, ROIs]) # (1, 300, 3) (1, 300, 8)

            # Calculate bboxes coordinates on resized image
            for ii in range(P_cls.shape[1]):
                # Ignore 'bg' class (P < threshold or last index)
                if np.max(P_cls[0, ii, :]) < self.C.bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                    continue
                
                cls_num = np.argmax(P_cls[0, ii, :])
                cls_name = self.classes[cls_num]

                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []

                (x, y, w, h) = ROIs[0, ii, :]

                try:
                    (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                    tx /= self.C.classifier_regr_std[0]
                    ty /= self.C.classifier_regr_std[1]
                    tw /= self.C.classifier_regr_std[2]
                    th /= self.C.classifier_regr_std[3]
                    x, y, w, h = self.apply_regr(x, y, w, h, tx, ty, tw, th)
                except:
                    pass
                bboxes[cls_name].append([self.C.rpn_stride*x, self.C.rpn_stride*y, self.C.rpn_stride*(x+w), self.C.rpn_stride*(y+h)])
                probs[cls_name].append(np.max(P_cls[0, ii, :]))
        return bboxes, probs

    def apply_regr(self, x, y, w, h, tx, ty, tw, th):
        try:
            cx = x + w/2.
            cy = y + h/2.
            cx1 = tx * w + cx
            cy1 = ty * h + cy
            w1 = math.exp(tw) * w
            h1 = math.exp(th) * h
            x1 = cx1 - w1/2.
            y1 = cy1 - h1/2.
            x1 = int(round(x1))
            y1 = int(round(y1))
            w1 = int(round(w1))
            h1 = int(round(h1))
            return x1, y1, w1, h1
        except ValueError:
            return x, y, w, h
        except OverflowError:
            return x, y, w, h
        except Exception as e:
            print(e)
            return x, y, w, h

    def show_result(self, image, multiplier, bboxes, probs, non_max_suppression_fast):
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for key in bboxes.keys(): # each label
            bbox = np.array(bboxes[key])
            new_boxes, new_probs = non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh= self.C.nms_dets)
            
            for jk in range(new_boxes.shape[0]):
                self.all_dets.append((key,100*new_probs[jk]))

                # Calculate real coordinates on original image 
                (x1, y1, x2, y2) = new_boxes[jk,:] 
                real_x1, real_y1, real_x2, real_y2 = int(x1 / multiplier), int(y1 / multiplier), int(x2 / multiplier), int(y2 / multiplier) 
                real_x1 = np.maximum(0, real_x1) # min x1 = 0 (cannot escape outside of image boundary)
                real_y1 = np.maximum(0, real_y1) # min y1 = 0 (cannot escape outside of image boundary)
                real_x2 = np.minimum(img.shape[1], real_x2) # x2 cannot exceed img_width (-1 to get the last element)
                real_y2 = np.minimum(img.shape[0], real_y2) # y2 cannot exceed img_height (-1 to get the last element)

                
                cv2.rectangle(img, (real_x1, real_y1), (real_x2, real_y2), self.C.color_array[self.C.classes.index(key)], 4)
                
                textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
                (retval, baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
                textOrg = (real_x1, real_y1+ retval[1] + 5)

                cv2.rectangle(img, (textOrg[0], textOrg[1]+baseLine), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1]), (0, 0, 0), 1)
                cv2.rectangle(img, (textOrg[0], textOrg[1]+baseLine), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1]), (255, 255, 255), -1)
                cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

        print(os.path.basename(os.path.realpath(image)), self.all_dets)
        # plt.figure(figsize=(10,10))
        # plt.grid()
        cv2.imwrite(f"./logs/images/{os.path.basename(os.path.realpath(image))}", img)
        # plt.imshow(img)
        # plt.show()

C.augment = None
C.anchor_box_sizes = [8, 16, 32, 64, 128, 256, 512] # C.anchor_box_sizes = [128, 256, 512]
C.num_anchors = len(C.anchor_box_sizes) * len(C.anchor_box_ratios)
C.color_array = [(255,0,0), (0,0,255)]
C.bbox_threshold = 0.5
C.nms_max_ROIs = 300
C.nms_dets = 0.2

from data.data import get_data
all_data, multiple_bbox_data, multiple_labels_data = get_data(LABEL_JSON = "./data/labels_processed.json") # 32892 6235 231
test_dir = './data/open-images-v6/test/data/' #"./data/coco-2017/test/data/" #'./data/' #open-images-v6
C.image_parent_dir = test_dir
test_data = [] #x['image'] for x in multiple_bbox_data
for file in os.listdir(test_dir):
    test_data.append(file)

fcnn_eval = Evaluation(C, "fcnn")
print("initalize done!")
fcnn_eval.run_image_offline(test_data[50:60])
print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default='-1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu (-1)')


    parser.add_argument('--download', action='store_true', help='Action: Download dataset')
    parser.add_argument('--process', action='store_true', help='Action: Process Dataset to suit model criteria')
    parser.add_argument('--save-dir', type=str, default = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset"),  help='Parent directory of where the dataset is saved (relative to cwd)')
    parser.add_argument('--dataset', type=str, default='coco-2017', help='dataset to be downloaded/processed, check fiftyone api for more details') # open-images-v6, coco-2017
    parser.add_argument('--classes', required= True, nargs='+', type=str, help='classes to be downloaded/processed. For e.g. [motorcycle, car, truck]')
    parser.add_argument('--splits', default=['train', 'validation', 'test'], nargs='+', type=str, help='split to be downloaded/processed. For e.g. [train, validation, test]')

    opt = parser.parse_args()