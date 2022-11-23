import os
import time
import cv2
import json
import argparse
import pandas as pd
from IPython.display import display


##############################
# Get data
##############################

def download_data(dataset = "open-images-v6", split = "train", label_types=["detections"], classes=["Cat", "Dog"], data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset"), max_samples = None, seed=51):
    """ Download data from fiftyone api
    
    Refer to https://storage.googleapis.com/openimages/web/download.html for more details
    """

    import fiftyone as fo
    import fiftyone.zoo as foz

    start = time.time()
    fo.config.dataset_zoo_dir = data_dir # Set to current folder if data_dir is None

    download = fo.zoo.load_zoo_dataset(
        dataset,
        split=split,
        label_types=label_types,
        classes=classes,
        max_samples = max_samples,
        seed=seed
    )

    print(f"Dataset- {dataset}:{split}:{classes}:{max_samples}:{seed} Downloaded! {time.time() - start:.2f}s")
 
def process_openimages_format(model_type="yolo", main_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset"), split = ["train", "validation", "test"], classes = []):
    """ Process fiftyone OpenImages to suit yolo/fcnn (custom)
    - label format: XMin, XMax, YMin, YMax (relative to image size)

    Parameters
    ----------
    model_type : str, optional
        Process dataset to suit the model input (Defaults = r"yolo")
    main_dir : str, optional
        Directory of where the data files are stored (Defaults = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset"))
    split : list, optional
        List of splits involved (Defaults = ["train", "validation", "test"])
    classes : list, optional
        Labels classes. (Defaults = [])
    """
    import glob, time
    from tqdm import tqdm
    start = time.time()

    assert model_type in ['yolo', 'fcnn-custom'], f"Processing is not valid for model_type: {model_type}"
    assert os.path.exists(main_dir), "Please download the required dataset."
    assert len(classes) > 0

    for each_split in split:
        split_start = time.time()
        label_dir = os.path.join(main_dir, each_split, "labels")
        o_image_dir = os.path.join(main_dir, each_split, "data")
        image_dir = os.path.join(main_dir, each_split, "images")
        class_file = os.path.join( main_dir, each_split, "metadata", "classes.csv")
        # print(label_dir, o_image_dir, image_dir, class_file, sep='\n')
        assert os.path.exists(label_dir) and os.path.exists(class_file), "Check that all required folders exist."
        assert os.path.exists(o_image_dir) or os.path.exists(image_dir), "Check that all required folders exist."

        if os.path.exists(o_image_dir): os.rename(o_image_dir, image_dir) # rename folder to suit yolo
        # os.makedirs(label_dir, exist_ok=True) # create label folder if not exists


        # Get class labels
        print(f"Get class label - {each_split}")
        df_class = pd.read_csv(class_file, header= None).rename(columns={0: "LabelName", 1: "Label"})
        df_class = df_class[df_class["Label"].isin(classes)]
        assert len(df_class) == len(classes), f"Classes not found in {class_file}"

        # Process detection data and merge with class labels
        print(f"Merge label and class df - {each_split}")
        label_file = os.path.join(label_dir, "detections.csv")
        image_list = [os.path.basename(x).split(".")[0] for x in glob.glob(os.path.join(image_dir , "*.jpg"))]
        df = pd.read_csv(label_file)
        image_id_list = df["ImageID"].unique().tolist()
        assert len(set(image_id_list + image_list )) == len(image_id_list), "Some images are not in label file"
        df = df.loc[df["ImageID"].isin(image_list),["ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax"]]
        df = pd.merge(df, df_class, how = "inner", on = "LabelName")

        # Get Labels Statistics (Optional) - Need label and ImageID columns
        print(f"Label Statistics - {each_split}")
        label_count_df = df["Label"].value_counts()
        label_count_normalized_df = df["Label"].value_counts(normalize=True)
        image_count_df = df.groupby("Label")['ImageID'].agg(['nunique'])
        df_stats = pd.concat([label_count_df,label_count_normalized_df, image_count_df], axis = 1).reset_index()
        df_stats = df_stats.set_axis(["class", "label_count", 'label_count_normalized', 'image_count'], axis=1, copy=False)
        df_stats.to_json(os.path.join(main_dir, each_split, 'label_stats.json'), orient='records') #, lines=True
        display(df_stats)

        # Process
        image_txt = [] # compilation of image names for yolo
        if model_type == "yolo":
            labels_txt = {} # bbox for each image for yolo
        elif model_type == "fcnn-custom":
            labels_arr = []

        image_list = [os.path.basename(x) for x in glob.glob(os.path.join(image_dir , "*.jpg"))]
        pbar = tqdm(image_list)
        for image_file in pbar:
            pbar.set_description(f"Processing {each_split}:{image_file}")
            image = image_file.split(".")[0]
            image_name = os.path.join(image_dir, image_file)

            image_txt.append(os.path.realpath(image_name)) # compilation of image names for yolo

            # process labels
            h,w,d = cv2.imread(image_name).shape
            df_image = df[df['ImageID'] == image].copy()

            if model_type == "yolo":
                labels_txt[image] = []
            elif model_type == "fcnn-custom":
                each_label_dict = {"image": image_name, "classes": df_image["Label"].unique().tolist(), "bbox": []}

            for index, row in df_image.iterrows():
                x1 = row['XMin'] * w
                x2 = row['XMax'] * w
                y1 = row['YMin'] * h
                y2 = row['YMax'] * h

                if model_type == "yolo":
                    x_center = ((x2 + x1) / 2) / w
                    y_center = ((y2 + y1) / 2) / h
                    w_bbox = ((x2 - x1) / 2) / w
                    h_bbox = ((y2 - y1) / 2) / h

                    labels_txt[image].append(f"{classes.index(row['Label'])} {x_center} {y_center} {w_bbox} {h_bbox}")
                elif model_type == "fcnn-custom":
                    each_label_dict["bbox"].append([row["ImageID"], x1, y1, x2, y2, row["Label"]])

            if model_type == "fcnn-custom": labels_arr.append(each_label_dict)
        
        # Write file
        with open(os.path.join(main_dir, f"{each_split}.txt"), "w") as f:
            f.write("\n".join(image_txt))

        if model_type == "yolo":
            for key, value in labels_txt.items():
                with open(os.path.join(label_dir, f"{key}.txt"), "w") as f:
                    f.write("\n".join(value))
        elif model_type == "fcnn-custom":
            with open(os.path.join(main_dir, f"{each_split}_labels.json"), 'w') as f:
                json.dump(labels_arr, f , indent=4)

        print(f"Process {each_split}: {len(image_list)} - {(time.time() - split_start):.2f}s")

    print("Done - ALL!", f"{(time.time() - start):.2f}s")

def process_coco_format(model_type="yolo", main_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset"), split = ["train", "validation", "test"], classes = []):
    """ Process fiftyone COCO-2017 to suit yolo/fcnn (custom)
    - label format: XMin, YMin, Width, Height (relative to image size)

    Parameters
    ----------
    model_type : str, optional
        Process dataset to suit the model input (Defaults = r"yolo")
    main_dir : str, optional
        Directory of where the data files are stored (Defaults = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset"))
    split : list, optional
        List of splits involved (Defaults = ["train", "validation", "test"])
    classes : list, optional
        Labels classes. (Defaults = [])
    """
    import glob, time
    from tqdm import tqdm
    start = time.time()


    assert model_type in ['yolo', 'fcnn-custom'], f"Processing is not valid for model_type: {model_type}"
    assert os.path.exists(main_dir), "Please download the required dataset."
    assert len(classes) > 0

    for each_split in split:
        split_start = time.time()
        label_dir = os.path.join(main_dir, each_split, "labels")
        o_image_dir = os.path.join(main_dir, each_split, "data")
        image_dir = os.path.join(main_dir, each_split, "images")
        class_file = os.path.join(main_dir, each_split, "labels.json")
        assert os.path.exists(class_file), "Check that all required folders exist."
        assert os.path.exists(o_image_dir) or os.path.exists(image_dir), "Check that all required folders exist."

        if os.path.exists(o_image_dir): os.rename(o_image_dir, image_dir) # rename folder to suit yolo
        os.makedirs(label_dir, exist_ok=True) # create label folder if not exists


        with open(class_file, 'r') as f:
            content = json.load(f)

        # Class dictionary
        classes_list = list(filter(lambda x: x["name"] in classes,content['categories']))
        assert len(classes_list) == len(classes), "Some classes are not found in label categories"
        classes_dict = {x['id']: x['name'] for x in classes_list}

        # Image dict
        image_dict = {x['id']: x['file_name'] for x in content["images"]}
        image_list = [os.path.basename(x) for x in glob.glob(os.path.join(image_dir , "*.jpg"))]
        assert len(image_dict) == len(image_list), "Images count does not tally with label.json"

        # Annotations of selected classes
        try:
            annotations = list(filter(lambda x: x["category_id"] in classes_dict.keys(),content['annotations']))
        except Exception:
            print(f"No Annotation available for {each_split}")
            image_txt = [os.path.join(image_dir, x) for x in image_list]
            with open(os.path.join(main_dir, f"{each_split}.txt"), "w") as f:
                f.write("\n".join(image_txt))
            continue

        # Process
        image_size = {}
        labels_txt = {}
        image_txt = []
        pbar = tqdm(annotations)
        count = 0
        classes_stats = {x: {"label_count": 0, "image_count": 0} for x in classes_dict.values()}
        for annotation in pbar:
            count +=1
            pbar.set_description(f"Processing {each_split}: {count}/{len(annotations)}: ")
            assert annotation['image_id'] in image_dict.keys() and image_dict[annotation['image_id']] in image_list, "Image error"
            image_file = image_dict[annotation['image_id']]
            image_name = os.path.join(image_dir, image_file)
            class_idx = classes.index(classes_dict[annotation['category_id']])

            if image_file in image_size:
                h,w,d = image_size[image_file]
            else:
                try:
                    h,w,d = cv2.imread(image_name).shape
                except Exception as e:
                    print(f"Fail cv2: {image_name}")
                    continue
                image_size[image_file] = [h,w,d]
                image_txt.append(os.path.realpath(image_name)) # compilation of image names for yolo
                if model_type == 'yolo':
                    labels_txt[image_file] = [] 
                elif model_type == 'fcnn-custom': 
                    labels_txt[image_file] = {"image": image_name, "classes": [class_idx], "bbox": []}
                
                classes_stats[classes_dict[annotation['category_id']]]['image_count'] += 1
            classes_stats[classes_dict[annotation['category_id']]]['label_count'] += 1

            # process labels
            x1 = annotation['bbox'][0]
            y1 = annotation['bbox'][1]
            w_bbox = annotation['bbox'][2]
            h_bbox = annotation['bbox'][3]
            x2 = x1 + w_bbox
            y2 = y1 + h_bbox

            if model_type == 'yolo':
                x_center = ((x2 + x1) / 2) / w
                y_center = ((y2 + y1) / 2) / h
                w_bbox = ((x2 - x1) / 2) / w
                h_bbox = ((y2 - y1) / 2) / h

                labels_txt[image_file].append(f"{class_idx} {x_center} {y_center} {w_bbox} {h_bbox}")
            elif model_type == "fcnn-custom":
                if class_idx not in labels_txt[image_file]["classes"]:
                    labels_txt[image_file]["classes"].append(class_idx)
                labels_txt[image_file]["bbox"].append([[image_file, x1, y1, x2, y2, classes_dict[annotation['category_id']]]])
                
        # Write file
        with open(os.path.join(main_dir, f"{each_split}.txt"), "w") as f:
            f.write("\n".join(image_txt))

        if model_type == "yolo":
            for key, value in labels_txt.items():
                with open(os.path.join(label_dir, f"{key.split('.')[0]}.txt"), "w") as f:
                    f.write("\n".join(value))
        elif model_type == "fcnn-custom":
            labels_arr = list(labels_txt.values())
            with open(os.path.join(main_dir, f"{each_split}_labels.json"), 'w') as f:
                json.dump(labels_arr, f , indent=4)
        
        with open(os.path.join(main_dir, each_split, f"label_stats.json"), 'w') as f:
            json.dump(classes_stats, f , indent=4)

        print(f"Process {each_split}: {len(image_list)} - {(time.time() - split_start):.2f}s")

    print("Done - ALL!", f"{(time.time() - start):.2f}s")


if __name__ == "__main__":
    """
    Sample Runs (Download):
    1. python utilis/dataset.py --download --dataset coco-2017 --classes motorcycle car truck --splits train --max-samples 10000 --save-dir ./dataset
    2. python utilis/dataset.py --download --dataset coco-2017 --classes motorcycle car truck --splits validation test --max-samples 3000 --save-dir ./dataset
    3. python utilis/dataset.py --download --dataset open-images-v6 --classes Motorcycle Car Truck --splits train --max-samples 10000 --save-dir ./dataset
    4. python utilis/dataset.py --download --dataset open-images-v6 --classes Motorcycle Car Truck --splits validation test --max-samples 3000 --save-dir ./dataset

    download_data(dataset = "open-images-v6", split = "train", label_types=["detections"], classes=['Motorcycle', 'Car', 'Truck'], data_dir = None, max_samples = 10000, seed=51)
    download_data(dataset = "coco-2017", split = "train", label_types=["detections"], classes=['motorcycle', 'car', 'truck'], data_dir = None, max_samples = 10000, seed=51)

    Sample Runs (Process):
    1. python utilis/dataset.py --process --dataset open-images-v6 --classes Motorcycle Car Truck --splits validation test --save-dir ./dataset --model yolo
    1. python utilis/dataset.py --process --dataset open-images-v6 --classes Motorcycle Car Truck --save-dir ./dataset --model fcnn-custom
    1. python utilis/dataset.py --process --dataset coco-2017 --classes motorcycle car truck --splits train --save-dir ./dataset --model yolo
    1. python utilis/dataset.py --process --dataset coco-2017 --classes motorcycle car truck --splits train --save-dir ./dataset --model fcnn-custom

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true', help='Action: Download dataset')
    parser.add_argument('--process', action='store_true', help='Action: Process Dataset to suit model criteria')
    parser.add_argument('--save-dir', type=str, default = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset"),  help='Parent directory of where the dataset is saved (relative to cwd)')
    parser.add_argument('--dataset', type=str, default='coco-2017', help='dataset to be downloaded/processed, check fiftyone api for more details') # open-images-v6, coco-2017
    parser.add_argument('--classes', required= True, nargs='+', type=str, help='classes to be downloaded/processed. For e.g. [motorcycle, car, truck]')
    parser.add_argument('--splits', default=['train', 'validation', 'test'], nargs='+', type=str, help='split to be downloaded/processed. For e.g. [train, validation, test]')

    # Download
    parser.add_argument('--max-samples', type=int, default=3000, help='Max samples to be downloaded')
    parser.add_argument('--seed', type=int, default=51, help='seed number for reproductivity')

    # preprocess
    parser.add_argument('--model', type=str, default='fcnn-custom', help='Process format to suit model')

    opt = parser.parse_args()
    assert opt.classes is not None, "Argument classes need to be supplied"
    assert len(set(opt.splits + ['train', 'validation', 'test'])) == 3, "Argument splits only accept the followings: 'train', 'validation', 'test'"

    save_dir = os.path.realpath(opt.save_dir)

    if opt.download:
        os.makedirs(os.path.realpath(opt.save_dir), exist_ok=True)
        print(f"Dataset (Downloading): {opt.dataset}", f"Classes: {opt.classes}", f"Splits: {opt.splits}", f"Max_samples: {opt.max_samples}",f"Save Directory: { save_dir}", f"Seed no.: {opt.seed}", sep='\n')

        # Download dataset
        for split in opt.splits:
            download_data(dataset = opt.dataset, split = split, label_types=["detections"], classes=opt.classes, data_dir = save_dir, max_samples = opt.max_samples, seed=opt.seed)
        
    if opt.process:
        assert os.path.exists(save_dir), "Save_directory does not exists. Please download the dataset first or supply a valid dataset parent directory."
        print(f"Dataset (Proccessing): {opt.dataset}", f"Classes: {opt.classes}", f"Splits: {opt.splits}",f"Save Directory: { save_dir}", sep='\n')

        if opt.dataset == "open-images-v6":
            process_openimages_format(model_type=opt.model, main_dir = os.path.join(save_dir, opt.dataset), split = opt.splits, classes = opt.classes)
        elif opt.dataset == "coco-2017":
            process_coco_format(model_type=opt.model, main_dir = os.path.join(save_dir, opt.dataset), split = opt.splits, classes = opt.classes)