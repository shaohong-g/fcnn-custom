import numpy as np
import os, sys, json
import contextlib
from tqdm import tqdm
import tensorflow as tf

def get_model_memory_usage(batch_size, model):
    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        out_shape = l.output_shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes


def save_or_load_model(model, folder, model_weights_file, optimizer_file, state = "save", optimizer = None):
    """ Save or Load Model Weights
    https://stackoverflow.com/questions/49503748/save-and-load-model-optimizer-state/49504376

    Parameters
    ----------
    model : Model Object
        Intended model
    folder : string
        Experiment folder to be saved for e.g. exp1, exp2
    model_weights_file : string
        model_weight file name
    optimizer_file : string
        optimizer_weight file name
    state : string
        Save or load model (Default = save)
    state2 : string
        train or test weights (Default = train)
    optimizer : Optimizer Object
        Optimizer object to set weights (Default = None)
    """

    os.makedirs(folder, exist_ok=True)
    model_weights_file = os.path.join(folder, model_weights_file)
    

    if state == "save":
        assert len(model) == 3, "RPN, classifer and overall model"
        assert len(optimizer_file) == 2, "optimizers for two models"
        model[2].save_weights(model_weights_file)

        for i in range(2):
            optimizer_file[i] = os.path.join(folder, optimizer_file[i])
            np.save(optimizer_file[i], model[i].optimizer.get_weights())

    elif state == "load":
        optimizer_file = os.path.join(folder, optimizer_file)
        assert os.path.exists(model_weights_file), f"{model_weights_file} does not exists!"
        assert os.path.exists(optimizer_file), f"{optimizer_file} does not exists!"
        
        opt_weights = np.load(optimizer_file, allow_pickle=True)

        grad_vars = model.trainable_weights
        zero_grads = [tf.zeros_like(w) for w in grad_vars]
        optimizer.apply_gradients(zip(zero_grads, grad_vars))

        # Set the weights 
        optimizer.set_weights(opt_weights)
        model.load_weights(model_weights_file, by_name=True)


def calculate_iou(ground_truth, pred, only_iou = True):
    """Calculate intersection over union metric
    Reason for +1 dimension for area: https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation

    Parameters
    ----------
    ground_truth : list
        [x1, y1, x2, y2]
    pred : list
        [x1, y1, x2, y2]
    only_iou : bool, optional
        Determine return results. default = True

    Returns
    -------
    iou if only_iou else iou,ix1,iy1,ix2,iy2
    """
    # coordinates of the area of intersection.
    ix1 = np.maximum(ground_truth[0], pred[0])
    iy1 = np.maximum(ground_truth[1], pred[1])
    ix2 = np.minimum(ground_truth[2], pred[2])
    iy2 = np.minimum(ground_truth[3], pred[3])
    
    # Intersection height and width. 0 if not overlap
    i_height = np.maximum(iy2 - iy1 + 1, np.array(0.))
    i_width = np.maximum(ix2 - ix1 + 1, np.array(0.))
    area_of_intersection = i_height * i_width
    
    # Ground Truth dimensions.
    gt_height = ground_truth[3] - ground_truth[1] + 1
    gt_width = ground_truth[2] - ground_truth[0] + 1

    # Prediction dimensions.
    pd_height = pred[3] - pred[1] + 1
    pd_width = pred[2] - pred[0] + 1
    
    area_of_union = gt_height * gt_width + pd_height * pd_width - area_of_intersection
    
    iou = area_of_intersection / area_of_union

    # print(gt_height * gt_width, pd_height * pd_width, area_of_intersection, area_of_union, iou)
    return iou if only_iou else (iou,ix1,iy1,ix2,iy2)

def get_ious(ground_truth_arr, pred):
    result = []
    for i in range(len(ground_truth_arr)):
        iou = calculate_iou(np.array([ground_truth_arr[i][1], ground_truth_arr[i][2],ground_truth_arr[i][3],ground_truth_arr[i][4]]).astype(np.float), pred)
        result.append(iou)
    result = np.array(result)

    return result, np.argsort(-result) #highest to lowest index


def addLoggingLevel(levelName, levelNum, methodName=None):
    """
    Comprehensively adds a new logging level to the `logging` module and the
    currently configured logging class.

    `levelName` becomes an attribute of the `logging` module with the value
    `levelNum`. `methodName` becomes a convenience method for both `logging`
    itself and the class returned by `logging.getLoggerClass()` (usually just
    `logging.Logger`). If `methodName` is not specified, `levelName.lower()` is
    used.

    To avoid accidental clobberings of existing attributes, this method will
    raise an `AttributeError` if the level name is already an attribute of the
    `logging` module or if the method name is already present 

    Example
    -------
    >>> addLoggingLevel('TRACE', logging.DEBUG - 5)
    >>> logging.getLogger(__name__).setLevel("TRACE")
    >>> logging.getLogger(__name__).trace('that worked')
    >>> logging.trace('so did this')
    >>> logging.TRACE
    5

    """
    import logging
    if not methodName:
        methodName = levelName.lower()

    if hasattr(logging, levelName):
       raise AttributeError('{} already defined in logging module'.format(levelName))
    if hasattr(logging, methodName):
       raise AttributeError('{} already defined in logging module'.format(methodName))
    if hasattr(logging.getLoggerClass(), methodName):
       raise AttributeError('{} already defined in logger class'.format(methodName))

    # This method was inspired by the answers to Stack Overflow post
    # http://stackoverflow.com/q/2183233/2988730, especially
    # http://stackoverflow.com/a/13638084/2988730
    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(levelNum):
            self._log(levelNum, message, args, **kwargs)
    def logToRoot(message, *args, **kwargs):
        logging.log(levelNum, message, *args, **kwargs)

    logging.addLevelName(levelNum, levelName)
    setattr(logging, levelName, levelNum)
    setattr(logging.getLoggerClass(), methodName, logForLevel)
    setattr(logging, methodName, logToRoot)

class DummyTqdmFile(object):
    """ Dummy file-like that will write to tqdm
    https://github.com/tqdm/tqdm/issues/313
    """
    file = None

    def __init__(self, file):
        self.file = file

    def write(self, x):
        # Avoid print() second call (useless \n)
        if len(x.rstrip()) > 0:
            tqdm.write(x, file=self.file, end='')

    def flush(self):
        return getattr(self.file, "flush", lambda: None)()

@contextlib.contextmanager
def std_out_err_redirect_tqdm(log):

    try:
        original_stream = log.handlers[0].stream
        log.handlers[0].stream = DummyTqdmFile( original_stream )
        yield original_stream
    except Exception as exc:
        raise exc
    finally:
        log.handlers[0].stream = original_stream

def get_logger(production=False, fixed_logfile=f"./log/train.log"):
    """Create a logger object that writes to both stdout and a file.

    https://www.machinelearningplus.com/python/python-logging-guide/
    https://stackoverflow.com/questions/13733552/logger-configuration-to-log-to-file-and-print-to-stdout
    https://docs.python.org/3/howto/logging.html

    Parameters
    ----------
    production : bool, optional (default=False)
        Set stricter level if production is True.

    fixed_logfile: bool, optional (default=True)
        If True, log to '/tmp/output.log'. If False, log to '/tmp/{file}.log'.

    Returns
    -------
    logger
        A logger object.
    """
    import logging, inspect, os

    # Get file name
    try:
        file_name = os.path.basename(inspect.stack()[1].filename)
    except Exception as e:
        file_name = os.path.basename(__file__)
    base_name = os.path.splitext(file_name)[0] 

    # Create logger
    logger = logging.getLogger(base_name)
    # logger.addLoggingLevel('TQDM', logging.DEBUG - 5)
    
    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(fixed_logfile) if fixed_logfile else logging.FileHandler(f"/tmp/{base_name}.log")

    # Set level
    if production:
        logger.setLevel(logging.INFO)
        c_handler.setLevel(logging.WARNING)
        f_handler.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.DEBUG)
        c_handler.setLevel(logging.DEBUG)
        f_handler.setLevel(logging.DEBUG)

    # Create formatters and add to handlers
    c_format = logging.Formatter('%(asctime)s::%(name)s::%(levelname)s - %(message)s', datefmt='%m-%d %H:%M:%S')
    f_format = logging.Formatter('%(asctime)s::%(name)s::%(funcName)s::%(lineno)d::%(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

if __name__ == "__main__": # Run at cwd: custom-fcnn/
    sys.path.insert(0, os.getcwd())
    from config.config import config as C
    from model.model import get_faster_rcnn_model

    # Get memory usage for models
    model_rpn, model_classifier, model_all = get_faster_rcnn_model(C = C, save_plot = False, base_SOTA = "VGG16", start = 0, end = None, training = True)
    print(get_model_memory_usage(batch_size=1, model = model_rpn))
    print(get_model_memory_usage(batch_size=1, model = model_classifier))
    print(get_model_memory_usage(batch_size=1, model = model_all))


