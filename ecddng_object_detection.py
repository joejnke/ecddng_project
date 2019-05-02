###Note: add the path to the 'object-detection' folder and 'research/slim' folder into PYTHONPATH


import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from object_detection.utils import ops as utils_ops
if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

#Imports from the object detection module.
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


class ecddng_obj_detection():
    """
    run any chosen object detection model on an image
    """
    def __init__(self, PATH_TO_LABELS, MODEL_PATH ='rlc_graph', NUM_CLASSES=3):
        # What model to download.
        self.MODEL_PATH = MODEL_PATH

        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        self.PATH_TO_FROZEN_GRAPH = self.MODEL_PATH + '/frozen_inference_graph.pb'

        # List of the strings that is used to add correct label for each box.
        self.PATH_TO_LABELS = PATH_TO_LABELS #os.path.join('training', 'object-detection.pbtxt')

        self.NUM_CLASSES = NUM_CLASSES#3

        self.detection_graph = None

        self.category_index = label_map_util.create_category_index_from_labelmap(self.PATH_TO_LABELS, use_display_name=True)

    def download_model(self):
        #TODO
        pass

    def load_frozen_tf_model(self):
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_FROZEN_GRAPH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

    def load_image_into_numpy_array(self, image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    def run_inference_for_single_image(self, image, graph):
        with graph.as_default():
            with tf.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                            tensor_name)
                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[0], image.shape[1])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run inference
                output_dict = sess.run(tensor_dict,
                                       feed_dict={image_tensor: np.expand_dims(image, 0)})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict[
                    'detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]

        return output_dict

    def visualize_boxes_and_labels_on_image_array(self, detection_output_dict, image_np, fig_size=(12, 8)):
        _, box_class = vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            detection_output_dict['detection_boxes'],
            detection_output_dict['detection_classes'],
            detection_output_dict['detection_scores'],
            self.category_index,
            instance_masks=detection_output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=8)
        plt.figure(figsize=fig_size)
        image_np = plt.imshow(image_np)
        return image_np, box_class

    def run_inference_for_multiple_image(self):
        #TODO
        pass


def run_ecddng_obj_detection():
    #TODO : test the functions in the ecddng_obj_detection() class
    MODEL_PATH = '/home/kira/cloned/tensorflow object detection API/models/research/object_detection/rlc_graph'
    PATH_TO_LABELS = '/home/kira/cloned/tensorflow object detection API/models/research/object_detection/training/object-detection.pbtxt'
    image_path = '/home/kira/cloned/tensorflow object detection API/models/research/object_detection/test_images/image3.jpg'

    try_ecddng= ecddng_obj_detection(MODEL_PATH=MODEL_PATH, PATH_TO_LABELS=PATH_TO_LABELS,NUM_CLASSES=3)
    try_ecddng.load_frozen_tf_model()

    image_np = try_ecddng.load_image_into_numpy_array(Image.open(image_path))

    output_dict=try_ecddng.run_inference_for_single_image(image=image_np, graph= try_ecddng.detection_graph)
    try_ecddng.visualize_boxes_and_labels_on_image_array(detection_output_dict= output_dict, image_np= image_np)

    plt.show()

    pass


__end__ = '__end__'

if __name__ == '__main__':
    run_ecddng_obj_detection()

    pass