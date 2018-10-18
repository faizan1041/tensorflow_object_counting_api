#----------------------------------------------
#--- Author         : Ahmet Ozlu
#--- Mail           : ahmetozlu93@gmail.com
#--- Date           : 27th January 2018
#----------------------------------------------

# Imports
import tensorflow as tf
import os
# Object detection imports
from utils import backbone
from api import object_counting_api
from utils import label_map_util

# if tf.__version__ < '1.4.0':
#   raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')


# What model to download.
MODEL_NAME = 'inference_graph'
# MODEL_FILE = MODEL_NAME + '.tar.gz'
# DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('training', 'labelmap.pbtxt')

NUM_CLASSES = 2

def count(image):
  input_image = image
  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')

  label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
  categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
  category_index = label_map_util.create_category_index(categories)


  fps = 30 # change it with your input video fps
  width = 626 # change it with your input video width
  height = 360 # change it with your input vide height
  is_color_recognition_enabled = 0

  result = object_counting_api.single_image_object_counting(input_image, detection_graph, category_index, is_color_recognition_enabled, fps, width, height) # targeted objects counting

  return result


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Detect number of objects')
    parser.add_argument("image",
                        metavar="<image>",
                        help="'image'")
    
    args = parser.parse_args()

    count(args.image)