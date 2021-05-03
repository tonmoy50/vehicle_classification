#Importing Libraries
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
#Importing OpenCV Library for Image Handling and Manipulation
import cv2
#For Object Detection Handling
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

class ModelEvaluate():
    '''
    A Class to Evaluate a given Model on given video data

    ...
    Attributes:
        model_path : string , path to the ssd mobilenet checkpoint/frozen graph
        video_path : string , path to the video file
        label_path : string , Path to the label file(e.g. .pbtxt file)
        num_class : integer , Total number of classes that the model was trained on


    '''

    model_path : str
    video_path : str
    label_path : str
    num_class : int

    def __init__(self, model_path, video_path, label_path, num_class):
        '''
        Instantiate the class with given parameters
        '''

        #Assiging Values to the model attributes
        self.model_path = model_path
        self.video_path = video_path
        self.label_path = label_path
        self.num_class = num_class

        #For system path issues
        sys.path.append("..")
    
    def load_image_into_numpy_array(self, image):
        '''
        Converts Given OpenCV loaded image to numpy array and return the array

        Parameters:
            image : OpenCV Image
        Returns: 
            np.array
        
        '''
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
                (im_height, im_width, 3)).astype(np.uint8
            )
    
    def run(self):
        '''
        Runs The model in tensorflow session to evaluate using given video and shows the real time detection
        '''

        cap = cv2.VideoCapture(self.video_path) #Captures the image from given path

        #Loading the paths of model, label and setting number of class
        PATH_TO_GRAPH = self.model_path 
        PATH_TO_LABELS = self.label_path
        NUM_CLASSES = self.num_class

        detection_graph = tf.compat.v1.Graph() #Definig default tensorflow graph
        #Loading the model checkpoint/frozen graph into the default graph
        with detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(PATH_TO_GRAPH, 'rb') as fid:
                serialized_graph = fid.read() #Reading the file
                od_graph_def.ParseFromString(serialized_graph) #Serializing
                tf.import_graph_def(od_graph_def, name='') #Setting graph from the serialized object

        #Load labels data and mapping it for usage
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)

        #Starting of the detection phase
        with detection_graph.as_default():
            with tf.compat.v1.Session(graph=detection_graph) as sess:
                while True:
                    ret, image_np = cap.read() #Read images from video
                    image_np_expanded = np.expand_dims(image_np, axis=0) #Expanding image dimension as per model requirements
                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0') #Getting Input Layer
                    boxes = detection_graph.get_tensor_by_name('detection_boxes:0') #Detection boxes represents where the image has been detected
                    scores = detection_graph.get_tensor_by_name('detection_scores:0') #This determines the probabalities score for each class
                    classes = detection_graph.get_tensor_by_name('detection_classes:0') #This determines the class labels 
                    num_detections = detection_graph.get_tensor_by_name('num_detections:0') #This determined number of detection per image
                    # Running Detection
                    (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})
                    # Visualizing the class label and detection box on the image
                    #The object detection library of tensorflow has been used here
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8)

                    cv2.imshow('object detection', cv2.resize(image_np, (800,600))) #Showing image with detection box and class label on them
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        break
