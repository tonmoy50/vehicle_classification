import tensorflow as tf
from tensorflow.python.platform import gfile
from io import StringIO
import sys

sys.path.append("..")
#Setting Path to the checkpoint/frozen inference graph
PATH_TO_GRAPH = "./frozen_inference_graph.pb"

#Setting input and output layer of the model
input_tensors = ["image_tensor:0"]
output_tensors = ["detection_scores:0"]

detection_graph = tf.compat.v1.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:

        input_tensors = [detection_graph.get_tensor_by_name('image_tensor:0')]
        output_tensors = [detection_graph.get_tensor_by_name('detection_boxes:0')]

        converter = tf.compat.v1.lite.TFLiteConverter.from_session(
            sess, 
            input_tensors,
            output_tensors
            )

        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
        tflite_model = converter.convert()

        open("vehicle_classifier_converted.tflite", "wb").write(tflite_model)
        print("Done")
