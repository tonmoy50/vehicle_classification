import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile


if __name__ == '__main__':
        
    GRAPH_PB_PATH = './frozen_inference_graph.pb'
    with tf.compat.v1.Session() as sess:
        print("load graph")
        with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
        graph_nodes=[n for n in graph_def.node]
        names = []
        f = open("node_info.txt", "w")
        for t in graph_nodes:
            names.append(t.name)
            f.write(t.name+"\n")
        print("Done")
        f.close()


