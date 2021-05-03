from .model_evaluation import ModelEvaluate


model_path = './frozen_inference_graph.pb'
label_path = './label_map.pbtxt'
video_path = 'vidtest.mkv'
num_class = 18

obj = ModelEvaluate(model_path, video_path, label_path, num_class)
obj.run()