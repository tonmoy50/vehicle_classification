# Vehicle Classification
A simple example of vehicle classification model evaluation using ssd mobilenet. Also, I have added functionality to convert a frozen graph file to tflite for using the model in arm architecture devices such as android, ios, raspberry pi, jetson nano etc.

## Setup
The requirements.txt file all the dependencies.
Setup an python virtual environment and from the root directory of the repository run,
`pip install -r requirements.txt`
This would install all the necessary packages to your environment for running all the files.

## Run

To directly evaluate go to `src/test.py` and change the following variable value,
```
model_path = "Your Model Path Goes Here"
label_path = "Your Label Path Goes Here"
video_path = "Your Video Path Goes Here"
num_class = "Your Class Value Goes Here" //Integer Value
```
Then execute the file
