Requirements:

You will need the following to run the code:
Python 3.7 or lower (Compatibility issues with Tensorflow on Python 3.8)
OpenCV
CUDA 10.0 , CUDNN 7.4.1 or CUDNN 7.3.0, Tensorflow 1.14.0 (Must same with this version)
	
For running:
streamlit run app.py

Folder Structure:
action_model : contain action training file and dataset
apps	: contain social distancing and human action recognition main function
deepSORT_tracking : contain deepSORT algorithm that used to track each individual person
demo_video	: contain video sample
models		: contain of several pretrained model, mobilenet_thin, mobilenet_v2_large and cmu
output_frame	: contain output frames (only alerted frames are captured)
tf_pose		: OpenPose library
yolo-coco	: contain yolo weights and cfg (can be downloaded from here: https://pjreddie.com/media/files/yolov3.weights)


If there are an error "No module named '_pafprocess' you need to build c++ library for pafprocess"
Please refer to the video below that start from 2.00 - 3.36 to install swig.