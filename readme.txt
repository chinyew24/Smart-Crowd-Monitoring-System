Requirements:

You will need the following to run the code:
Python 3.7 or lower (Compatibility issues with Tensorflow on Python 3.8)
OpenCV
CUDA 10.0 (To run with GPU)
	
For running:
streamlit run app.py

Folder Structure:
action_model
action_reg
apps	: contain social distancing and human action recognition main function
deepsort_tracking
demo_video	: contain video sample
images		
models		
output_action_frame
output_frame	: contain output frames (only alerted frames are captured)
scripts
tf_pose
yolo-coco	: contain yolo weights and cfg (can be downloaded from here: https://pjreddie.com/media/files/yolov3.weights)

