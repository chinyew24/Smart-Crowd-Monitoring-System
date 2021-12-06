# Smart Crowd Monitoring System
A GUI based web dashboard created using Streamlit.

## Project Description
This project is a Social Distancing & Human Action Recognition Detector implemented in Python with OpenCV and OpenPose. These tools are able to access live video streams from CCTV footage to automatically estimate interpersonal distance, detect crowd and abnormal human action. The goal is to help the community, including TARUC staff and students to ensure ensure social distancing protocol and safety in their workplace.

## Scope of Functionalities
### Social Distancing Detector
• Detect humans in the frame with yolov3

• Estimate number of people who violate social distance

• Create trend chart of social distancing violations

• Send real-time notification if crowd is detected

### Human Action Recognition Detector
• Detect humans in the frame with tensorflow openpose

• Estimate the abnormal behaviour

• Create trend chart of human actions

• Send real-time notification if abnormal behaviour occurs

## Requirements

    You will need the following to run the code:
    Python 3.7 or lower (Compatibility issues with Tensorflow on Python 3.8)
    OpenCV
    CUDA 10.0 (To run with GPU)
    Libraries (requirements.txt) 

    For running:
    streamlit run app.py

## Folder Structure

    action_model
    action_reg
    apps    : contain social distancing and human action recognition main function
    deepsort_tracking
    demo_video    : contain video sample
    images		
    models		
    output_action_frame
    output_frame    : contain output frames (only alerted frames are captured)
    scripts
    tf_pose
    yolo-coco   : contain yolo weights and cfg (can be downloaded from here: https://pjreddie.com/media/files/yolov3.weights)
