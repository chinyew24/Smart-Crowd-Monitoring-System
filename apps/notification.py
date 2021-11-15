import os

import streamlit as st
import cv2
import argparse
import time
import datetime
from tracking_human import framewise_recognize, load_action_premodel
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import requests


def app():
    token = '2126931721:AAGC-pJ7w4f3vgzYC3ETM78wy7ItNaLEBBY'
    method = 'sendPhoto'
    text = " "
    myuserid = -770638523

    st.title('Human Action Recognition Detector')
    st.subheader('Human Activity Recognition using Openpose')

    option = st.selectbox('Model to be use', ('mobilenet_thin', 'cmu', 'mobilenet_v2_large'))
    resolution = st.selectbox('Resolution to be use (Recommends : 432x368 for mobilenet_thin model)',
                              ('432x368', '656x368'))
    camera = st.selectbox('Choose your option',
                          ('Demo1', 'Demo2', 'Live Detection Using Webcam'))

    if st.button('Start'):
        fps_time = 0
        run = 1
        frame_count = 0
        audio_placeholder = st.empty()

        action_classifier = load_action_premodel('action_model/action_test.h5')
        parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
        parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                            help='if provided, resize heatmaps before they are post-processed. default=1.0')
        parser.add_argument('--show-process', type=bool, default=False,
                            help='for debug purpose, if enabled, speed for inference is dropped.')
        args = parser.parse_args()
        # pose_detector = tf_detector.SkeletonDetector(args.model, args.resize)

        if camera == "Demo1":
            cam = cv2.VideoCapture("demo_video/video.avi")
        elif camera == "Demo2":
            cam = cv2.VideoCapture("demo_video/fall.mp4")
        else:
            cam = cv2.VideoCapture(0)

        # cam = cv2.VideoCapture(args.camera)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        w, h = model_wh(resolution)
        if w > 0 and h > 0:
            e = TfPoseEstimator(get_graph_path(option), target_size=(w, h))
        else:
            e = TfPoseEstimator(get_graph_path(option), target_size=(432, 368))

        writer = None
        image_placeholder = st.empty()
        if cam.isOpened() is False:
            print("Error opening video stream or file")

        # create local directory
        new_folder = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        new_path = "output_action_frame/" + new_folder
        os.makedirs(new_path)

        while True:
            (grabbed, image) = cam.read()

            if not grabbed:
                break

            humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

            pose = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

            nice = framewise_recognize(pose, action_classifier)

            datet = str(datetime.datetime.now())
            no_people = len(humans)
            cv2.putText(image,
                        "People: %d" % no_people,
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 2)
            cv2.putText(image,
                        "FPS: %f" % (1.0 / (time.time() - fps_time)),
                        (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

            display = 1
            if display > 0:
                image_placeholder.image(
                    image, caption='Live Social Distancing Monitor Running..!', channels="BGR")
                fps_time = time.time()
            if writer is not None:
                writer.write(image)

            # alert sound (run once in a minute)
            curr_time = int(time.time())
            buffer_time = 30
            if run == 1:
                last_time = curr_time - (buffer_time + 1)
                run = 0

            if (curr_time - last_time > buffer_time):
                if nice:
                    audio_placeholder.write("""
                                <iframe src="https://www.soundjay.com/buttons/sounds/beep-01a.mp3" allow="autoplay" id="audio" style="display: none"></iframe>
                                """, unsafe_allow_html=True)
                    last_time = int(time.time())
                    file_path = new_path + "/frame " + str(frame_count) + ".jpg"
                    cv2.imwrite(file_path, image)
                    files = {'photo': open(file_path, 'rb')}
                    text = 'Fall down detected!!!'
                    textAppen = 'Time: '
                    resp = requests.post(
                        'https://api.telegram.org/bot2126931721:AAGC-pJ7w4f3vgzYC3ETM78wy7ItNaLEBBY/sendPhoto?chat_id=-770638523&caption={0}\n{1}{2}'.format(
                            text, textAppen, datet), files=files)
                    audio_placeholder.empty()
