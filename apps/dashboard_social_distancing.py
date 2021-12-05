from re import X
import streamlit as st
from streamlit.proto.Block_pb2 import Block
from socialdistancedetection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import imutils
import cv2
import os
import datetime
import wget
import time
import pandas as pd
import telepot

def app():

    # Telegram
    token = '2126931721:AAGC-pJ7w4f3vgzYC3ETM78wy7ItNaLEBBY'
    chat_id = -770638523
    bot = telepot.Bot(token)

    fps_time = 0    
    frame_count = 0
    chart = 1
    run = 1

    st.title("Social Distancing Detector")

    MIN_CONF = st.slider(
        'Minimum probability to filter Weak Detections', 0.0, 1.0, 0.3)
    NMS_THRESH = st.slider('Non-maxima suppression threshold', 0.0, 1.0, 0.3)

    st.subheader('Select a video')
    option = st.selectbox('Choose your option',
                          ('Demo1', 'Demo2','Demo3','Demo4','Live Detection'))
    maxPerson = st.number_input('Maximum number of person allowed', 5, 50, 20)

    MIN_CONF = float(MIN_CONF)
    NMS_THRESH = float(NMS_THRESH)

    # minimum safe distance (in pixels)
    MIN_DISTANCE = 50

    # file_url = 'https://pjreddie.com/media/files/yolov3.weights'
    # file_name = wget.download(file_url)

    labelsPath = "yolo-coco/coco.names"
    LABELS = open(labelsPath).read().strip().split("\n")

    weightsPath = "yolo-coco/yolov3.weights"
    # weightsPath = file_name
    configPath = "yolo-coco/yolov3.cfg"

    # Load YOLO object detector trained on COCO
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    # Set to use GPU
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    if st.button('Start'):

        if option == "Demo1":
            vs = cv2.VideoCapture("demo_video/Outdoor1.mp4")
        elif option == "Demo2":
            vs = cv2.VideoCapture("demo_video/Outdoor2.mp4")
        elif option == "Demo3":
            vs = cv2.VideoCapture("demo_video/Indoor1.mp4")
        elif option == "Demo4":
            vs = cv2.VideoCapture("demo_video/Indoor2.mp4")
        else:
            vs = cv2.VideoCapture(0)

        image_placeholder = st.empty()
        audio_placeholder = st.empty()

        # create local directory
        new_folder = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        new_path = "output_frame/" + new_folder
        os.makedirs(new_path)

        # loop for each frames from video stream
        while True:
            # read next frame from file
            (grabbed, frame) = vs.read()
            # reach the end if not grabbed
            if not grabbed:
                break

            # resize frame
            frame = imutils.resize(frame, width=700)
            results = detect_people(frame, net, ln, MIN_CONF, NMS_THRESH,
                                    personIdx=LABELS.index("person"))

            # initialize set of violators
            violate = set()

            # at least two people detection
            if len(results) >= 2:

                # extract all centroids from the results and compute the
                # Euclidean distances between all pairs of the centroids
                centroids = np.array([r[2] for r in results])
                D = dist.cdist(centroids, centroids, metric="euclidean")

                # loop over the upper triangular of the distance matrix
                for i in range(0, D.shape[0]):
                    for j in range(i + 1, D.shape[1]):
                        # check to see if the distance between any two
                        # centroid pairs is less than the configured number
                        # of pixels
                        if D[i, j] < MIN_DISTANCE:

                            violate.add(i)
                            violate.add(j)

                # loop over the results
            for (i, (prob, bbox, centroid)) in enumerate(results):
                # extract the bounding box and centroid coordinates, then
                # initialize the color of the annotation
                (startX, startY, endX, endY) = bbox
                (cX, cY) = centroid
                color = (0, 255, 0)

                if i in violate:
                    color = (0, 0, 255)

                # draw bounding box & centroid
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                cv2.circle(frame, (cX, cY), 5, color, 1)

            font = cv2.FONT_HERSHEY_SIMPLEX

            datet = str(datetime.datetime.now())
            frame = cv2.putText(frame, datet, (10, 45), font, 0.5,
                                (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "FPS: %f" % (1.0 / (time.time() - fps_time)),
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 2)
            text = "Social Distancing Violations: {}".format(len(violate))
            cv2.putText(frame, text, (10, 380),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 3)

            text_2 = "Total Count: {}".format(len(results))
            cv2.putText(frame, text_2, (10, 350),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 3)

            # output frame
            display = 1
            if display > 0:

                image_placeholder.image(
                    frame, channels="BGR")
                fps_time = time.time()

            # output chart
            if chart > 0:
                st.subheader('Chart')
                linechart = st.line_chart()
                chart = 0

            # write data into chart
            violate_num = [int(len(violate))]
            crowd_num = [int(len(results))]
            df1 = pd.DataFrame(violate_num, columns=['Number of violations'])
            df2 = pd.DataFrame(crowd_num, columns=['Number of crowd'])
            linechart.add_rows(df1)
            linechart.add_rows(df2)

            # alert sound (run once in a minute)
            curr_time = int(time.time())
            buffer_time = 60 

            if(run == 1):
                last_time = curr_time - (buffer_time+1)


            if(curr_time - last_time > buffer_time):
                if(int(len(results)) > maxPerson):
                    audio_placeholder.write("""
                    <iframe src="https://www.soundjay.com/buttons/sounds/beep-01a.mp3" allow="autoplay" id="audio" style="display: none"></iframe>
                    """, unsafe_allow_html=True)
                    last_time = int(time.time())
                    file_path = new_path + "/frame " + str(frame_count) + ".jpg"
                    cv2.imwrite(file_path,frame)
                    # Dashboard warning notification
                    if(run==1):
                        st.subheader('Notification')
                    st.warning('Crowd Detected \n Time: ' + str(datet) + '\nPerson count: ' + str(len(results)))
                    # Send notification on Telegram
                    bot.sendPhoto(chat_id, photo=open(file_path,'rb'),caption='🚨 *CROWD DETECTED* 🚨 \n Video: ' + str(option) + ' \n Time: ' + str(datet) + ' \n Max person allowed: ' + str(maxPerson) + ' \n Current person count: ' + str(len(results)), parse_mode='Markdown')
                    audio_placeholder.empty()
                    run = 0 

    
            frame_count += 1




        
                

