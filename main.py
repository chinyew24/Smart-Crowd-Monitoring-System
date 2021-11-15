"""
Main File that runs everything
"""
import numpy as np
import cv2
import argparse
import time
import run_tf_pose as tf_detector
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from tracking_human import framewise_recognize, load_action_premodel
fps_time = 0
action_classifier = load_action_premodel('action_model/framewise_recognition_under_scene.h5')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=str, default="video/video.avi")

    parser.add_argument('--resize', type=str, default='432x368',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    parser.add_argument('--model', type=str, default='mobilenet_thin',
                        help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()

    pose_detector = tf_detector.SkeletonDetector(args.model, args.resize)

    cam = cv2.VideoCapture(args.camera)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))

    if cam.isOpened() is False:
        print("Error opening video stream or file")
    while cam.isOpened():
        ret_val, image = cam.read()
        #humans = pose_detector.detect(image)

        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        #skeletons, scale_y = pose_detector.humans_to_skels_list(humans)

       # image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        #pose = pose_detector.draw(image, humans)

        pose = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        framewise_recognize(pose, action_classifier)

        no_people = len(humans)
        #print("no of people:", no_people)
        cv2.putText(image,
                    "People: %d" % no_people,
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2)
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
