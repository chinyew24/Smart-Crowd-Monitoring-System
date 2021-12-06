# -*- coding: UTF-8 -*-
import numpy as np
import cv2
from pathlib import Path
from deepsort_tracking.deep_sort import preprocessing
from deepsort_tracking.deep_sort.nn_matching import NearestNeighborDistanceMetric
from deepsort_tracking.deep_sort.detection import Detection
from deepsort_tracking.tools import generate_detections as gdet
from deepsort_tracking.deep_sort.tracker import Tracker
from keras.models import load_model

from action_model.action_enum import Actions

# Use Deep-sort(Simple Online and Realtime Tracking)
# To track multi-person for multi-person actions recognition

file_path = Path.cwd()
clip_length = 15
max_cosine_distance = 0.3
nn_budget = None
nms_max_overlap = 1.0

model_filename = str(file_path / 'deepsort_tracking/model_data/mars-small128.pb')
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

trk_clr = (255, 255, 255)


def load_action_premodel(model):
    return load_model(model)


def framewise_recognize(pose, pretrained_model):
    walking = 0
    standing = 0
    operate = 0
    falldown = 0
    dangerous = 0
    frame, joints, bboxes, xcenter = pose[0], pose[1], pose[2], pose[3]
    joints_norm_per_frame = np.array(pose[-1])
    # print("pose[-1]:", joints_norm_per_frame)

    if bboxes:
        bboxes = np.array(bboxes)
        features = encoder(frame, bboxes)

        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(bboxes, features)]

        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        tracker.predict()
        tracker.update(detections)

        trk_result = []
        for trk in tracker.tracks:
            if not trk.is_confirmed() or trk.time_since_update > 1:
                continue
            bbox = trk.to_tlwh()
            trk_result.append([bbox[0], bbox[1], bbox[2], bbox[3], trk.track_id])

            # trk_id = 'ID-' + str(trk.track_id)
            # cv.putText(frame, trk_id, (int(bbox[0]), int(bbox[1] - 45)), cv.FONT_HERSHEY_SIMPLEX, 0.8, trk_clr, 3)

        for d in trk_result:
            xmin = int(d[0])
            ymin = int(d[1])
            xmax = int(d[2]) + xmin
            ymax = int(d[3]) + ymin
            # id = int(d[4])
            try:
                # calculate the xcenter distance between track_box and human
                tmp = np.array([abs(i - (xmax + xmin) / 2.) for i in xcenter])
                j = np.argmin(tmp)
                # print("j", j)
            except:
                j = 0

            if joints_norm_per_frame.size > 0:
                joints_norm_single_person = joints_norm_per_frame[j * 36:(j + 1) * 36]
                joints_norm_single_person = np.array(joints_norm_single_person).reshape(-1, 36)
                pred = np.argmax(pretrained_model.predict(joints_norm_single_person, None))
                init_label = Actions(pred).name

                if init_label == "fall_down":
                    falldown += 1
                elif init_label == "walk":
                    walking += 1
                elif init_label == "stand":
                    standing += 1
                else:
                    operate += 1

                cv2.putText(frame, init_label, (xmin + 80, ymin - 45), cv2.FONT_HERSHEY_SIMPLEX, 1, trk_clr, 3)

                if init_label == 'fall_down':
                    cv2.putText(frame, 'WARNING: someone is falling down!', (20, 60), cv2.FONT_HERSHEY_SIMPLEX,
                               1.5, (0, 0, 255), 4)
                    dangerous = 1
                else:
                    dangerous = 0

            cv2.rectangle(frame, (xmin - 10, ymin - 30), (xmax + 10, ymax), trk_clr, 2)

    return falldown, walking, standing, operate, dangerous
