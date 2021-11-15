"""
For detecting the skeletons by using tf-pose estimation model
"""
import time
import cv2

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
# -- Settings
IS_DRAW_FPS = True
fps_time = 0

class SkeletonDetector(object):
    def __init__(self, model=None, image_size=None):
        if model is None:
            model = "mobilenet_v2_large"
        if image_size is None:
            image_size = "432x368"

        models = set({"mobilenet_thin", "mobilenet_v2_large", "cmu"})
        self.model = model if model in models else "mobilenet_thin"
        self.resize_out_ratio = 4.0
        w, h = model_wh(image_size)
        if w > 0 and h > 0:
            e = TfPoseEstimator(get_graph_path(self.model), target_size=(w, h))
        else:
            e = TfPoseEstimator(get_graph_path(self.model), target_size=(432, 368))

        # self.args = args
        self.w, self.h = w, h
        self.e = e
        self.fps_time = time.time()
        self.cnt_image = 0

    def detect(self, image):
        # Inference
        humans = self.e.inference(image, resize_to_default=(self.w > 0 and self.h > 0),
                                  upsample_size=self.resize_out_ratio)
        return humans

    def draw(self, img_disp, humans):
        img_disp = TfPoseEstimator.draw_humans(img_disp, humans, imgcopy=False)
        return img_disp

    def humans_to_skels_list(self, humans, scale_y=None):  # Get skeleton data of (x, y * scale_h) from humans
        if scale_y is None:
            scale_y = self.scale_y
        skeletons = []
        NaN = 0
        for human in humans:
            skeleton = [NaN] * (18 * 2)
            for i, body_part in human.body_parts.items():  # iterate dict
                idx = body_part.part_idx
                skeleton[2 * idx] = body_part.x
                skeleton[2 * idx + 1] = body_part.y
            #print("skeleton: ", skeleton)
            #print("done")
            skeletons.append(skeleton)
        return skeletons, scale_y


def run_tf_pose():
    fps_time = 0
    cam = cv2.VideoCapture("video.avi")
    ret_val, image = cam.read()
    my_detector = SkeletonDetector("mobilenet_v2_large", "656x368")
    while True:
        ret_val, image = cam.read()
        humans = my_detector.detect(image)
        # img_disp = image.copy()
        skeletons, scale_y = my_detector.humans_to_skels_list(humans)
        print(skeletons)
        print(scale_y)
        pose = my_detector.draw(image, humans)
        no_people = len(humans)
        print("no of people:", no_people)
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
        # logger.debug('finished+')

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_tf_pose()
