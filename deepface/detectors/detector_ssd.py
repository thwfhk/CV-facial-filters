import os
import sys

import tensorflow as tf
import numpy as np
import cv2

from .detector_base import FaceDetector
from deepface.confs.conf import DeepFaceConfs
from deepface.utils.bbox import BoundingBox


class FaceDetectorSSD(FaceDetector):
    NAME = 'detector_ssd'

    def __init__(self, specific_model):
        super(FaceDetectorSSD, self).__init__()
        self.specific_model = specific_model
        graph_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            DeepFaceConfs.get()['detector'][self.specific_model]['frozen_graph']
        )
        self.detector = self._load_graph(graph_path)

        self.tensor_image = self.detector.get_tensor_by_name('prefix/image_tensor:0')
        self.tensor_boxes = self.detector.get_tensor_by_name('prefix/detection_boxes:0')
        self.tensor_score = self.detector.get_tensor_by_name('prefix/detection_scores:0')
        self.tensor_class = self.detector.get_tensor_by_name('prefix/detection_classes:0')

        predictor_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            DeepFaceConfs.get()['detector']['dlib']['landmark_detector']
        )

        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        self.session = tf.Session(graph=self.detector, config=config)

    def _load_graph(self, graph_path):
        # We load the protobuf file from the disk and parse it to retrieve the
        # unserialized graph_def
        with tf.gfile.GFile(graph_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # Then, we import the graph_def into a new Graph and return it
        with tf.Graph().as_default() as graph:
            # The name var will prefix every op/nodes in your graph
            # Since we load everything in a new graph, this is not needed
            tf.import_graph_def(graph_def, name="prefix")
        return graph

    def name(self):
        return 'detector_%s' % self.specific_model

    def detect(self, npimg, resize=(480, 640)):
        """

        :param npimg:
        :param resize: False or tuple
        :return:
        """
        height, width = npimg.shape[:2]
        if not resize:
            infer_img = npimg
        else:
            infer_img = cv2.resize(npimg, resize, cv2.INTER_AREA)   # TODO : Resize or not?

        dets, scores, classes = self.session.run([self.tensor_boxes, self.tensor_score, self.tensor_class], feed_dict={
            self.tensor_image: [infer_img]
        })
        dets, scores = dets[0], scores[0]

        faces = []
        for det, score in zip(dets, scores):
            if score < DeepFaceConfs.get()['detector'][self.specific_model]['score_th']:
                continue

            y = int(max(det[0], 0) * height)
            x = int(max(det[1], 0) * width)
            h = int((det[2] - det[0]) * height)
            w = int((det[3] - det[1]) * width)

            if w <= 1 or h <= 1:
                continue

            bbox = BoundingBox(x, y, w, h, score)

            # find landmark

            # loop over the 68 facial landmarks and convert them
            # to a 2-tuple of (x, y)-coordinates

            faces.append((bbox.x, bbox.y, bbox.x+bbox.w, bbox.y+bbox.h, bbox.score))

        return faces


class FaceDetectorSSDInceptionV2(FaceDetectorSSD):
    def __init__(self):
        super(FaceDetectorSSDInceptionV2, self).__init__('ssd_inception_v2')


class FaceDetectorSSDMobilenetV2(FaceDetectorSSD):
    def __init__(self):
        super(FaceDetectorSSDMobilenetV2, self).__init__('ssd_mobilenet_v2')
