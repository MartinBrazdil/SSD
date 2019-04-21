import json
import os
from collections import OrderedDict
from time import time
from typing import List
from PIL import Image

import cv2
import numpy as np


class Segmentor:
    def get_segm_thr(self):
        return self.segm_thr

    def set_segm_thr(self, x):
        self.segm_thr = x

    def get_open_kernel(self):
        return self.open_kernel

    def set_open_kernel(self, x):
        self.open_kernel = np.ones([x, x])

    def get_close_kernel(self):
        return self.close_kernel

    def set_close_kernel(self, x):
        self.close_kernel = np.ones([x, x])

    def get_blur_radius(self):
        return self.blur_radius

    def set_blur_radius(self, x):
        self.blur_radius = x + 1 if x % 2 == 0 else x

    def get_object_id(self):
        return self.object_id

    def set_object_id(self, x):
        self.object_id = x

    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.cap = cv2.VideoCapture('/dev/video0')
        # self.cap = cv2.VideoCapture('rtsp://192.168.96.138/8000')

        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")

        self.font_color = (255, 0, 0)

        self.set_segm_thr(66)
        self.set_blur_radius(6)
        self.set_open_kernel(19)
        self.set_close_kernel(6)
        self.object_id = 0

        self.thresh = None
        self.morphed = None
        self.obj_cnt = None
        self.markers = None

    def read_frame(self):
        while True:
            _, frame = self.cap.read()
            yield frame

            key = cv2.waitKey(1)
            if key == 27:
                break
            elif key == 32:
                self.snapshot()

        self.cap.release()
        cv2.destroyAllWindows()

    def find_objects(self, frame):
            self.frame = cv2.resize(frame, None, fx=1.0, fy=1.0, interpolation=cv2.INTER_AREA)
            self.grey = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            self.blurred = cv2.medianBlur(self.grey, self.blur_radius)
            self.morphed = self.blurred
            self.morphed = cv2.morphologyEx(self.morphed, cv2.MORPH_OPEN, self.open_kernel)
            self.morphed = cv2.morphologyEx(self.morphed, cv2.MORPH_CLOSE, self.close_kernel)
            ret, self.thresh = cv2.threshold(self.morphed, int(self.segm_thr), 255, cv2.THRESH_BINARY_INV)
            self.obj_cnt, self.markers = cv2.connectedComponents(self.thresh)

            self.bboxes = []
            for i in range(1, self.obj_cnt):
                flat_markers = self.markers.flatten()
                select = np.where(flat_markers == i)[0]
                mask = np.zeros(len(flat_markers), dtype='uint8')
                mask[select] = 1
                mask = mask.reshape(self.markers.shape)
                contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) == 1:
                    self.bboxes.append(cv2.boundingRect(contours[0]))

            self.imshow()

    def snapshot(self):
        timestamp = time()
        seconds = str(timestamp).split('.')[0]
        snapshot_folder = os.path.join(self.data_folder, seconds)
        os.makedirs(snapshot_folder)
        anns = OrderedDict()
        for i, bbox in enumerate(self.bboxes):
            anns[i] = bbox
        with open(os.path.join(snapshot_folder, 'image.jpg'), 'w') as imf:
            Image.fromarray(self.frame).save(imf)
        with open(os.path.join(snapshot_folder, 'anns.json'), 'w') as af:
            json.dump(anns, af)

    def imshow(self, _initialize=[True]):
        if _initialize[0]:
            def _init_window(name, scale, wh=(320, 240)):
                cv2.namedWindow(name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(name, scale * wh[0], scale * wh[1])
            _init_window('Frame', 2)
            _init_window('Blurred', 2)
            cv2.createTrackbar('Blur radius', 'Blurred', self.blur_radius, 100, self.set_blur_radius)
            _init_window('Thresh', 2)
            cv2.createTrackbar('Close kernel', 'Thresh', self.close_kernel.shape[0], 100, self.set_close_kernel)
            cv2.createTrackbar('Open kernel', 'Thresh', self.open_kernel.shape[0], 100, self.set_open_kernel)
            cv2.createTrackbar('Threshold', 'Thresh', self.segm_thr, 255, self.set_segm_thr)
            _init_window('Morphed', 2)
            _init_window('Markers', 2)
            cv2.createTrackbar('Object ID', 'Markers', 0, 1, self.set_object_id)
            _initialize[0] = False

        frame = self.frame.astype('float')
        for i, (x, y, w, h) in enumerate(self.bboxes):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, str(i), (x + w//2, y + h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, self.font_color, 2)
        cv2.imshow('Frame', frame)
        cv2.imshow('Blurred', self.blurred)
        cv2.imshow('Thresh', self.thresh)
        cv2.imshow('Morphed', self.morphed)
        markers = 255. * ((self.markers + 1.) / self.obj_cnt)
        markers = np.stack([markers]*3, 2).astype('float')
        markers = self.write_column(markers, [self.obj_cnt])
        cv2.imshow('Markers', markers.astype('uint8'))

    def write_column(self, img, vals=List):
        for i, v in enumerate(vals):
            cv2.putText(img, str(v), ((i+1)*50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, self.font_color, 2)
        return img


if __name__ == '__main__':
    bboxes = []
    segm = Segmentor('/host_home/projects/data/unit/')
    for frame in segm.read_frame():
        new_bboxes = segm.find_objects(frame)
