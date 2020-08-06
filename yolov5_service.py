import random

import cv2
import numpy as np
import torch

from yolov5_models import attempt_load
from yolov5_service_util import xyxy2xywh, non_max_suppression, scale_coords, plot_one_box, letterbox, select_device, \
    time_synchronized, check_img_size


class BoundingBox:
    left: int
    right: int
    top: int
    bottom: int
    x: int
    y: int
    width: int
    height: int

    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h


class Detection:
    xyxy: object
    xywh: object
    bbox: BoundingBox
    confidence: float
    category: int
    category_name: str

    def __init__(self, det, original_shape, names):
        *xyxy, conf, cls = det
        self.xyxy = xyxy
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / original_shape).view(-1).tolist()
        self.xywh = xywh
        self.bbox = BoundingBox(xywh[0], xywh[1], xywh[2], xywh[3])
        self.confidence = conf
        self.category = cls
        self.category_name = names[int(self.category)]


class YoloV5Service:

    def __init__(self, model_path, img_size=640, conf_threshold=0.4, iou_threshold=0.5, device=''):
        self.__set_hyper_parameters(img_size, conf_threshold, iou_threshold, device)
        self.device = select_device(self.device)
        self.model = attempt_load(model_path, map_location=self.device)  # load FP32 model
        self.img_size = check_img_size(self.img_size, s=self.model.stride.max())
        self.half = self.device.type != 'cpu'

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

    def __initialize_model(self, model_path):
        pass

    def __set_hyper_parameters(self, img_size, conf_threshold, iou_threshold, device):
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.augment = False
        self.classes = False
        self.agnostic_nms = False
        self.view_img = True

    def detect(self, img_origin):

        img = self._pre_process_image(img_origin)

        t1 = time_synchronized()
        preds = self.model(img, augment=self.augment)[0]

        # Apply NMS
        preds_after_nms = non_max_suppression(preds, self.conf_threshold, self.iou_threshold, classes=self.classes,
                                              agnostic=self.agnostic_nms)
        t2 = time_synchronized()

        dets = preds_after_nms[0]
        gn = torch.tensor(img_origin.shape)[[1, 0, 1, 0]]
        s = ''
        s += '%gx%g ' % img.shape[2:]  # print string
        detections = []
        if dets is not None and len(dets):
            # Rescale boxes from img_size to im0 size
            dets[:, :4] = scale_coords(img.shape[2:], dets[:, :4], img_origin.shape).round()

            # Print results
            for c in dets[:, -1].unique():
                n = (dets[:, -1] == c).sum()  # detections per class
                s += '%g %ss, ' % (n, self.names[int(c)])  # add to string


            for det in dets:
                detection = Detection(det, gn, self.names)
                detections.append(detection)

        return detections

    def detect_image_file(self, image_file_path):
        img_origin = cv2.imread(image_file_path)
        return self.detect(img_origin)

    def _pre_process_image(self, img_origin):
        img = letterbox(img_origin, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def visualize_detection(self, image, detections):
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]

        for detection in detections:
            xyxy = detection.xyxy
            category = detection.category
            conf = detection.confidence
            label = '%s %.2f' % (self.names[int(category)], conf)
            plot_one_box(xyxy, image, label=label, color=colors[int(category)], line_thickness=3)

        cv2.imshow("", image)
        if cv2.waitKey(0) == ord('q'):  # q to quit
            raise StopIteration


if __name__ == "__main__":
    yolov5_service = YoloV5Service('D:/Data/model/yolov5/yolov5m.pt')

    image_file_path = 'D:/Media/Image/img/2.jpg'
    image = cv2.imread(image_file_path)
    detections = yolov5_service.detect(image)
    yolov5_service.visualize_detection(image, detections)
    print('hello')
