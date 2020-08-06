import random

import time

import cv2
import numpy as np
import torch

from models.experimental import attempt_load, check_img_size
from utils import torch_utils
from utils.utils import scale_coords, xyxy2xywh, plot_one_box
from yolov5_service_util import letterbox, non_max_suppression


class YoloV5Service:

    def __init__(self, model_path, img_size=640, conf_threshold=0.4, iou_threshold=0.5, device=''):
        self.__set_hyper_parameters(img_size, conf_threshold, iou_threshold, device)
        self.device = torch_utils.select_device(self.device)
        self.model = attempt_load(model_path, map_location=self.device)  # load FP32 model
        self.img_size = check_img_size(self.img_size, s=self.model.stride.max())
        self.half = self.device.type != 'cpu'

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]

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

        t1 = torch_utils.time_synchronized()
        pred = self.model(img, augment=self.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_threshold, self.iou_threshold, classes=self.classes,
                                   agnostic=self.agnostic_nms)
        t2 = torch_utils.time_synchronized()

        for i, det in enumerate(pred):
            gn = torch.tensor(img_origin.shape)[[1, 0, 1, 0]]
            s = ''
            s += '%gx%g ' % img.shape[2:]  # print string
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_origin.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, self.names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    print(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    label = '%s %.2f' % (self.names[int(cls)], conf)
                    plot_one_box(xyxy, img_origin, label=label, color=self.colors[int(cls)], line_thickness=3)

        # Stream results
        if self.view_img:
            cv2.imshow("", img_origin)
            if cv2.waitKey(0) == ord('q'):  # q to quit
                raise StopIteration

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



if __name__ == "__main__":
    yolov5_service = YoloV5Service('D:/Data/model/yolov5/yolov5m.pt')
    yolov5_service.detect_image_file('D:/Media/Image/img/zidane.jpg')
    print('hello')
