import os
import sys
from pathlib import Path
import time
from datetime import datetime
import torch
from celery import Celery, Task

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
# print('1 ROOT:{}'.format(ROOT))
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
# print('2 ROOT:{}'.format(ROOT))

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2, increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


class StreamInfer():
    def __init__(self):
        print('{} # # 02 StreamInfer Start making DetcInfer instance..'.format(datetime.now()))
        super().__init__()

        self.weights=ROOT / 'yolov5s.pt'  # model path or triton URL
        self.source='rtsp://prezzie77:1q2w3e4r5t@183.106.132.155:554/stream1'
        self.data=ROOT / 'data/coco128.yaml'  # dataset.yaml path
        self.imgsz=(640, 640)  # inference size (height, width)
        self.conf_thres=0.25  # confidence threshold
        self.iou_thres=0.45  # NMS IOU threshold
        self.max_det=1000  # maximum detections per image
        self.device=''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.view_img=False  # show results
        self.save_img = True
        self.save_txt=False  # save results to *.txt
        self.save_conf=False  # save confidences in --save-txt labels
        self.save_crop=False  # save cropped prediction boxes
        self.nosave=False  # do not save images/videos
        self.classes=None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms=False  # class-agnostic NMS
        self.augment=False  # augmented inference
        self.visualize=False  # visualize features
        self.update=False  # update all models
        self.project=ROOT / 'runs/detect'  # save results to project/name
        self.name='exp'  # save results to project/name
        self.exist_ok=False  # existing project/name ok, do not increment
        self.line_thickness=3  # bounding box thickness (pixels)
        self.hide_labels=False  # hide labels
        self.hide_conf=False  # hide confidences
        self.half=False  # use FP16 half-precision inference
        self.dnn=False  # use OpenCV DNN for ONNX inference
        self.vid_stride=1  # video frame-rate stride

        # opt 안받으려고 아래 source 주석 후 두줄 추가
        # source = str(source) # source : rtsp://prezzie77:1q2w3e4r5t@183.106.132.155:554/stream2

        self.is_file = Path(self.source).suffix[1:] in (IMG_FORMATS + VID_FORMATS) # is_file : False
        self.is_url = self.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')) # is_url : True
        self.webcam = self.source.isnumeric() or self.source.endswith('.txt') or (self.is_url and not self.is_file) # webcam : True

        # Directories
        self.save_dir = increment_path(Path(self.project) / self.name, exist_ok=self.exist_ok)  # increment run, save_dir : runs/detect/exp36
        (self.save_dir / 'labels' if self.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # # Load model
        # device = select_device(self.device) # device : cpu
        # self.model = DetectMultiBackend(self.weights, device=device, dnn=self.dnn, data=self.data, fp16=self.half)
        # self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        # imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size

        # # Dataloader
        # print('{} # # 03 StreamInfer Start getting rtsp..'.format(datetime.now()))
        # self.dataset = LoadStreams(self.source, img_size=imgsz, stride=self.stride, auto=self.pt, vid_stride=self.vid_stride)
        # bs = len(self.dataset)

        # # Run inference
        # print('{} # # 04 StreamInfer Start loading detection model..'.format(datetime.now()))
        # self.model.warmup(imgsz=(1 if self.pt or self.model.triton else bs, 3, *imgsz))  # warmup
        # self.seen, self.windows, self.dt = 0, [], (Profile(), Profile(), Profile())
        # print('{} # # 05 StreamInfer Detection model is ready!'.format(datetime.now()))

    def load_model_n_get_dataloader(self):
        # Load model
        device = select_device(self.device) # device : cpu
        self.model = DetectMultiBackend(self.weights, device=device, dnn=self.dnn, data=self.data, fp16=self.half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size

        # Dataloader
        print('{} # # 03 StreamInfer Start getting rtsp..'.format(datetime.now()))
        self.dataset = LoadStreams(self.source, img_size=imgsz, stride=self.stride, auto=self.pt, vid_stride=self.vid_stride)
        bs = len(self.dataset)

        # Run inference
        print('{} # # 04 StreamInfer Start loading detection model..'.format(datetime.now()))
        self.model.warmup(imgsz=(1 if self.pt or self.model.triton else bs, 3, *imgsz))  # warmup
        self.seen, self.windows, self.dt = 0, [], (Profile(), Profile(), Profile())
        print('{} # # 05 StreamInfer Detection model is ready!'.format(datetime.now()))

    # @smart_inference_mode()
    def run(self):
        print('{} # # 07 StreamInfer Start model infer!'.format(datetime.now()))

        NumNum = 0
        for path, im, im0s, vid_cap, s in self.dataset:
            NumNum += 1
            print('{} # NumNum : {}'.format(datetime.now(), NumNum))

            # Preprocess image
            with self.dt[0]:
                im = torch.from_numpy(im).to(self.model.device)
                im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with self.dt[1]:
                visualize = increment_path(self.save_dir / Path(path).stem, mkdir=True) if self.visualize else False
                pred = self.model(im, augment=self.augment, visualize=visualize)

            # NMS
            with self.dt[2]:
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
                # print(varname(pred)) # pred : [tensor([[4.92419e+02, 5.03329e+01, 6.40432e+02, 1.47021e+02, 3.21195e-01, 6.00000e+00]])]

            # Process predictions
            for i, det in enumerate(pred):  # per image
                self.seen += 1
                if self.webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), self.dataset.count
                    s += f'{i}: '

                p = Path(p)  # to Path
                save_path = str(self.save_dir / p.name)  # im.jpg
                os.makedirs(save_path, exist_ok=True) # save_path : runs/detect/exp36/stream2
                txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')  # im.txt, 
                # txt_path : runs/detect/exp36/labels/stream2_0
                s += '%gx%g ' % im.shape[2:]  # print string # s : 0: 384x640
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))

                # Stream results
                im0 = annotator.result()

                if len(det):
                    # print('something is detected !')
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # image box label
                    for *xyxy, conf, cls in reversed(det):
                        # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))

            # yield detection result image
            ret, buffer = cv2.imencode('.jpg', im0)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            # concat frame one by one and show result

class BackgroundInferTask(Task):
    name = 'Traninig Task'
    concurrency = 1

    def __init__(self):
        print('{} # # 02 BackgroundInferTask Start making DetcInfer instance..'.format(datetime.now()))
        super().__init__()

        self.weights=ROOT / 'yolov5s.pt'  # model path or triton URL
        self.source='rtsp://prezzie77:1q2w3e4r5t@183.106.132.155:554/stream1'
        self.data=ROOT / 'data/coco128.yaml'  # dataset.yaml path
        self.imgsz=(640, 640)  # inference size (height, width)
        self.conf_thres=0.25  # confidence threshold
        self.iou_thres=0.45  # NMS IOU threshold
        self.max_det=1000  # maximum detections per image
        self.device=''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.view_img=False  # show results
        self.save_img = True
        self.save_txt=False  # save results to *.txt
        self.save_conf=False  # save confidences in --save-txt labels
        self.save_crop=False  # save cropped prediction boxes
        self.nosave=False  # do not save images/videos
        self.classes=None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms=False  # class-agnostic NMS
        self.augment=False  # augmented inference
        self.visualize=False  # visualize features
        self.update=False  # update all models
        self.project=ROOT / 'runs/detect'  # save results to project/name
        self.name='exp'  # save results to project/name
        self.exist_ok=False  # existing project/name ok, do not increment
        self.line_thickness=3  # bounding box thickness (pixels)
        self.hide_labels=False  # hide labels
        self.hide_conf=False  # hide confidences
        self.half=False  # use FP16 half-precision inference
        self.dnn=False  # use OpenCV DNN for ONNX inference
        self.vid_stride=1  # video frame-rate stride

        # opt 안받으려고 아래 source 주석 후 두줄 추가
        # source = str(source) # source : rtsp://prezzie77:1q2w3e4r5t@183.106.132.155:554/stream2

        self.is_file = Path(self.source).suffix[1:] in (IMG_FORMATS + VID_FORMATS) # is_file : False
        self.is_url = self.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')) # is_url : True
        self.webcam = self.source.isnumeric() or self.source.endswith('.txt') or (self.is_url and not self.is_file) # webcam : True

        # Directories
        self.save_dir = increment_path(Path(self.project) / self.name, exist_ok=self.exist_ok)  # increment run, save_dir : runs/detect/exp36
        (self.save_dir / 'labels' if self.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        self.model = None
        self.stride = None
        self.names = None
        self.pt = None
        self.dataset = None

    def load_model_n_get_dataloader(self):
        self.__init__()
        # Load model
        device = select_device(self.device) # device : cpu
        self.model = DetectMultiBackend(self.weights, device=device, dnn=self.dnn, data=self.data, fp16=self.half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size

        # Dataloader
        print('{} # # 03 BackgroundInferTask Start getting rtsp..'.format(datetime.now()))
        self.dataset = LoadStreams(self.source, img_size=imgsz, stride=self.stride, auto=self.pt, vid_stride=self.vid_stride)
        bs = len(self.dataset)

        # Run inference
        print('{} # # 04 BackgroundInferTask Start loading detection model..'.format(datetime.now()))
        self.model.warmup(imgsz=(1 if self.pt or self.model.triton else bs, 3, *imgsz))  # warmup
        self.seen, self.windows, self.dt = 0, [], (Profile(), Profile(), Profile())
        print('{} # # 05 BackgroundInferTask Detection model is ready!'.format(datetime.now()))




    # @smart_inference_mode()
    def run(self, test_str):
        self.__init__()

        print('{} # # 07 BackgroundInferTask Start model infer! test_str : {}'.format(datetime.now(), test_str))

        NumNum = 0
        for path, im, im0s, vid_cap, s in self.dataset:
            NumNum += 1
            print('{} # NumNum : {}'.format(datetime.now(), NumNum))

            # Preprocess image
            with self.dt[0]:
                im = torch.from_numpy(im).to(self.model.device)
                im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with self.dt[1]:
                visualize = increment_path(self.save_dir / Path(path).stem, mkdir=True) if self.visualize else False
                pred = self.model(im, augment=self.augment, visualize=visualize)

            # NMS
            with self.dt[2]:
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
                # print(varname(pred)) # pred : [tensor([[4.92419e+02, 5.03329e+01, 6.40432e+02, 1.47021e+02, 3.21195e-01, 6.00000e+00]])]

            # Process predictions
            for i, det in enumerate(pred):  # per image
                self.seen += 1
                if self.webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), self.dataset.count
                    s += f'{i}: '

                p = Path(p)  # to Path
                save_path = str(self.save_dir / p.name)  # im.jpg
                os.makedirs(save_path, exist_ok=True) # save_path : runs/detect/exp36/stream2
                txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')  # im.txt, 
                # txt_path : runs/detect/exp36/labels/stream2_0
                s += '%gx%g ' % im.shape[2:]  # print string # s : 0: 384x640
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))

                # Stream results
                im0 = annotator.result()

                if len(det):
                    # print('something is detected !')
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # image box label
                    for *xyxy, conf, cls in reversed(det):
                        # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))

            # save png file of detection result image
            print('{} s: {}'.format(datetime.now(), s)) # s: 0: 384x640 1 car,
            if 'car' in str(s) or 'person' in str(s) :
                save_img_file = '{}/{}-car.png'.format(save_path, datetime.now())
                print('############## save_img_file : {}'.format(save_img_file))
                cv2.imwrite(save_img_file, im0)
                time.sleep(5)

        return 'success'

BROKER_URL = 'redis://localhost:6379/0'
CELERY_RESULT_BACKEND = 'redis://localhost:6379/0'
background_infer_task_app = Celery('my_celery_task', broker=BROKER_URL, backend=CELERY_RESULT_BACKEND)
background_infer_task_app.conf.broker_transport_options = {'visibility_timeout':259200}
background_infer_task_app.register_task(BackgroundInferTask())


