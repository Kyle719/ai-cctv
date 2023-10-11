# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

python3 detect9.py --weights yolov5n.pt --save-txt --source 'rtsp://prezzie77:1q2w3e4r5t@183.106.132.155:554/stream2'
python3 detect9.py --weights yolov5s.pt --save-txt --source 'rtsp://prezzie77:1q2w3e4r5t@183.106.132.155:554/stream1'
python3 detect9.py --weights yolov5m.pt --save-txt --source 'rtsp://prezzie77:1q2w3e4r5t@183.106.132.155:554/stream1'
python3 detect9.py --weights yolov5l.pt --save-txt --source 'rtsp://prezzie77:1q2w3e4r5t@183.106.132.155:554/stream1'
python3 detect9.py --weights yolov5x.pt --save-txt --source 'rtsp://prezzie77:1q2w3e4r5t@183.106.132.155:554/stream1'
python3 detect9.py --weights yolov5s.pt --save-txt --source 'rtsp://prezzie77:1q2w3e4r5t@172.30.1.60:554/stream2'

gunicorn -b 0.0.0.0:7000 detect9:detect_app -k gthread -w 1 --threads 2



Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path
import time
from datetime import datetime
from flask import Flask, render_template, Response, request, redirect, url_for
import torch

detect_app = Flask(__name__)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
# print('1 ROOT:{}'.format(ROOT))
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
# print('2 ROOT:{}'.format(ROOT))

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

# print line num and variable
import inspect, re
def varname(p):
    cf = inspect.currentframe()
    linenum = cf.f_back.f_lineno
    # linenum = inspect.getlineno(inspect.getouterframes(inspect.currentframe())[-1][0])
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return '\n### LINENUM : {} - {} : {}'.format(linenum, m.group(1), p)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


class DetcInfer():
    def __init__(self):
        print('{} # # 02 Start making DetcInfer instance..'.format(datetime.now()))
        super().__init__()
        self.temp_test = '########### test value'
        self.rtsp_requested = False

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
        self.save_txt=True  # save results to *.txt
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

        # Load model
        device = select_device(self.device) # device : cpu
        self.model = DetectMultiBackend(self.weights, device=device, dnn=self.dnn, data=self.data, fp16=self.half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size

        # Dataloader
        print('{} # # 03 Start getting rtsp..'.format(datetime.now()))
        self.dataset = LoadStreams(self.source, img_size=imgsz, stride=self.stride, auto=self.pt, vid_stride=self.vid_stride)
        bs = len(self.dataset)

        # Run inference
        print('{} # # 04 Start loading detection model..'.format(datetime.now()))
        self.model.warmup(imgsz=(1 if self.pt or self.model.triton else bs, 3, *imgsz))  # warmup
        self.seen, self.windows, self.dt = 0, [], (Profile(), Profile(), Profile())
        print('{} # # 05 Detection model is ready!'.format(datetime.now()))


    # @smart_inference_mode()
    def run_n_show(self):
        print(self.temp_test)
        print('{} # # 07 Start model infer!'.format(datetime.now()))

        NumNum = 0
        for path, im, im0s, vid_cap, s in self.dataset:
            NumNum += 1
            print('{} ############ NumNum : {}'.format(datetime.now(), NumNum))

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
                    # print('s:{}'.format(s)) # s : 0: 384x640 1 car,

                    # Write results
                    # print(varname(txt_path)) # txt_path : runs/detect/exp4/labels/stream1_1
                    for *xyxy, conf, cls in reversed(det):
                        if self.save_txt:  # Write to file, True
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if self.save_conf else (cls, *xywh)  # label format
                            # print(varname(line)) # line : (tensor(2.), 0.7697916626930237, 0.21944443881511688, 0.10520832985639572, 0.0962962955236435)
                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')
                                f.write(s)

                        # Add bbox to image
                        if self.save_img or self.save_crop or self.view_img:  
                            c = int(cls)  # integer class
                            label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                    print(s)
                    # if 'car' in str(s) :
                    #     save_img_file = '{}/{}-car.png'.format(save_path, datetime.now())
                    #     print('############## save_img_file : {}'.format(save_img_file))
                    #     cv2.imwrite(save_img_file, im0)
                    # if 'person' in str(s) :
                    #     save_img_file = '{}/{}-person.png'.format(save_path, datetime.now())
                    #     print('############## save_img_file : {}'.format(save_img_file))
                    #     cv2.imwrite(save_img_file, im0)
                    # if 'car' in str(s) or 'person' in str(s) :
                    #     time.sleep(5)


            # Send detection result image on website page
            # print('Send detection result image on website page')
            ret, buffer = cv2.imencode('.jpg', im0)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

    def run_n_save(self):
        for num in range(1000):
            time.sleep(1)
            temp_str = '{} # # run_n_save test {}'.format(datetime.now(), num)
            yield temp_str


print('{} # # 01 Start service..'.format(datetime.now()))

yolo_detector = DetcInfer()

# gen = yolo_detector.run_n_save()
# for i in gen:
#     print(i)

@detect_app.route('/video_feed')
def video_feed():
    print('{} @ @ 2 request:{}'.format(datetime.now(), request))
    print('{} @ @ 2 str(request):{}'.format(datetime.now(), str(request)))
    print('{} @ @ 2 request.method:{}'.format(datetime.now(), request.method))
    #Video streaming route. Put this in the src attribute of an img tag
    print('{} # # 06 Start detecting and visualizing on website..'.format(datetime.now()))
    detc_generator = yolo_detector.run_n_show()
    return Response(detc_generator, mimetype='multipart/x-mixed-replace; boundary=frame')
    # return Response(yolo_detector.run_n_show(), mimetype='multipart/x-mixed-replace; boundary=frame')

@detect_app.route('/video_stream')
def video_stream():
    print('{} @ @ 1 request:{}'.format(datetime.now(), request))
    """Video streaming home page."""
    return render_template('index2.html')

@detect_app.route('/')
def index():
    print('{} @ @ 0 main page request:{}'.format(datetime.now(), request))
    """Main page."""
    return render_template('index.html')

