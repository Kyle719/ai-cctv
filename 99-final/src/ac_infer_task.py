import os
import sys
from pathlib import Path
import time
from datetime import datetime
import torch
from celery import Celery, Task
import json

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

RTSP_URL = 'rtsp://prezzie77:1q2w3e4r5t@183.106.132.155:554/stream1'
CONF_THRES = 0.60
CLASSES = [0,2,14,15,16]

'''
0: person
2: car
14: bird
15: cat
16: dog
'''

class StreamInfer():
    def __init__(self):
        print(f'{datetime.now()} # # 02 StreamInfer Start making DetcInfer instance..')
        super().__init__()
        self.model_speed_accu = '6040'
        self.weights=ROOT / 'yolov5s.pt'  # model path or triton URL
        self.source=RTSP_URL
        self.data=ROOT / 'data/coco128.yaml'  # dataset.yaml path
        self.imgsz=(640, 640)  # inference size (height, width)
        # self.conf_thres=0.25  # confidence threshold
        self.conf_thres=CONF_THRES  # confidence threshold
        self.iou_thres=0.45  # NMS IOU threshold
        self.max_det=1000  # maximum detections per image
        self.device=''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.view_img=False  # show results
        self.save_img = True
        self.save_txt=False  # save results to *.txt
        self.save_conf=False  # save confidences in --save-txt labels
        self.save_crop=False  # save cropped prediction boxes
        self.nosave=False  # do not save images/videos
        self.classes=CLASSES  # filter by class: --class 0, or --class 0 2 3
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

        self.is_file = Path(self.source).suffix[1:] in (IMG_FORMATS + VID_FORMATS) # is_file : False
        self.is_url = self.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')) # is_url : True
        self.webcam = self.source.isnumeric() or self.source.endswith('.txt') or (self.is_url and not self.is_file) # webcam : True

        # Directories
        self.save_dir = increment_path(Path(self.project) / self.name, exist_ok=self.exist_ok)  # increment run, save_dir : runs/detect/exp36

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
        print(f'{datetime.now()} # # 04 StreamInfer Start loading detection model..')
        self.model.warmup(imgsz=(1 if self.pt or self.model.triton else bs, 3, *imgsz))  # warmup
        self.seen, self.windows, self.dt = 0, [], (Profile(), Profile(), Profile())
        print(f'{datetime.now()} # # 05 StreamInfer Detection model is ready!')

    # @smart_inference_mode()
    def run(self):
        print(f'{datetime.now()} # # 07 StreamInfer Start model infer!')

        task_parameters_json_dir = './record_configs'
        task_parameters_json_files = os.listdir(task_parameters_json_dir)
        task_parameters_json_files.sort()
        task_parameters_recent_json = f'{task_parameters_json_dir}/{task_parameters_json_files[-1]}'

        with open(task_parameters_recent_json, 'r') as file:
            task_parameters = json.load(file)
            print(f'task_parameters:{task_parameters}')
        # task_parameters:{'checkbox_person': 'yes', 'checkbox_car': 'yes', 'checkbox_dog': 'yes', 'checkbox_cat': 'None', 'checkbox_bird': 'None', 'save_photos': 'on', 'send_msg': 'on', 'redio_speed_accu': '4060', 'min_save_period': '3', 'detc_threshold': '0.6'}

        # AI 모델 변경 시 모델 교체 적용
        if task_parameters['redio_speed_accu'] == self.model_speed_accu :
            pass
        else :
            if task_parameters['redio_speed_accu'] == '6040' :
                device = select_device(self.device) # device : cpu
                self.weights = ROOT / 'yolov5s.pt'
                self.model_speed_accu = '6040'
            elif task_parameters['redio_speed_accu'] == '5050' :
                device = select_device(self.device) # device : cpu
                self.weights = ROOT / 'yolov5m.pt'
                self.model_speed_accu = '5050'
            elif task_parameters['redio_speed_accu'] == '4060' :
                device = select_device(self.device) # device : cpu
                self.weights = ROOT / 'yolov5x.pt'
                self.model_speed_accu = '4060'
            self.model = DetectMultiBackend(self.weights, device=device, dnn=self.dnn, data=self.data, fp16=self.half)
            self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
            imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size
            # Run inference
            bs = len(self.dataset)
            print(f'{datetime.now()} # # 04 StreamInfer Start loading detection model.. model_speed_accu : {self.model_speed_accu}')
            self.model.warmup(imgsz=(1 if self.pt or self.model.triton else bs, 3, *imgsz))  # warmup
            self.seen, self.windows, self.dt = 0, [], (Profile(), Profile(), Profile())
            print(f'{datetime.now()} # # 05 StreamInfer Detection model is ready! model_speed_accu : {self.model_speed_accu}')

        # 감시 대상 변경 시 적용
        task_classes = []
        if task_parameters['checkbox_person'] == 'yes':
            task_classes.append(0)
        if task_parameters['checkbox_car'] == 'yes':
            task_classes.append(2)
        if task_parameters['checkbox_dog'] == 'yes':
            task_classes.append(16)
        if task_parameters['checkbox_cat'] == 'yes':
            task_classes.append(15)
        if task_parameters['checkbox_bird'] == 'yes':
            task_classes.append(14)
        if task_classes == []:
            task_classes.append(0)
            task_classes.append(2)
        self.classes = task_classes
        print(f'self.classes:{self.classes}')

        # Threshold 변경 시 적용
        task_detc_threshold = 0.6
        if len(task_parameters['detc_threshold']) > 0:
            task_detc_threshold = float(task_parameters['detc_threshold'])
        self.conf_thres = task_detc_threshold
        print(f'self.conf_thres:{self.conf_thres}')

        # NumNum = 0
        for path, im, im0s, vid_cap, s in self.dataset:
            # NumNum += 1
            # print('{} # NumNum : {}'.format(datetime.now(), NumNum))

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
            _, buffer = cv2.imencode('.jpg', im0)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


BROKER_URL = 'redis://localhost:6379/0'
CELERY_RESULT_BACKEND = 'redis://localhost:6379/0'
background_infer_task_app = Celery('my_celery_task', broker=BROKER_URL, backend=CELERY_RESULT_BACKEND)
# background_infer_task_app.conf.broker_transport_options = {'visibility_timeout':259200}
# background_infer_task_app.register_task(BackgroundInferTask())

@background_infer_task_app.task
def BackgroundInfer(test_str, first_infer_flag):

    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[0]  # YOLOv5 root directory
    # print('1 ROOT:{}'.format(ROOT))
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))  # add ROOT to PATH
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

    print(test_str)

    task_parameters_json_dir = './record_configs'
    task_parameters_json_files = os.listdir(task_parameters_json_dir)
    task_parameters_json_files.sort()
    task_parameters_recent_json = f'{task_parameters_json_dir}/{task_parameters_json_files[-1]}'

    with open(task_parameters_recent_json, 'r') as file:
        task_parameters = json.load(file)
        print(f'task_parameters:{task_parameters}')

    # task_parameters:{'checkbox_person': 'yes', 'checkbox_car': 'yes', 'checkbox_dog': 'yes', 'checkbox_cat': 'None', 'checkbox_bird': 'None', 'save_photos': 'on', 'send_msg': 'on', 'redio_speed_accu': '4060', 'min_save_period': '3', 'detc_threshold': '0.6'}

    task_parameters['checkbox_person']
    task_parameters['checkbox_car']
    task_parameters['checkbox_dog']
    task_parameters['checkbox_cat']
    task_parameters['checkbox_bird']
    task_parameters['redio_speed_accu']
    task_parameters['detc_threshold']
    task_parameters['min_save_period']
    task_parameters['save_photos']
    task_parameters['send_msg']

    task_classes = []
    if task_parameters['checkbox_person'] == 'yes':
        task_classes.append(0)
    if task_parameters['checkbox_car'] == 'yes':
        task_classes.append(2)
    if task_parameters['checkbox_dog'] == 'yes':
        task_classes.append(16)
    if task_parameters['checkbox_cat'] == 'yes':
        task_classes.append(15)
    if task_parameters['checkbox_bird'] == 'yes':
        task_classes.append(14)
    if task_classes == []:
        task_classes.append(0)
        task_classes.append(2)
    print(f'task_classes:{task_classes}')

    task_ai_model = ''
    if task_parameters['redio_speed_accu'] == '6040':
        task_ai_model = 'yolov5s.pt'
    if task_parameters['redio_speed_accu'] == '5050':
        task_ai_model = 'yolov5m.pt'
    if task_parameters['redio_speed_accu'] == '4060':
        task_ai_model = 'yolov5x.pt'
    if task_ai_model == '':
        task_ai_model = 'yolov5m.pt'
    print(f'task_ai_model:{task_ai_model}')

    task_detc_threshold = 0.6
    if len(task_parameters['detc_threshold']) > 0:
        task_detc_threshold = float(task_parameters['detc_threshold'])

    task_min_save_period = 5
    if len(task_parameters['min_save_period']) > 0:
        task_min_save_period = int(task_parameters['min_save_period'])

    save_photos = True
    if task_parameters['save_photos'] == 'off':
        save_photos = False
    send_msg = True
    if task_parameters['send_msg'] == 'off':
        send_msg = False

    weights=ROOT / task_ai_model  # model path or triton URL
    source=RTSP_URL
    data=ROOT / 'data/coco128.yaml'  # dataset.yaml path
    imgsz=(640, 640)  # inference size (height, width)
    # conf_thres=0.25  # confidence threshold
    conf_thres=task_detc_threshold  # confidence threshold
    iou_thres=0.45  # NMS IOU threshold
    max_det=1000  # maximum detections per image
    device=''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False  # show results
    save_img = True
    save_txt=False  # save results to *.txt
    save_conf=False  # save confidences in --save-txt labels
    save_crop=False  # save cropped prediction boxes
    nosave=False  # do not save images/videos
    # classes=None  # filter by class: --class 0, or --class 0 2 3
    classes=task_classes
    agnostic_nms=False  # class-agnostic NMS
    augment=False  # augmented inference
    visualize=False  # visualize features
    update=False  # update all models
    project=ROOT / 'runs/detect'  # save results to project/name
    name='exp'  # save results to project/name
    exist_ok=False  # existing project/name ok, do not increment
    line_thickness=3  # bounding box thickness (pixels)
    hide_labels=False  # hide labels
    hide_conf=False  # hide confidences
    half=False  # use FP16 half-precision inference
    dnn=False  # use OpenCV DNN for ONNX inference
    vid_stride=1  # video frame-rate stride


    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS) # is_file : False
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')) # is_url : True
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file) # webcam : True

    yy_mm_dd = str((datetime.now()).strftime('%Y-%m-%d'))
    save_dir = './runs/detect/{}'.format(yy_mm_dd)  # ./runs/detect/2020-01-07/
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Load model
    device = select_device(device) # device : cpu
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    print('# # 03 BackgroundInferTask Start getting rtsp..')
    dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    bs = len(dataset)

    # Run inference
    print('# # 04 BackgroundInferTask Start loading detection model..')
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    print('{} # # 05 BackgroundInferTask Detection model is ready!'.format(datetime.now()))



    print('# # 07 BackgroundInferTask Start model infer! test_str : {}')

    for path, im, im0s, vid_cap, s in dataset:

        yy_mm_dd = str((datetime.now()).strftime('%Y-%m-%d'))
        save_dir = './runs/detect/{}'.format(yy_mm_dd)  # ./runs/detect/2020-01-07/
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        hh_mm_ss = str((datetime.now()).strftime('%H:%M:%S'))   # 15:40:15

        # Preprocess image
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        detc_per_img_summary = []
        detc_per_img = []

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '

            p = Path(p)  # to Path
            s += '%gx%g ' % im.shape[2:]  # print string # s : 0: 384x640
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            # Stream results
            im0 = annotator.result()

            # print(f'len(det):{len(det)}') # 찾은 obj 개수
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    # print('n:{}'.format(n)) # n:2
                    # print('c:{}'.format(c)) # c:0.0
                    # print('names[int(c)]:{}'.format(names[int(c)])) # names[int(c)]:person
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    detc_per_img_summary.append(n)
                    detc_per_img_summary.append(names[int(c)])

                # image box label and get detc info
                # detc_per_img = ['person', 0.64, '1359,131,1397,248', 'person'..., 'dog'...]
                j = 0
                for *xyxy, conf, cls in reversed(det):
                    # Add bbox to image
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')

                    detc_per_img.append(f'{names[c]}')
                    detc_per_img.append(f'{conf:.2f}')
                    detc_per_img.append(f'{int(xyxy[0])},{int(xyxy[1])},{int(xyxy[2])},{int(xyxy[3])}')
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    j += 1

        # save png file of detection result image
        if first_infer_flag :
            save_img_file = f'{save_dir}/{hh_mm_ss}-task-started.png'
            print(f'# # save_img_file : {save_img_file}')
            cv2.imwrite(save_img_file, im0)
            first_infer_flag = False
        if len(detc_per_img) > 0:
            print(f'# # s: {s}') # # s: 0: 384x640 2 persons,
            print(f'# # detc_per_img_summary: {detc_per_img_summary}')  # [tensor(1), 'dog']
            print(f'# # detc_per_img: {detc_per_img}')  # ['person', '0.26', '1897,358,1919,441']
            detc_res_summ = ''
            for i, obj in enumerate(detc_per_img_summary):
                if i % 2 == 0 :
                    detc_res_summ += '_'
                    detc_res_summ += str(int(detc_per_img_summary[i]))
                else :
                    detc_res_summ += str(detc_per_img_summary[i])
            print(f'# # detc_res_summ: {detc_res_summ}')    # _1person
            save_img_file = f'{save_dir}/{hh_mm_ss}{detc_res_summ}.png'
            print(f'# # save_img_file : {save_img_file}')   # ./runs/detect/2023-10-10/09:27:07_1person.png

            if save_photos :
                cv2.imwrite(save_img_file, im0)
            # if send_msg :
            #     send_message()
            time.sleep(task_min_save_period)


def get_task_info():
    # worker_ping_res = background_infer_task_app.control.inspect().ping()
    # celery worker 가 떠있으면 pong 이 돌아옴
    insp_val = background_infer_task_app.control.inspect()
    active_tasks = insp_val.active()
    reserved_tasks = insp_val.reserved()
    # 현재 돌고 있는 task 랑 브로커 큐에 쌓여있는 에약 task 정보 가져옴
    return active_tasks, reserved_tasks

def kill_executing_task(task_id):
    result = background_infer_task_app.control.revoke(task_id, terminate=True)
    return result

def get_folder_file_list(path):
    folder_file_list = os.listdir(path)
    folder_file_list.sort()
    return folder_file_list

def get_image_bytedata(img_dir):
    # img_dir:['2023-09-22', '17:51:08.png']
    img_dir = eval(img_dir)
    img_rel_path = './runs/detect/{}/{}'.format(img_dir[0], img_dir[1])
    image = cv2.imread(img_rel_path)
    _, enc_image = cv2.imencode('.jpg', image)
    image_bytes = enc_image.tobytes()
    res_image =  (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + image_bytes + b'\r\n')

    return res_image

