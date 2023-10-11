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

python3 detect7.py --weights yolov5n.pt --save-txt --source 'rtsp://prezzie77:1q2w3e4r5t@183.106.132.155:554/stream2'
python3 detect7.py --weights yolov5s.pt --save-txt --source 'rtsp://prezzie77:1q2w3e4r5t@183.106.132.155:554/stream1'
python3 detect7.py --weights yolov5m.pt --save-txt --source 'rtsp://prezzie77:1q2w3e4r5t@183.106.132.155:554/stream1'
python3 detect7.py --weights yolov5l.pt --save-txt --source 'rtsp://prezzie77:1q2w3e4r5t@183.106.132.155:554/stream1'
python3 detect7.py --weights yolov5x.pt --save-txt --source 'rtsp://prezzie77:1q2w3e4r5t@183.106.132.155:554/stream1'
python3 detect7.py --weights yolov5s.pt --save-txt --source 'rtsp://prezzie77:1q2w3e4r5t@172.30.1.60:554/stream2'


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
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
print('1 ROOT:{}'.format(ROOT))
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
print('2 ROOT:{}'.format(ROOT))

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

# @smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=True,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source) # source : rtsp://prezzie77:1q2w3e4r5t@183.106.132.155:554/stream2
    save_img = not nosave and not source.endswith('.txt')  # save inference images, save_img : True

    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS) # is_file : False
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')) # is_url : True
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file) # webcam : True

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run, save_dir : runs/detect/exp36
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device) # device : cpu
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    bs = len(dataset)

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    NumNum = 0
    start_time = time.time()
    start_time_2 = None
    save_first_time = True
    for path, im, im0s, vid_cap, s in dataset:
        time.sleep(1) # sec
        NumNum += 1
        # print('############ NumNum : {}'.format(NumNum))
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
            # print(varname(pred)) # pred : [tensor([[4.92419e+02, 5.03329e+01, 6.40432e+02, 1.47021e+02, 3.21195e-01, 6.00000e+00]])]

        # Process predictions
        save_img_file = None
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            os.makedirs(save_path, exist_ok=True) # save_path : runs/detect/exp36/stream2
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt, 
            # txt_path : runs/detect/exp36/labels/stream2_0
            s += '%gx%g ' % im.shape[2:]  # print string # s : 0: 384x640
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                # print('s:{}'.format(s)) # s : 0: 384x640 1 car,

                # Write results
                # print(varname(txt_path)) # txt_path : runs/detect/exp4/labels/stream1_1
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file, True
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        # print(varname(line)) # line : (tensor(2.), 0.7697916626930237, 0.21944443881511688, 0.10520832985639572, 0.0962962955236435)
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                            f.write(s)

                    # Add bbox to image
                    if save_img or save_crop or view_img:  
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))

                # Stream results
                im0 = annotator.result()

                # Save results (image with detections)
                if save_first_time :
                    save_img_file = '{}/img{}.png'.format(save_path, NumNum)
                    cv2.imwrite(save_img_file, im0)
                    start_time_2 = time.time()
                    save_first_time = False
                elif 'car' in str(s) or 'person' in str(s) :
                    save_img_file = '{}/img{}.png'.format(save_path, NumNum)
                    cv2.imwrite(save_img_file, im0)
                    start_time_2 = time.time()
                    time.sleep(5)
                elif 'handbag' in str(s) or 'suitcase' in str(s) :
                    pass
                elif time.time() - start_time_2 > 600 :
                    save_img_file = '{}/img{}.png'.format(save_path, NumNum)
                    cv2.imwrite(save_img_file, im0)
                    start_time_2 = time.time()

        # Print time (inference-only)
        s_n_img_file = '{}{} '.format(s, save_img_file)
        # LOGGER.info(f"{datetime.now()} | {s_n_img_file}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
        print(f"{datetime.now()} | {s_n_img_file}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
        with open('/home/wasadmin/workspace/ai-cctv/99-final/src/detect7_log.txt', 'a') as f:
            print(f"{datetime.now()} | {s_n_img_file}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms", file=f)

        # if (time.time() - start_time) > 172800 :
        if (time.time() - start_time) > 172800 :
            print('---------- Finishing detection ! -----------')
            with open('/home/wasadmin/workspace/ai-cctv/99-final/src/detect7_log.txt', 'a') as f:
                print('---------- Finishing detection ! -----------', file=f)
            break

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    # Speed: 0.9ms pre-process, 1409.3ms inference, 2.7ms NMS per image at shape (1, 3, 640, 640)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        # Results saved to runs/detect/exp4
        # 1 labels saved to runs/detect/exp4/labels

def parse_opt():
    time.sleep(20)
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


def main(opt):
    # check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
