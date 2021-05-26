import argparse
import time
from pathlib import Path

import cv2
import torch, torchvision
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


def detect(opt):
    source, weight_near, weight_far, view_img, save_txt, imgsz = opt.source, opt.weight_near, opt.weight_far, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model_far = attempt_load(weight_far, map_location=device)  # load FP32 model
    model_near = attempt_load(weight_near, map_location=device)  # load FP32 model
    stride = int(model_near.stride.max())  # model stride

    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model_near.module.names if hasattr(model_near, 'module') else model_near.names  # get class names
    if half:
        model_near.half()  # to FP16
        model_far.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        model_near(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model_near.parameters())))  # run once
        model_far(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model_far.parameters())))  # run once

    x_lim_upper = 0.75
    x_lim_lower = 0.25
    y_lim_upper = 0.6
    y_lim_lower = 0

    x_delta = x_lim_upper - x_lim_lower
    y_delta = y_lim_upper - y_lim_lower

    # print(img.shape) : (3, 384, 640)
    # print(im0s.shape) : (1080, 1920, 3)
    # print(img_crop.shape) : (3, 448, 640)
    # print(im_crop0s.shape) : (648, 960, 3)

    t0 = time.time()
    for path, img, im0s, img_crop, im_crop0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        img_crop = torch.from_numpy(img_crop).to(device)
        img_crop = img_crop.half() if half else img_crop.float()  # uint8 to fp16/32
        img_crop /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img_crop.ndimension() == 3:
            img_crop = img_crop.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred_far = model_far(img_crop, augment=opt.augment)[0]
        pred_near = model_near(img, augment=opt.augment)[0]

        # Apply NMS
        pred_far = non_max_suppression(pred_far, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms,
                                   max_det=opt.max_det)
        pred_near = non_max_suppression(pred_near, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms,
                                   max_det=opt.max_det)
        
        # Perform superimposition
        pred_final=torch.tensor([]).to(device)

        for det in pred_far:
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img_crop.shape[2:], det[:, :4], im_crop0s.shape).round()

            # det [x1,y1,x2,y2]        
            det[:,[0,2]] = det[:,[0,2]] + x_lim_lower*im0s.shape[1]
            det[:,[1,3]] = det[:,[1,3]] + y_lim_lower*im0s.shape[0]
            pred_final=torch.cat([pred_final,det])
          

        for det in pred_near:
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
            pred_final=torch.cat([pred_final,det])


        pred_final = pred_final.unsqueeze(0)

        output = [torch.zeros((0, 6), device=pred_final.device)] * pred_final.shape[0]
        for xi, x in enumerate(pred_final):
            # If none remain process next image
            if not x.shape[0]:
                continue

            boxes, scores = x[:, :4], x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, 0.45)  # NMS
            output[xi] = x[i]


        t2 = time_synchronized()

        # Apply Classifier
        # if classify:
        #     pred_final = apply_classifier(pred_final, modelc, img, im0s)


        # Process detections
        for i, det in enumerate(output):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if opt.save_crop else im0  # for opt.save_crop
            
            if len(det):
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or opt.save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=opt.line_thickness)
                        if opt.save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight-far', nargs='+', type=str, default='yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--weight-near', nargs='+', type=str, default='yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum number of detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weight_near in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect(opt=opt)
                strip_optimizer(opt.weight_near)
        else:
            detect(opt=opt)
