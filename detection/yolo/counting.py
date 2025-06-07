import argparse
import time
import uuid
from pathlib import Path
import logging

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from osgeo import gdal
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_geotiff(filename):
    """Read GeoTIFF file with error handling."""
    try:
        ds = gdal.Open(filename)
        if ds is None:
            raise ValueError(f"Could not open GeoTIFF file: {filename}")
        
        band = ds.GetRasterBand(1)
        if band is None:
            raise ValueError(f"Could not read band from: {filename}")
            
        arr = band.ReadAsArray()
        return arr, ds
    except Exception as e:
        logger.error(f"Error reading GeoTIFF {filename}: {str(e)}")
        raise


def write_geotiff(filename, arr, in_ds):
    """Write GeoTIFF file with proper resource management."""
    try:
        if arr.dtype == np.float32:
            arr_type = gdal.GDT_Float32
        else:
            arr_type = gdal.GDT_Int32

        driver = gdal.GetDriverByName("GTiff")
        out_ds = driver.Create(filename, arr.shape[1], arr.shape[0], 1, arr_type)
        
        if out_ds is None:
            raise ValueError(f"Could not create output file: {filename}")
            
        out_ds.SetProjection(in_ds.GetProjection())
        out_ds.SetGeoTransform(in_ds.GetGeoTransform())
        
        band = out_ds.GetRasterBand(1)
        band.WriteArray(arr)
        band.FlushCache()
        band.ComputeStatistics(False)
        
        # Proper cleanup
        band = None
        out_ds = None
        
        logger.info(f"GeoTIFF saved: {filename}")
        
    except Exception as e:
        logger.error(f"Error writing GeoTIFF {filename}: {str(e)}")
        raise


def count(founded_classes, im0, total_count_list):
    """Count objects and draw on image."""
    try:
        aligns = im0.shape
        color_blue = (255, 0, 0)
        color_red = (0, 0, 255)
        thickness = 2

        # Add current frame count to total
        current_frame_total = sum(founded_classes.values())
        total_count_list.append(current_frame_total)
        
        # Display total count
        total_objects = sum(total_count_list)
        text = f"Total Objects: {total_objects}"
        cv2.putText(im0, text, (10, 170), cv2.FONT_HERSHEY_SIMPLEX,
                    1, color_blue, thickness, cv2.LINE_AA)
        
        # Display current frame counts
        y_offset = 200
        for fruit_type, count in founded_classes.items():
            text = f"{fruit_type}: {count}"
            cv2.putText(im0, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color_red, thickness, cv2.LINE_AA)
            y_offset += 30
            
        logger.info(f"Frame objects: {current_frame_total}, Total: {total_objects}")
        return total_objects
        
    except Exception as e:
        logger.error(f"Error in count function: {str(e)}")
        return 0


def detect(save_img=False):
    """Main detection function."""
    source, weights, view_img, save_txt, imgsz, trace = (
        opt.source, opt.weights, opt.view_img, opt.save_txt, 
        opt.img_size, not opt.no_trace
    )
    
    # Initialize count list for this detection session
    total_count_list = []
    
    save_img = not opt.nosave and not source.endswith('.txt')
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(Path(opt.project) / opt.name)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = False  # device.type != 'cpu'

    try:
        # Load model
        model = attempt_load(weights, map_location=device)
        stride = int(model.stride.max())
        imgsz = check_img_size(imgsz, s=stride)

        if trace:
            model = TracedModel(model, device, opt.img_size)

        if half:
            model.half()

        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
                next(model.parameters())))

        t0 = time.time()
        
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()
            img /= 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=opt.augment)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(
                pred, opt.conf_thres, opt.iou_thres, 
                classes=opt.classes, agnostic=opt.agnostic_nms)
            t3 = time_synchronized()

            # Process detections
            for i, det in enumerate(pred):
                if webcam:
                    p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                p = Path(p)
                save_path = str(save_dir / p.name)
                txt_path = str(save_dir / 'labels' / p.stem) + \
                    ('' if dataset.mode == 'image' else f'_{frame}')
                
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                
                if len(det):
                    # Rescale boxes
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    founded_classes = {}
                    
                    # Count objects by class
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()
                        class_index = int(c)
                        founded_classes[names[class_index]] = int(n)
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                    # Apply counting and visualization
                    total_objects = count(founded_classes=founded_classes, 
                                        im0=im0, total_count_list=total_count_list)

                    # Draw bounding boxes
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or view_img:
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label,
                                       color=colors[int(cls)], line_thickness=2)

                # Stream results
                if view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)

                # Save results
                if save_img:
                    if dataset.mode == 'image':
                        # Generate unique filename for GeoTIFF output
                        unique_id = str(uuid.uuid4())[:8]
                        geotiff_output = save_dir / f"result_{unique_id}.tif"
                        
                        try:
                            if source.lower().endswith(('.tif', '.tiff')):
                                nlcd16_arr, nlcd16_ds = read_geotiff(source)
                                write_geotiff(str(geotiff_output), im0, nlcd16_ds)
                        except Exception as e:
                            logger.warning(f"Could not save GeoTIFF: {str(e)}")
                        
                        cv2.imwrite(save_path, im0)
                        logger.info(f"Image saved: {save_path}")
                        
                    else:  # video
                        if vid_path != save_path:
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()
                            
                            if vid_cap:
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            
                            vid_writer = cv2.VideoWriter(
                                save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer.write(im0)

        # Cleanup
        if isinstance(vid_writer, cv2.VideoWriter):
            vid_writer.release()
            
        processing_time = time.time() - t0
        logger.info(f'Detection completed in {processing_time:.3f}s')
        
        # Return total count for API integration
        return sum(total_count_list) if total_count_list else 0
        
    except Exception as e:
        logger.error(f"Error during detection: {str(e)}")
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    
    opt = parser.parse_args()

    with torch.no_grad():
        if opt.update:
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            result = detect()
            print(f"Total detected objects: {result}")