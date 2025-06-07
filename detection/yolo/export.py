import argparse
import sys
import time
import warnings
import logging
from pathlib import Path

sys.path.append('./')  # to run '$ python *.py' files in subdirectories

import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile

import models
from models.experimental import attempt_load, End2End
from utils.activations import Hardswish, SiLU
from utils.general import set_logging, check_img_size
from utils.torch_utils import select_device
from utils.add_nms import RegisterNMS

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def export_torchscript(model, img, weights_path):
    """Export model to TorchScript format."""
    try:
        logger.info(f'Starting TorchScript export with torch {torch.__version__}...')
        f = weights_path.replace('.pt', '.torchscript.pt')
        ts = torch.jit.trace(model, img, strict=False)
        ts.save(f)
        logger.info(f'TorchScript export success, saved as {f}')
        return f
    except Exception as e:
        logger.error(f'TorchScript export failure: {str(e)}')
        return None


def export_coreml(ts, img, weights_path, opt):
    """Export model to CoreML format."""
    try:
        import coremltools as ct
        
        logger.info(f'Starting CoreML export with coremltools {ct.__version__}...')
        
        # Convert model from torchscript
        ct_model = ct.convert(ts, inputs=[ct.ImageType('image', shape=img.shape, scale=1/255.0, bias=[0, 0, 0])])
        
        # Apply quantization if requested
        bits, mode = (8, 'kmeans_lut') if opt.int8 else (16, 'linear') if opt.fp16 else (32, None)
        if bits < 32:
            if sys.platform.lower() == 'darwin':  # quantization only supported on macOS
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=DeprecationWarning)
                    ct_model = ct.models.neural_network.quantization_utils.quantize_weights(ct_model, bits, mode)
                logger.info(f'Applied {bits}-bit quantization')
            else:
                logger.warning('Quantization only supported on macOS, skipping...')

        f = weights_path.replace('.pt', '.mlmodel')
        ct_model.save(f)
        logger.info(f'CoreML export success, saved as {f}')
        return f
    except ImportError:
        logger.warning('CoreML export requires: pip install coremltools')
        return None
    except Exception as e:
        logger.error(f'CoreML export failure: {str(e)}')
        return None


def export_torchscript_lite(model, img, weights_path):
    """Export model to TorchScript-Lite format."""
    try:
        logger.info(f'Starting TorchScript-Lite export with torch {torch.__version__}...')
        f = weights_path.replace('.pt', '.torchscript.ptl')
        tsl = torch.jit.trace(model, img, strict=False)
        tsl = optimize_for_mobile(tsl)
        tsl._save_for_lite_interpreter(f)
        logger.info(f'TorchScript-Lite export success, saved as {f}')
        return f
    except Exception as e:
        logger.error(f'TorchScript-Lite export failure: {str(e)}')
        return None


def export_onnx(model, img, weights_path, opt, labels):
    """Export model to ONNX format."""
    try:
        import onnx
        
        logger.info(f'Starting ONNX export with onnx {onnx.__version__}...')
        f = weights_path.replace('.pt', '.onnx')
        model.eval()
        
        # Configure output names
        output_names = ['classes', 'boxes'] if opt.include_nms else ['output']
        
        # Configure dynamic axes
        dynamic_axes = None
        if opt.dynamic:
            dynamic_axes = {
                'images': {0: 'batch', 2: 'height', 3: 'width'},
                'output': {0: 'batch', 2: 'y', 3: 'x'}
            }
        
        if opt.dynamic_batch:
            opt.batch_size = 'batch'
            dynamic_axes = {'images': {0: 'batch'}}
            
            if opt.end2end and opt.max_wh is None:
                output_axes = {
                    'num_dets': {0: 'batch'},
                    'det_boxes': {0: 'batch'},
                    'det_scores': {0: 'batch'},
                    'det_classes': {0: 'batch'},
                }
            else:
                output_axes = {'output': {0: 'batch'}}
            dynamic_axes.update(output_axes)

        # Configure end-to-end export
        if opt.grid:
            if opt.end2end:
                backend = 'TensorRT' if opt.max_wh is None else 'onnxruntime'
                logger.info(f'Configuring end2end onnx model for {backend}...')
                model = End2End(model, opt.topk_all, opt.iou_thres, opt.conf_thres, opt.max_wh, model.device, len(labels))
                
                if opt.end2end and opt.max_wh is None:
                    output_names = ['num_dets', 'det_boxes', 'det_scores', 'det_classes']
                    shapes = [opt.batch_size, 1, opt.batch_size, opt.topk_all, 4,
                             opt.batch_size, opt.topk_all, opt.batch_size, opt.topk_all]
                else:
                    output_names = ['output']
            else:
                model.model[-1].concat = True

        # Export to ONNX
        torch.onnx.export(
            model, img, f, verbose=False, opset_version=12,
            input_names=['images'], output_names=output_names,
            dynamic_axes=dynamic_axes
        )

        # Validate ONNX model
        onnx_model = onnx.load(f)
        onnx.checker.check_model(onnx_model)
        logger.info('ONNX model validation passed')

        # Configure output shapes for end2end
        if opt.end2end and opt.max_wh is None:
            for i in onnx_model.graph.output:
                for j in i.type.tensor_type.shape.dim:
                    j.dim_param = str(shapes.pop(0))

        # Simplify ONNX model if requested
        if opt.simplify:
            try:
                import onnxsim
                logger.info('Starting ONNX simplification...')
                onnx_model, check = onnxsim.simplify(onnx_model)
                assert check, 'ONNX simplification check failed'
                logger.info('ONNX simplification completed')
            except ImportError:
                logger.warning('ONNX simplification requires: pip install onnx-simplifier')
            except Exception as e:
                logger.error(f'ONNX simplification failure: {str(e)}')

        # Save final ONNX model
        onnx.save(onnx_model, f)
        logger.info(f'ONNX export success, saved as {f}')

        # Register NMS plugin if requested
        if opt.include_nms:
            try:
                logger.info('Registering NMS plugin for ONNX...')
                mo = RegisterNMS(f)
                mo.register_nms()
                mo.save(f)
                logger.info('NMS plugin registration completed')
            except Exception as e:
                logger.error(f'NMS plugin registration failed: {str(e)}')

        return f
    except ImportError:
        logger.warning('ONNX export requires: pip install onnx')
        return None
    except Exception as e:
        logger.error(f'ONNX export failure: {str(e)}')
        return None


def main():
    """Main export function."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./yolov7.pt', help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--dynamic', action='store_true', help='dynamic ONNX axes')
    parser.add_argument('--dynamic-batch', action='store_true', help='dynamic batch onnx for tensorrt and onnx-runtime')
    parser.add_argument('--grid', action='store_true', help='export Detect() layer grid')
    parser.add_argument('--end2end', action='store_true', help='export end2end onnx')
    parser.add_argument('--max-wh', type=int, default=None, help='None for tensorrt nms, int value for onnx-runtime nms')
    parser.add_argument('--topk-all', type=int, default=100, help='topk objects for every images')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='iou threshold for NMS')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='conf threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--simplify', action='store_true', help='simplify onnx model')
    parser.add_argument('--include-nms', action='store_true', help='export end2end onnx')
    parser.add_argument('--fp16', action='store_true', help='CoreML FP16 half-precision export')
    parser.add_argument('--int8', action='store_true', help='CoreML INT8 quantization')
    
    # Format selection
    parser.add_argument('--formats', nargs='+', default=['torchscript', 'onnx'], 
                       choices=['torchscript', 'coreml', 'torchscript-lite', 'onnx'],
                       help='export formats')
    
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    opt.dynamic = opt.dynamic and not opt.end2end
    opt.dynamic = False if opt.dynamic_batch else opt.dynamic
    
    logger.info(f"Export configuration: {opt}")
    
    # Check if weights file exists
    if not Path(opt.weights).exists():
        logger.error(f"Weights file not found: {opt.weights}")
        return
    
    set_logging()
    t = time.time()
    exported_files = []

    try:
        # Load PyTorch model
        device = select_device(opt.device)
        logger.info(f"Loading model: {opt.weights}")
        model = attempt_load(opt.weights, map_location=device)
        labels = model.names
        logger.info(f"Model loaded successfully. Classes: {labels}")

        # Verify image size
        gs = int(max(model.stride))
        opt.img_size = [check_img_size(x, gs) for x in opt.img_size]
        logger.info(f"Image size: {opt.img_size}, Grid size: {gs}")

        # Create dummy input
        img = torch.zeros(opt.batch_size, 3, *opt.img_size).to(device)

        # Update model for export compatibility
        for k, m in model.named_modules():
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
            if isinstance(m, models.common.Conv):
                if isinstance(m.act, nn.Hardswish):
                    m.act = Hardswish()
                elif isinstance(m.act, nn.SiLU):
                    m.act = SiLU()

        model.model[-1].export = not opt.grid
        y = model(img)  # dry run
        
        if opt.include_nms:
            model.model[-1].include_nms = True
            y = None

        logger.info("Model preparation completed")

        # Export to selected formats
        if 'torchscript' in opt.formats:
            ts_file = export_torchscript(model, img, opt.weights)
            if ts_file:
                exported_files.append(ts_file)
                
                # CoreML requires TorchScript
                if 'coreml' in opt.formats:
                    ts = torch.jit.load(ts_file)
                    coreml_file = export_coreml(ts, img, opt.weights, opt)
                    if coreml_file:
                        exported_files.append(coreml_file)

        if 'torchscript-lite' in opt.formats:
            tsl_file = export_torchscript_lite(model, img, opt.weights)
            if tsl_file:
                exported_files.append(tsl_file)

        if 'onnx' in opt.formats:
            onnx_file = export_onnx(model, img, opt.weights, opt, labels)
            if onnx_file:
                exported_files.append(onnx_file)

    except Exception as e:
        logger.error(f"Export failed: {str(e)}")
        return

    # Summary
    export_time = time.time() - t
    logger.info(f"\nExport complete ({export_time:.2f}s)")
    logger.info(f"Exported files: {exported_files}")
    logger.info("Visualize with https://github.com/lutzroeder/netron")


if __name__ == '__main__':
    main()