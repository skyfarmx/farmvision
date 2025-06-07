"""PyTorch Hub models for Farm Vision

Usage:
    import torch
    
    # Load fruit detection models
    model = torch.hub.load('farmvision', 'mandalina', pretrained=True)
    model = torch.hub.load('farmvision', 'elma', pretrained=True)
    
    # Custom model loading
    model = torch.hub.load('farmvision', 'custom', path='path/to/model.pt')
    
    # Multi-fruit detector
    model = torch.hub.load('farmvision', 'multi_fruit', pretrained=True)
"""

from pathlib import Path
import logging

import torch

from models.yolo import Model
from utils.general import check_requirements, set_logging
from utils.google_utils import attempt_download
from utils.torch_utils import select_device

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dependencies = ['torch', 'yaml', 'opencv-python', 'pillow', 'numpy']
check_requirements(Path(__file__).parent / 'requirements.txt', exclude=('pycocotools', 'thop'))
set_logging()

# Farm Vision fruit classes mapping
FRUIT_CLASSES = {
    'mandalina': {'classes': 1, 'names': ['mandalina'], 'weight': 0.125},
    'elma': {'classes': 1, 'names': ['elma'], 'weight': 0.105},
    'armut': {'classes': 1, 'names': ['armut'], 'weight': 0.220},
    'seftali': {'classes': 1, 'names': ['seftali'], 'weight': 0.185},
    'nar': {'classes': 1, 'names': ['nar'], 'weight': 0.300},
    'hurma': {'classes': 1, 'names': ['hurma'], 'weight': 0.150},
    'multi_fruit': {
        'classes': 6, 
        'names': ['mandalina', 'elma', 'armut', 'seftali', 'nar', 'hurma'],
        'weights': [0.125, 0.105, 0.220, 0.185, 0.300, 0.150]
    }
}


def create(name, pretrained, channels, classes, autoshape, fruit_type=None):
    """Creates a specified Farm Vision model

    Arguments:
        name (str): name of model, i.e. 'yolov7'
        pretrained (bool): load pretrained weights into the model
        channels (int): number of input channels
        classes (int): number of model classes
        autoshape (bool): apply autoshape wrapper
        fruit_type (str): specific fruit type for specialized models

    Returns:
        pytorch model with Farm Vision optimizations
    """
    try:
        # Try to find model config
        config_paths = [
            Path(__file__).parent / 'cfg' / f'{name}.yaml',
            Path(__file__).parent / 'cfg' / 'yolov7.yaml'  # fallback
        ]
        
        cfg = None
        for config_path in config_paths:
            if config_path.exists():
                cfg = config_path
                break
        
        if cfg is None:
            raise FileNotFoundError(f"No config found for model: {name}")
            
        logger.info(f"Loading model config: {cfg}")
        model = Model(cfg, channels, classes)
        
        if pretrained:
            # Try fruit-specific weights first, then generic
            weight_options = [
                f'{name}.pt',
                f'{fruit_type}.pt' if fruit_type else None,
                'yolov7.pt'  # fallback
            ]
            
            loaded = False
            for fname in weight_options:
                if fname is None:
                    continue
                    
                try:
                    weight_path = Path(fname)
                    if weight_path.exists():
                        logger.info(f"Loading weights: {fname}")
                    else:
                        logger.info(f"Attempting to download: {fname}")
                        attempt_download(fname)
                    
                    ckpt = torch.load(fname, map_location=torch.device('cpu'))
                    msd = model.state_dict()
                    csd = ckpt['model'].float().state_dict()
                    csd = {k: v for k, v in csd.items() if k in msd and msd[k].shape == v.shape}
                    model.load_state_dict(csd, strict=False)
                    
                    # Set class names
                    if hasattr(ckpt['model'], 'names') and len(ckpt['model'].names) == classes:
                        model.names = ckpt['model'].names
                    elif fruit_type and fruit_type in FRUIT_CLASSES:
                        model.names = FRUIT_CLASSES[fruit_type]['names']
                    
                    logger.info(f"Successfully loaded weights from: {fname}")
                    loaded = True
                    break
                    
                except Exception as e:
                    logger.warning(f"Failed to load {fname}: {str(e)}")
                    continue
            
            if not loaded:
                logger.warning("No pretrained weights loaded, using random initialization")
        
        # Apply autoshape wrapper
        if autoshape:
            model = model.autoshape()
            
        # Add Farm Vision specific attributes
        if fruit_type and fruit_type in FRUIT_CLASSES:
            model.fruit_type = fruit_type
            model.fruit_weight = FRUIT_CLASSES[fruit_type].get('weight', 0.15)
            model.fruit_weights = FRUIT_CLASSES[fruit_type].get('weights', [])
        
        # Select device
        device = select_device('0' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        logger.info(f"Model created successfully on device: {device}")
        return model

    except Exception as e:
        error_msg = f'Failed to create model {name}: {str(e)}. Try force_reload=True.'
        logger.error(error_msg)
        raise Exception(error_msg) from e


def custom(path_or_model='path/to/model.pt', autoshape=True, fruit_type=None):
    """Load custom Farm Vision model

    Arguments:
        path_or_model (str/dict/nn.Module): model source
        autoshape (bool): apply autoshape wrapper
        fruit_type (str): fruit type for weight calculation

    Returns:
        pytorch model
    """
    try:
        logger.info(f"Loading custom model: {path_or_model}")
        
        # Load model
        if isinstance(path_or_model, str):
            if not Path(path_or_model).exists():
                raise FileNotFoundError(f"Model file not found: {path_or_model}")
            model = torch.load(path_or_model, map_location=torch.device('cpu'))
        else:
            model = path_or_model
        
        # Extract model from checkpoint
        if isinstance(model, dict):
            model = model['ema' if model.get('ema') else 'model']

        # Create hub model
        hub_model = Model(model.yaml).to(next(model.parameters()).device)
        hub_model.load_state_dict(model.float().state_dict())
        hub_model.names = model.names
        
        # Add Farm Vision attributes
        if fruit_type and fruit_type in FRUIT_CLASSES:
            hub_model.fruit_type = fruit_type
            hub_model.fruit_weight = FRUIT_CLASSES[fruit_type].get('weight', 0.15)
        
        if autoshape:
            hub_model = hub_model.autoshape()
        
        device = select_device('0' if torch.cuda.is_available() else 'cpu')
        hub_model = hub_model.to(device)
        
        logger.info("Custom model loaded successfully")
        return hub_model
        
    except Exception as e:
        logger.error(f"Failed to load custom model: {str(e)}")
        raise


# Farm Vision fruit detection models
def mandalina(pretrained=True, channels=3, autoshape=True):
    """Mandalina (Mandarin) detection model"""
    return create('yolov7', pretrained, channels, FRUIT_CLASSES['mandalina']['classes'], autoshape, 'mandalina')


def elma(pretrained=True, channels=3, autoshape=True):
    """Elma (Apple) detection model"""
    return create('yolov7', pretrained, channels, FRUIT_CLASSES['elma']['classes'], autoshape, 'elma')


def armut(pretrained=True, channels=3, autoshape=True):
    """Armut (Pear) detection model"""
    return create('yolov7', pretrained, channels, FRUIT_CLASSES['armut']['classes'], autoshape, 'armut')


def seftali(pretrained=True, channels=3, autoshape=True):
    """Şeftali (Peach) detection model"""
    return create('yolov7', pretrained, channels, FRUIT_CLASSES['seftali']['classes'], autoshape, 'seftali')


def nar(pretrained=True, channels=3, autoshape=True):
    """Nar (Pomegranate) detection model"""
    return create('yolov7', pretrained, channels, FRUIT_CLASSES['nar']['classes'], autoshape, 'nar')


def hurma(pretrained=True, channels=3, autoshape=True):
    """Hurma (Persimmon) detection model"""
    return create('yolov7', pretrained, channels, FRUIT_CLASSES['hurma']['classes'], autoshape, 'hurma')


def multi_fruit(pretrained=True, channels=3, autoshape=True):
    """Multi-fruit detection model (all fruits)"""
    return create('yolov7', pretrained, channels, FRUIT_CLASSES['multi_fruit']['classes'], autoshape, 'multi_fruit')


# Legacy compatibility
def yolov7(pretrained=True, channels=3, classes=80, autoshape=True):
    """Standard YOLOv7 model"""
    return create('yolov7', pretrained, channels, classes, autoshape)


def calculate_fruit_weight(detections, fruit_type):
    """Calculate total fruit weight from detections
    
    Arguments:
        detections: model detection results
        fruit_type (str): type of fruit detected
        
    Returns:
        dict: count and weight information
    """
    if fruit_type not in FRUIT_CLASSES:
        logger.warning(f"Unknown fruit type: {fruit_type}")
        return {'count': 0, 'weight': 0.0}
    
    try:
        # Count detections
        if hasattr(detections, 'pandas'):
            df = detections.pandas().xyxy[0]
            count = len(df)
        else:
            count = len(detections)
        
        # Calculate weight
        unit_weight = FRUIT_CLASSES[fruit_type]['weight']
        total_weight = count * unit_weight
        
        result = {
            'count': count,
            'unit_weight_kg': unit_weight,
            'total_weight_kg': total_weight,
            'fruit_type': fruit_type
        }
        
        logger.info(f"Detection results: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Weight calculation error: {str(e)}")
        return {'count': 0, 'weight': 0.0}


if __name__ == '__main__':
    # Example usage
    logger.info("Testing Farm Vision models...")
    
    try:
        # Test mandalina model
        model = mandalina(pretrained=True)
        logger.info(f"Mandalina model loaded: {model.names}")
        
        # Test custom model
        # model = custom(path_or_model='mandalina.pt', fruit_type='mandalina')
        
        # Verify inference
        import numpy as np
        
        # Create test image
        test_img = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # Run inference
        results = model(test_img)
        logger.info("Inference test successful")
        
        # Calculate weight
        weights = calculate_fruit_weight(results, 'mandalina')
        logger.info(f"Weight calculation: {weights}")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")