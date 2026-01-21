"""
YOLOv8 model eğitim scripti.

Usage:
    python scripts/train_yolo.py --data configs/strawberry_data.yaml --config configs/train_config.yaml
    python scripts/train_yolo.py --data datasets/processed/data.yaml --epochs 100 --batch 16
"""

import argparse
import logging
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Config dosyasını yükler."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Config yüklenemedi: {e}")
        return {}


def train_yolo(data_yaml: str, config: Dict[str, Any]) -> bool:
    """YOLOv8 modelini eğitir.
    
    Args:
        data_yaml: Dataset config dosya yolu
        config: Eğitim konfigürasyonu
        
    Returns:
        Başarılı ise True
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("Ultralytics yüklü değil. 'pip install ultralytics' ile yükleyin.")
        return False
    
    try:
        model_name = config.get('model', 'yolov8n.pt')
        logger.info(f"Model yükleniyor: {model_name}")
        model = YOLO(model_name)
        
        train_args = {
            'data': data_yaml,
            'epochs': config.get('epochs', 100),
            'batch': config.get('batch', 16),
            'imgsz': config.get('imgsz', 640),
            'device': config.get('device', 0),
            'workers': config.get('workers', 8),
            'optimizer': config.get('optimizer', 'AdamW'),
            'lr0': config.get('lr0', 0.01),
            'lrf': config.get('lrf', 0.01),
            'momentum': config.get('momentum', 0.937),
            'weight_decay': config.get('weight_decay', 0.0005),
            'box': config.get('box', 7.5),
            'cls': config.get('cls', 0.5),
            'dfl': config.get('dfl', 1.5),
            'hsv_h': config.get('hsv_h', 0.015),
            'hsv_s': config.get('hsv_s', 0.7),
            'hsv_v': config.get('hsv_v', 0.4),
            'degrees': config.get('degrees', 10.0),
            'translate': config.get('translate', 0.1),
            'scale': config.get('scale', 0.5),
            'shear': config.get('shear', 0.0),
            'perspective': config.get('perspective', 0.0),
            'flipud': config.get('flipud', 0.0),
            'fliplr': config.get('fliplr', 0.5),
            'mosaic': config.get('mosaic', 1.0),
            'mixup': config.get('mixup', 0.1),
            'copy_paste': config.get('copy_paste', 0.0),
            'val': config.get('val', True),
            'save': config.get('save', True),
            'save_period': config.get('save_period', 10),
            'plots': config.get('plots', True),
            'conf': config.get('conf', 0.25),
            'iou': config.get('iou', 0.7),
            'patience': config.get('patience', 50),
            'resume': config.get('resume', False),
            'amp': config.get('amp', True),
            'fraction': config.get('fraction', 1.0),
            'profile': config.get('profile', False),
            'freeze': config.get('freeze', None),
            'multi_scale': config.get('multi_scale', False),
            'project': config.get('project', 'runs/train'),
            'name': config.get('name', 'strawberry_exp'),
            'exist_ok': config.get('exist_ok', False),
            'pretrained': config.get('pretrained', True),
            'verbose': config.get('verbose', True),
            'seed': config.get('seed', 0),
            'deterministic': config.get('deterministic', True),
            'single_cls': config.get('single_cls', False),
            'rect': config.get('rect', False),
            'cos_lr': config.get('cos_lr', False),
            'close_mosaic': config.get('close_mosaic', 10),
        }
        
        logger.info("Eğitim başlıyor...")
        logger.info(f"Parametreler: epochs={train_args['epochs']}, batch={train_args['batch']}, imgsz={train_args['imgsz']}")
        
        results = model.train(**train_args)
        
        logger.info("Eğitim tamamlandı!")
        logger.info(f"Sonuçlar: {train_args['project']}/{train_args['name']}")
        
        metrics = results.results_dict if hasattr(results, 'results_dict') else {}
        if metrics:
            logger.info(f"Final Metrics:")
            logger.info(f"  mAP@0.5: {metrics.get('metrics/mAP50(B)', 'N/A')}")
            logger.info(f"  mAP@0.5:0.95: {metrics.get('metrics/mAP50-95(B)', 'N/A')}")
        
        best_model_path = Path(train_args['project']) / train_args['name'] / 'weights' / 'best.pt'
        if best_model_path.exists():
            logger.info(f"✅ En iyi model: {best_model_path}")
        
        return True
    except Exception as e:
        logger.error(f"Eğitim hatası: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_data_yaml(data_yaml: str) -> bool:
    """data.yaml dosyasını doğrular."""
    if not os.path.exists(data_yaml):
        logger.error(f"data.yaml bulunamadı: {data_yaml}")
        return False
    
    try:
        with open(data_yaml, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)
        
        required_keys = ['train', 'val', 'nc', 'names']
        for key in required_keys:
            if key not in data_config:
                logger.error(f"data.yaml'da eksik key: {key}")
                return False
        
        logger.info(f"Dataset config doğrulandı: {data_config['nc']} sınıf")
        logger.info(f"Sınıflar: {data_config['names']}")
        
        return True
    except Exception as e:
        logger.error(f"data.yaml doğrulama hatası: {e}")
        return False


def resolve_default_data_yaml() -> Optional[str]:
    """Google Drive'a yüklenen dataset'i otomatik bulur.
    Kontrol sırası:
    1) Env: DRIVE_DATA_YAML
    2) configs/drive_dir.txt + dataset/data.yaml
    3) Colab varsayılan: /content/drive/MyDrive/StrawberryVision/dataset/data.yaml
    4) Lokal fallback: datasets/roboflow/data.yaml
    """
    # 1) Explicit env
    env_path = os.environ.get("DRIVE_DATA_YAML", "").strip()
    if env_path and os.path.exists(env_path):
        return env_path

    # 2) configs/drive_dir.txt
    try:
        drive_file = Path("configs") / "drive_dir.txt"
        if drive_file.exists():
            drive_dir = drive_file.read_text(encoding="utf-8").strip()
            if drive_dir:
                candidate = Path(os.path.expandvars(os.path.expanduser(drive_dir))) / "dataset" / "data.yaml"
                if candidate.exists():
                    return str(candidate)
    except Exception:
        pass

    # 3) Colab default
    colab_candidate = Path("/content/drive/MyDrive/StrawberryVision/dataset/data.yaml")
    if colab_candidate.exists():
        return str(colab_candidate)

    # 4) Lokal fallback
    local_candidate = Path("datasets/roboflow/data.yaml")
    if local_candidate.exists():
        return str(local_candidate)

    return None


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 model eğitimi")
    parser.add_argument("--data", type=str, required=False, help="Dataset YAML dosyası")
    parser.add_argument("--config", type=str, default=None, help="Eğitim config YAML dosyası")
    
    parser.add_argument("--model", type=str, default=None, help="Model adı (yolov8n.pt, yolov8s.pt, ...)")
    parser.add_argument("--epochs", type=int, default=None, help="Epoch sayısı")
    parser.add_argument("--batch", type=int, default=None, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=None, help="Görüntü boyutu")
    parser.add_argument("--device", type=str, default=None, help="Device (0, 1, cpu)")
    parser.add_argument("--name", type=str, default=None, help="Experiment adı")
    
    args = parser.parse_args()
    
    data_yaml_path = args.data
    if not data_yaml_path:
        logger.info("--data verilmedi. Google Drive ve yerel konumlardan data.yaml otomatik aranıyor...")
        data_yaml_path = resolve_default_data_yaml()
        if data_yaml_path:
            logger.info(f"Bulunan dataset: {data_yaml_path}")
        else:
            logger.error(
                "Dataset bulunamadı. Aşağıdakilerden birini yapın:\n"
                " - --data ile data.yaml yolunu verin\n"
                " - configs/drive_dir.txt içinde Drive klasörünü (StrawberryVision) tanımlayın ve dataset/data.yaml mevcut olsun\n"
                " - Colab'ta /content/drive/MyDrive/StrawberryVision/dataset/data.yaml yolunu kullanın\n"
                " - veya datasets/roboflow/data.yaml oluşturun"
            )
            return 1

    if not validate_data_yaml(data_yaml_path):
        return 1
    
    if args.config:
        config = load_config(args.config)
        logger.info(f"Config yüklendi: {args.config}")
    else:
        config = {}
        logger.info("Varsayılan config kullanılıyor")
    
    if args.model:
        config['model'] = args.model
    if args.epochs:
        config['epochs'] = args.epochs
    if args.batch:
        config['batch'] = args.batch
    if args.imgsz:
        config['imgsz'] = args.imgsz
    if args.device:
        config['device'] = args.device
    if args.name:
        config['name'] = args.name
    
    success = train_yolo(data_yaml_path, config)
    
    if success:
        logger.info("✅ Eğitim başarıyla tamamlandı!")
        logger.info("📝 Sonraki adım: Model değerlendirme ve test")
    else:
        logger.error("❌ Eğitim başarısız!")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
