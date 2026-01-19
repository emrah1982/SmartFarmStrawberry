"""
Roboflow'dan dataset indirme scripti.

Usage:
    python scripts/download_dataset.py --api-key YOUR_KEY --workspace strawberry --project ripeness --version 1
"""

import argparse
import logging
import os
from pathlib import Path
import time
import io
import zipfile
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def download_roboflow_dataset(api_key: str, workspace: str, project: str, version: int, output_dir: str) -> bool:
    """Roboflow'dan dataset indirir.
    
    Args:
        api_key: Roboflow API key
        workspace: Workspace adı
        project: Proje adı
        version: Dataset versiyonu
        output_dir: İndirme dizini
        
    Returns:
        Başarılı ise True
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        logger.error("Roboflow paketi yüklü değil. 'pip install roboflow' ile yükleyin.")
        return False
    
    try:
        logger.info(f"Roboflow'a bağlanılıyor: {workspace}/{project}")
        rf = Roboflow(api_key=api_key)
        
        project_obj = rf.workspace(workspace).project(project)
        logger.info(f"Proje bulundu: {project}")
        
        dataset = project_obj.version(version).download("yolov8", location=output_dir)
        logger.info(f"Dataset indirildi: {output_dir}")
        
        return True
    except Exception as e:
        logger.error(f"Dataset indirme hatası: {e}")
        return False


def download_universe_url(url: str, output_dir: str, retries: int = 3, timeout: int = 120) -> bool:
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        last_err = None
        for attempt in range(1, retries + 1):
            try:
                logger.info(f"ZIP indiriliyor (deneme {attempt}/{retries})...")
                r = requests.get(url, stream=True, timeout=timeout)
                r.raise_for_status()
                with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                    zf.extractall(output_dir)
                logger.info(f"ZIP çıkarıldı: {output_dir}")
                return True
            except Exception as e:
                last_err = e
                wait = min(5 * attempt, 15)
                logger.warning(f"İndirme hatası: {e}. {wait}s sonra yeniden denenecek...")
                time.sleep(wait)
        if last_err:
            logger.error(f"URL'den indirme başarısız: {last_err}")
        return False
    except Exception as e:
        logger.error(f"URL indirme hatası: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Roboflow'dan dataset indir")
    parser.add_argument("--api-key", type=str, help="Roboflow API key")
    parser.add_argument("--workspace", type=str, help="Workspace adı")
    parser.add_argument("--project", type=str, help="Proje adı")
    parser.add_argument("--version", type=int, default=1, help="Dataset versiyonu")
    parser.add_argument("--url", type=str, help="Roboflow Universe direct download URL (ds/... ?key=...)")
    parser.add_argument("--output", type=str, default="datasets/roboflow", help="İndirme dizini")
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    success = False

    if args.url:
        logger.info("Doğrudan URL ile indirme başlatılıyor")
        success = download_universe_url(args.url, args.output)
    else:
        if not args.api_key or not args.workspace or not args.project:
            logger.error("API yöntemi için --api-key, --workspace ve --project zorunludur veya --url kullanın")
            return 1
        success = download_roboflow_dataset(
            api_key=args.api_key,
            workspace=args.workspace,
            project=args.project,
            version=args.version,
            output_dir=args.output,
        )
    
    if success:
        logger.info("✅ Dataset başarıyla indirildi!")
        logger.info(f"📁 Konum: {args.output}")
        logger.info("📝 Sonraki adım: python scripts/relabel_dataset.py")
    else:
        logger.error("❌ Dataset indirilemedi!")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
