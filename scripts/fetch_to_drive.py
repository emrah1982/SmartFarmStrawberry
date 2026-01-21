"""
Roboflow Universe URL'inden dataset'i indirip Google Drive senkron klasörüne çıkarır.

Okunan dosyalar:
- configs/roboflow_url.txt : Roboflow Universe direkt indirme URL'i (https://universe.roboflow.com/ds/... ?key=...)
- configs/drive_dir.txt    : Hedef Drive klasör yolu (örn: C:\\Users\\User\\Google Drive\\My Drive\\StrawberryVision)

Kullanım:
    python scripts/fetch_to_drive.py

Notlar:
- URL'e format=yolov8 parametresi ekli değilse otomatik eklenir.
- Dataset zip'i indirilir ve hedef klasöre çıkarılır.
- Hedef klasör yoksa oluşturulur.
"""

import io
import os
import sys
import time
import zipfile
import logging
from pathlib import Path
from typing import Tuple

import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CONFIGS_DIR = Path("configs")
URL_FILE = CONFIGS_DIR / "roboflow_url.txt"
DRIVE_DIR_FILE = CONFIGS_DIR / "drive_dir.txt"


def read_text_file(path: Path) -> str:
    try:
        if not path.exists():
            raise FileNotFoundError(f"Dosya bulunamadı: {path}")
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            raise ValueError(f"Dosya boş: {path}")
        return content
    except Exception as e:
        raise RuntimeError(f"{path} okunamadı: {e}")


def normalize_universe_url(url: str) -> str:
    # format=yolov8 parametresi yoksa ekle
    if "format=" not in url:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}format=yolov8"
    return url


def download_and_extract_zip(url: str, output_dir: Path, retries: int = 3, timeout: int = 120) -> bool:
    output_dir.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    headers = {
        "User-Agent": "SmartFarmStrawberry/1.0 (+https://github.com/emrah1982)",
        "Accept": "*/*",
    }
    for attempt in range(1, retries + 1):
        try:
            logger.info(f"ZIP indiriliyor (deneme {attempt}/{retries})...")
            with requests.get(url, stream=True, timeout=timeout, headers=headers) as r:
                r.raise_for_status()
                ctype = r.headers.get("Content-Type", "")
                # ZIP olmayan yanıtları erken teşhis et
                if "zip" not in ctype.lower() and "application/octet-stream" not in ctype.lower():
                    # Roboflow bazen HTML döndürebilir; ilk birkaç yüz baytı logla
                    preview = r.iter_content(chunk_size=512)
                    first_chunk = next(preview, b"")
                    logger.error(
                        "Beklenmeyen içerik tipi: %s. İlk baytlar: %s",
                        ctype,
                        first_chunk[:120].decode(errors="ignore"),
                    )
                    raise RuntimeError(f"Beklenmeyen içerik tipi: {ctype}")

                # Geçici dosyaya yaz, sonra çıkar
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            tmp.write(chunk)
                    tmp_path = Path(tmp.name)
                try:
                    with zipfile.ZipFile(tmp_path) as zf:
                        zf.extractall(str(output_dir))
                    logger.info(f"ZIP çıkarıldı: {output_dir}")
                    return True
                finally:
                    try:
                        tmp_path.unlink(missing_ok=True)
                    except Exception:
                        pass
        except Exception as e:
            last_err = e
            wait = min(5 * attempt, 15)
            logger.warning(f"İndirme/çıkarma hatası: {e}. {wait}s sonra yeniden denenecek...")
            time.sleep(wait)
    if last_err:
        logger.error(f"URL'den indirme başarısız: {last_err}")
    return False


def resolve_paths() -> Tuple[str, Path]:
    url = read_text_file(URL_FILE)
    drive_dir_raw = read_text_file(DRIVE_DIR_FILE)

    # Ortam değişkenlerini genişlet ve ~ çözümlensin
    drive_dir = Path(os.path.expandvars(os.path.expanduser(drive_dir_raw))).resolve()
    return url, drive_dir


def main() -> int:
    try:
        url, drive_dir = resolve_paths()
    except Exception as e:
        logger.error(e)
        return 1

    url = normalize_universe_url(url)

    # İndirme konumu: drive_dir/dataset (karışıklığı önlemek için alt klasör)
    output_dir = drive_dir / "dataset"
    logger.info(f"Hedef klasör: {output_dir}")
    logger.info(f"İndirme URL'i: {url}")

    ok = download_and_extract_zip(url, output_dir)
    if not ok:
        logger.error("❌ Dataset indirilemedi!")
        return 1

    # data.yaml dosyasını üst seviyeye kopyalamak isteyebiliriz; ancak şimdilik olduğu yerde bırakıyoruz.
    data_yaml_candidates = list(output_dir.rglob("data.yaml"))
    if data_yaml_candidates:
        # En uzun (en derin) yolu seçelim ve kullanıcıya bildirip yolu yazalım
        data_yaml = sorted(data_yaml_candidates, key=lambda p: len(p.as_posix()))[-1]
        logger.info(f"✅ Dataset hazır. data.yaml: {data_yaml}")
    else:
        logger.warning("⚠️ data.yaml bulunamadı. Roboflow paket formatını doğrulayın (format=yolov8).")

    logger.info("\n📝 Eğitim için örnek komut:")
    logger.info(f"python scripts/train_yolo.py --data \"{(output_dir / 'data.yaml')}\" --config configs/train_config.yaml")

    return 0


if __name__ == "__main__":
    sys.exit(main())
