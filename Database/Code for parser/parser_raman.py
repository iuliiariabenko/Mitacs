import re
import json
from typing import Dict, Any
import unicodedata

def parse_raman_metadata(file_path: str) -> Dict[str, Any]:
    """
    Читает метаданные Raman-файла и возвращает словарь.
    """
    metadata = {}

    # Кодировка, которая используется при создании файла через LabSpec (Horiba) может быть ISO-8859-1 (Latin-1) или cp1252, а не UTF-8. 
    with open(file_path, encoding="latin-1", errors="ignore") as f:
        for line in f:
            # Метаданные всегда начинаются с #
            if not line.startswith("#"):
                break

            if "=" not in line:
                continue

            # срезом пропускаем "#" и разделяем строку через "=" один раз
            raw_key, raw_value = line[1:].split("=", 1) 

            key = normalize_key(raw_key)
            value = normalize_value(raw_value)
            # после нормализации записываем в словарь ключ: значение
            metadata[key] = value

    return metadata


def normalize_key(key: str) -> str:
    """
    Приводит ключ к чистому виду: range_cm_1, slit_um и т.д.
    """
    key = key.strip().lower()
    
    # 1. Сначала нормализуем юникод (NFKC превращает '¹' в '1')
    key = unicodedata.normalize('NFKC', key)
    
    # 2. Ручные замены для надежности (особенно для микронов)
    replacements = {
        "¹": "1",
        "²": "2",
        "µ": "u",
        "μ": "u", # греческая мю
    }
    for old, new in replacements.items():
        key = key.replace(old, new)

    # 3. Заменяем все не-буквы и не-цифры на подчеркивание
    # Используем [^a-z0-9]+ чтобы точно оставить только латиницу и цифры
    key = re.sub(r"[^a-z0-9]+", "_", key)
    
    # 4. Убираем лишние подчеркивания по краям (например, в конце от закрывающей скобки)
    key = key.strip("_")
    
    return key


def normalize_value(value: str) -> str:
    value = value.strip()
    if value == "" or value.upper() == "NA":
        return None
    return value


# Парсер диапазона
def parse_range(range_str: str | None):
    if range_str is None:
        return None, None

    if "..." not in range_str:
        return None, None

    start, stop = range_str.split("...", 1)

    try:
        return float(start), float(stop)
    except ValueError:
        return None, None


# Парсер процентов
def parse_percent(value: str | None):
    if value is None:
        return None

    value = value.replace("%", "").strip()

    try:
        return float(value)
    except ValueError:
        return None


# Парсер grating_gmm
def parse_grating(value: str | None):
    if value is None:
        return None
    try:
        return int(value.split()[0]) # "1200 gr/mm" разбиваем по пробелу и получаем первый элемент списка, т.е. значение.
    except (ValueError, IndexError):
        return None


# Парсер mm:ss → секунды
def parse_mmss(value: str | None):
    if value is None:
        return None

    if ":" not in value:
        return None

    try:
        minutes, seconds = value.split(":")
        return int(minutes) * 60 + int(seconds)
    except ValueError:
        return None


# ---- Универсальные числовые конвертеры ----
def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def map_to_experiment(m: Dict[str, Any]) -> Dict[str, Any]:
    """#Project -> experiment.name"""
    return {
        "name": m.get("project")
    }

def map_to_sample(m: Dict[str, Any]) -> Dict[str, Any]:
    """#Sample -> sample.sample_name"""
    return {
        "sample_name": m.get("sample")
    }

def map_to_measurement(m: Dict[str, Any]) -> Dict[str, Any]:
    """#Detector temperature (°C) -> measurement.temperature"""
    return {
        "temperature": to_float(m.get("detector_temperature_c"))
    }


# Разбор данных под spectrum_file
def map_to_spectrum_file(metadata: Dict[str, Any]) -> Dict[str, Any]:
    range_min, range_max = parse_range(metadata.get("range_cm_1"))

    return {
        "measurement_time": metadata.get("acquired"),
        "x_axis_name": "raman_shift",
        "x_axis_unit": "1/cm",
        "y_axis_name": "intensity",
        "y_axis_unit": "cnt",
        "x_start": range_min,
        "x_stop": range_max,
        "resolution": to_float(metadata.get("spectral_res_cm_1")),
        "acquisition_time": parse_mmss(metadata.get("full_time_mm_ss")),
        "laser_wavelength": to_float(metadata.get("laser_nm")),
        "excitation_wavelength": to_float(metadata.get("laser_nm")),
        "instrument_metadata_json": json.dumps(extract_instrument_metadata(metadata)),
        "instrument_config_json": json.dumps(extract_instrument_config(metadata)),
    }


def extract_instrument_metadata(m: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "instrument": m.get("instrument"),
        "detector": m.get("detector"),
        "detector_gain": m.get("detector_gain"),
        "detector_adc": m.get("detector_adc"),
        "stagexy": m.get("stagexy"),
        "stagez": m.get("stagez"),
        "fiber": m.get("fiber"),
        "e_cal_neon": to_int(m.get("e_cal_neon")),
        "ac": m.get("ac"),
    }


def extract_instrument_config(m: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "delay_time": to_float(m.get("delay_time_s")),
        "binning": to_int(m.get("binning")),
        "site": m.get("site"),
        "title": m.get("title"),
        "remark": m.get("remark"),
        "date": m.get("date"),
    }


def map_to_raman_table(metadata: Dict[str, Any]) -> Dict[str, Any]:
    rmin, rmax = parse_range(metadata.get("range_cm_1"))

    return {
        "laser_nm": to_float(metadata.get("laser_nm")),
        "grating_gmm": parse_grating(metadata.get("grating")),
        "objective": metadata.get("objective"),
        "slit_um": to_float(metadata.get("slit_um")),
        "hole_um": to_float(metadata.get("hole_um")),
        "filter_percent": parse_percent(metadata.get("filter")),
        "acq_time_s": to_float(metadata.get("acq_time_s")),
        "accumulations": to_int(metadata.get("accumulations")),
        "range_cm1_min": rmin,
        "range_cm1_max": rmax,
        "spectral_res_cm1": to_float(metadata.get("spectral_res_cm_1")),
        "stage_x_um": to_float(metadata.get("x_um")),
        "stage_y_um": to_float(metadata.get("y_um")),
        "stage_z_um": to_float(metadata.get("z_um")),
        "corrections_flags": json.dumps(extract_corrections_flags(metadata)),
    }


def extract_corrections_flags(m: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "auto_scanning": m.get("auto_scanning") == "on",
        "autofocus": m.get("autofocus") == "on",
        "autoexposure": m.get("autoexposure") == "on",
        "spike_filter": m.get("spike_filter"),
        "readout_mode": m.get("readout_mode"),
        "denoise": m.get("denoise"),
        "ics_correction": m.get("ics_correction") == "on",
        "dark_correction": m.get("dark_correction") == "on",
        "inst_process": m.get("inst_process") == "on",
        "windows": to_int(m.get("windows")),
    }


def parse_raman_full(file_path: str) -> Dict[str, Any]:
    """
    Главная функция: читает файл и возвращает словарь, 
    разбитый по именам таблиц БД.
    """
    raw_metadata = parse_raman_metadata(file_path)
    
    if not raw_metadata:
        return {}

    return {
        "experiment": map_to_experiment(raw_metadata),
        "sample": map_to_sample(raw_metadata),
        "measurement": map_to_measurement(raw_metadata),
        "spectrum_file": map_to_spectrum_file(raw_metadata),
        "raman": map_to_raman_table(raw_metadata)
    }

