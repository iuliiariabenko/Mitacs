import csv
import json
import re
from typing import Dict, Any, List, Tuple

def normalize_key(key: str) -> str:
    if not key: return ""
    key = key.strip().lower()
    key = re.sub(r"[^a-z0-9]+", "_", key)
    return key.strip("_")

def to_float(value: Any) -> float | None:
    if value is None: return None
    try:
        s = str(value).strip().replace(',', '.')
        return float(s)
    except (ValueError, TypeError):
        return None

def extract_units(header: str) -> Tuple[str, str]:
    """
    Извлекает имя и юнит из строки типа 'Wavelength (nm)' или '%R'.
    Возвращает ('Wavelength', 'nm') или ('Reflectance', '%R').
    """
    header = header.strip()
    match = re.search(r"(.*?)\s*\((.*?)\)", header)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    
    # Если скобок нет (например, просто %R)
    if header == "%R": return "Reflectance", "%R"
    if header == "Abs": return "Absorbance", "Abs"
    return header, None

def parse_uvvis_full(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, encoding="latin-1", errors="ignore") as f:
        reader = list(csv.reader(f))

    if len(reader) < 2: return []

    # 1. Читаем имена образцов (1-я строка) и заголовки осей (2-я строка)
    sample_names = [name.strip() for name in reader[0] if name.strip()]
    headers_row = reader[1]

    # Создаем мапу: какой индекс колонки какому образцу принадлежит
    # Каждому образцу выделено 2 колонки (X и Y)
    sample_axis_info = {}
    for i, name in enumerate(sample_names):
        x_header = headers_row[i*2]
        y_header = headers_row[i*2 + 1]
        sample_axis_info[name] = {
            "x_name": extract_units(x_header)[0],
            "x_unit": extract_units(x_header)[1],
            "y_name": extract_units(y_header)[0],
            "y_unit": extract_units(y_header)[1],
        }

    # 2. Находим блоки метаданных
    blocks_data = []
    for i in range(len(reader)):
        row = reader[i]
        if row and row[0] in sample_names:
            if i + 2 < len(reader) and "Collection Time" in str(reader[i+2]):
                blocks_data.append((row[0], i))

    results = []

    # 3. Парсим блоки
    for j in range(len(blocks_data)):
        name, start_index = blocks_data[j]
        end_index = blocks_data[j+1][1] if j+1 < len(blocks_data) else len(reader)
        
        raw_meta = extract_metadata_from_rows(reader[start_index:end_index])
        raw_meta["internal_sample_name"] = name
        # Добавляем инфо об осях, которое мы вытащили из шапки файла
        raw_meta["axis_info"] = sample_axis_info.get(name, {})

        results.append({
            "experiment": {"name": raw_meta.get("project")}, # Если будет поле Project
            "sample": {"sample_name": name},
            "measurement": map_to_measurement(raw_meta),
            "spectrum_file": map_to_spectrum_file(raw_meta),
            "uvvis": map_to_uvvis_table(raw_meta)
        })

    return results

def extract_metadata_from_rows(rows: List[List[str]]) -> Dict[str, Any]:
    meta = {}
    mod_lines = []
    in_mods = False

    for row in rows:
        line = " ".join([c.strip() for c in row if c.strip()]).strip()
        if not line: continue

        # Логика Method Modifications
        if "Method Modifications:" in line:
            in_mods = True
            continue
        if "End Method Modifications" in line:
            in_mods = False
            continue
        if in_mods:
            mod_lines.append(line)
            continue

        if line.startswith("<") and ">" in line:
            parts = line.split(">")
            key = parts[0].replace("<", "")
            val = parts[1].replace(",", "").strip()
            meta[normalize_key(key)] = val
            continue

        if ":" in line:
            key, val = line.split(":", 1)
            meta[normalize_key(key)] = val.strip()
            continue

        parts = re.split(r'\s{2,}', line)
        if len(parts) >= 2:
            meta[normalize_key(parts[0])] = parts[1].strip()

    meta["_mod_lines"] = mod_lines
    return meta

def parse_modifications(lines: List[str]) -> List[Dict]:
    mods = []

    # Улучшенный паттерн:
    # [,\s]+ означает "один или более символов, которые могут быть запятой или пробелом"
    pattern = (
        r"^(.*?)\s+Changed:\s+"    # Параметр до "Changed:"
        r"(.+?)[,\s]+"             # Таймстамп до запятой/пробела перед Old
        r"Old:\s*(.*?)[,\s]+"      # Старое значение до запятой/пробела перед New
        r"New:\s*(.*)$"            # Новое значение до конца строки
    )

    for line in lines:
        m = re.match(pattern, line)
        if not m:
            # Если строка не подходит под паттерн лога, пропускаем её
            continue

        mods.append({
            "parameter": m.group(1).strip(),
            "timestamp": m.group(2).strip(),
            "old": to_float(m.group(3)) if to_float(m.group(3)) is not None else m.group(3).strip(),
            "new": to_float(m.group(4)) if to_float(m.group(4)) is not None else m.group(4).strip(),
        })

    return mods


# --- МАППИНГ ---

def map_to_measurement(m: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "software_version": m.get("scan_software_version"),
        "measurement_time": m.get("collection_time"),
        "operator": m.get("operator_name")
    }

def map_to_spectrum_file(m: Dict[str, Any]) -> Dict[str, Any]:
    axis = m.get("axis_info", {})
    ave_time = to_float(m.get("uv_vis_ave_time_sec"))

    return {
        "measurement_time": m.get("collection_time"),
        "x_axis_name": axis.get("x_name", "Wavelength"),
        "x_axis_unit": axis.get("x_unit", "nm"),
        "y_axis_name": axis.get("y_name", "Intensity"),
        "y_axis_unit": axis.get("y_unit", "Abs"),
        "x_start": to_float(m.get("start_nm")),
        "x_stop": to_float(m.get("stop_nm")),
        "resolution": to_float(m.get("uv_vis_data_interval_nm")),
        "acquisition_time": ave_time, # Время на точку
        "laser_wavelength": None,
        "excitation_wavelength": None,
        "instrument_metadata_json": json.dumps({
            "instrumentcarry": m.get("instrument"),
            "instrument_version": m.get("instrument_version"),
            "uv_source": m.get("uv_source"),
            "vis_source": m.get("vis_source"),
            "signal_to_noise_mode": m.get("signal_to_noise_mode"),
            "energy": to_float(m.get("energy")),
            "current_wavelength": to_float(m.get("current_wavelength")),
            "sbw": to_float(m.get("sbw_nm")),
            "baseline_type": m.get("baseline_type")
        }, ensure_ascii=False),
        "instrument_config_json": json.dumps({
            "spectrum_name": m.get("internal_sample_name"),
            "baseline_file_name": m.get("baseline_file_name"),
            "baseline_std_ref_file_name": m.get("baseline_std_ref_file_name"),
            "cycle_mode": m.get("cycle_mode"),
            "comments": m.get("comments")
        }, ensure_ascii=False)
    }

def map_to_uvvis_table(m: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "uvvis_mode": None, 
        "x_unit": m.get("x_mode"),
        "y_unit": m.get("y_mode"),
        "wl_start_nm": to_float(m.get("start_nm")),
        "wl_stop_nm": to_float(m.get("stop_nm")),
        "sbw_nm": to_float(m.get("sbw_nm")),
        "ave_time_s": to_float(m.get("uv_vis_ave_time_sec")),
        "data_interval_nm": to_float(m.get("uv_vis_data_interval_nm")),
        "scan_rate_nm_min": to_float(m.get("uv_vis_scan_rate_nm_min")),
        "beam_mode": m.get("beam_mode"),
        "baseline_correction": m.get("baseline_correction"),
        "source_changeover_nm": to_float(m.get("source_changeover_nm")),
        "method_log": json.dumps({
            "method_log": m.get("method_log"),
            "method_name": m.get("method_name"),
            "date_time_stamp": m.get("date_time_stamp"),
            "method_modifications": parse_modifications(m.get("_mod_lines", []))
        }, ensure_ascii=False)
    }
         
