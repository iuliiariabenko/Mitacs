import os
import json

# Имя файла всегда имеет вид:
# <technique>_<instrument>_<sample_block>_<mode_block>_<replicate>.<ext>
# где sample_block всегда одинаковой внутренней структуры:
# matrix-analyte-concValue concUnit-solvent-physicalState-container-treatment



def parse_concentration(raw: str | None):
    """
    Парсит концентрацию вида '0.8wt'.
    Возвращает (value, unit) или (None, None).
    """
    if raw is None:
        return None, None

    raw = raw.strip()
    if raw == "" or raw.upper() == "NA":
        return None, None

    value_part = ""
    unit_part = ""

    for ch in raw: # перебираем слитную запись по символам
        if ch.isdigit() or ch == ".": # если символ является числом или точкой, то
            value_part += ch # записываем это в переменную значения (величины)
        else: # иначе в переменную единицы 
            unit_part += ch

    if value_part == "":
        return None, None

    try:
        value = float(value_part)
    except ValueError:
        return None, None

    unit = unit_part if unit_part != "" else None
    return value, unit




def parse_filename(filename: str, strict_blocks: bool = False, strict_sample_block: bool = False) -> dict:
    """
    Парсит имя спектрального файла и извлекает семантические данные.
    Возвращает словарь, готовый к последующей записи в БД.
    """

    # ------------------------------------------------------------
    # 1. Отделяем имя файла от расширения
    # ------------------------------------------------------------
    name, ext = os.path.splitext(filename)
    file_format = ext.lstrip(".").lower()

    # ------------------------------------------------------------
    # 2. Разбиваем имя файла на логические блоки
    #    Пример:
    #    Raman_Horiba_Oil-BHT-0.8wt-NA-liquid-Q2-fresh_NA_R02
    # ------------------------------------------------------------
    parts = name.split("_")

    if len(parts) != 5: # Если структура состоит меньше, чем из 5-ти блоков
        raise ValueError(f"Некорректное имя файла: {filename}")

    # Проверка пустых блоков после split("_")
    if strict_blocks:
        for i, part in enumerate(parts):
            if part == "":
                raise ValueError(f"Пустой блок в имени файла (позиция {i+1}): '{filename}'")
    
    def normalize_block(value: str | None) -> str | None:
        if value is None:
            return None
        if value.strip() == "":
            return None
        return value

    technique = normalize_block(parts[0])
    instrument = normalize_block(parts[1])
    sample_block = normalize_block(parts[2])
    mode = normalize_block(parts[3])
    replicate_id = normalize_block(parts[4])

    # ------------------------------------------------------------
    # 3. Парсим sample_block
    #    Oil-BHT-0.8wt-NA-liquid-Q2-fresh
    # ------------------------------------------------------------
    sample_parts = sample_block.split("-") # разбивка блока по элементам

    if len(sample_parts) != 7:
        raise ValueError(f"Некорректный sample-блок: {sample_block}")

    if strict_sample_block:
        for i, part in enumerate(sample_parts):
            if part == "":
                raise ValueError(
                    f"Пустой элемент в sample_block (позиция {i+1}): '{sample_block}'"
                )
    
    matrix_code = sample_parts[0]
    analyte_code = sample_parts[1]

    # Концентрация всегда записана слитно: 0.8wt
    concentration_raw = sample_parts[2]
    concentration_value, concentration_unit = parse_concentration(concentration_raw)
    print(concentration_value, concentration_unit)

    solvent_code = sample_parts[3]
    physical_state = sample_parts[4]
    container_code = sample_parts[5]
    treatment_code = sample_parts[6]

    # ------------------------------------------------------------
    # 4. Формируем sample_semantic_json
    # ------------------------------------------------------------
    sample_semantics = {
        "matrix_code": matrix_code,
        "analyte_code": analyte_code,
        "solvent_code": solvent_code,
        "physical_state": physical_state,
        "container_code": container_code,
        "treatment_code": treatment_code
    }

    # ------------------------------------------------------------
    # 5. Собираем итоговую структуру
    # ------------------------------------------------------------
    parsed_data = {
        "measurement": {
            "technique": technique,
            "instrument": instrument,
            "mode": mode
        },
        "sample": {
            "concentration_value": concentration_value,
            "concentration_unit": concentration_unit,
            "replicate_id": replicate_id,
            "sample_semantics_json": json.dumps(sample_semantics, ensure_ascii=False)
        },
        "spectrum_file": {
            "file_name": filename,
            "file_format": file_format,
            "spectrum_type": mode
        }
    }

    return parsed_data
