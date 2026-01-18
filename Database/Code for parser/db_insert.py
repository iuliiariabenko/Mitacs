import sqlite3
from typing import Dict, Any, Tuple, List


# ------------------------------------------------------------
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ------------------------------------------------------------

def normalize_value(value: Any) -> Any:
    """
    Преобразует NA-подобные значения в None (→ NULL в БД).
    """
    if value is None:
        return None

    if isinstance(value, str):
        v_strip = value.strip()
        # Если строка пустая или равна "NA", возвращаем None
        if v_strip == "" or v_strip.upper() == "NA":
            return None
        return v_strip # Возвращаем строку без лишних пробелов по краям
    
    return value


def get_table_schema(cursor, table_name: str) -> Dict[str, Dict]:
    """
    Читает реальную схему таблицы из SQLite.
    Возвращает словарь:
    {
        column_name: {
            "type": "...",
            "notnull": bool
        }
    }
    """
    cursor.execute(f"PRAGMA table_info({table_name})")
    schema = {}

    for cid, name, col_type, notnull, default, pk in cursor.fetchall():
        schema[name] = {
            "type": col_type.upper(),
            "notnull": bool(notnull)
        }

    return schema


def python_type_matches_sql(value: Any, sql_type: str) -> bool:
    """
    Проверяет совместимость Python-типа со столбцом SQLite.
    """
    if value is None:
        return True

    if "INT" in sql_type:
        return isinstance(value, int) 
    # isinstance(object, classinfo) 
    # Возвращает True, если object является экземпляром classinfo, его подкласса, или если object является одним из типов в classinfo (если classinfo — кортеж).

    if "REAL" in sql_type:
        return isinstance(value, (int, float))

    if "TEXT" in sql_type:
        return isinstance(value, str)

    return True


# ------------------------------------------------------------
# ВАЛИДАЦИЯ parsed_data ПО РЕАЛЬНОЙ СХЕМЕ БД
# ------------------------------------------------------------

def validate_parsed_data(cursor, parsed_data: Dict):
    """
    Проверяет: наличие обязательных секций, соответствие полей реальной схеме БД, NOT NULL ограничения и типы данных.
    """

    required_sections = ["sample", "measurement", "spectrum_file"] # используемые таблицы

    for key in required_sections:
        if key not in parsed_data:
            raise ValueError(f"parsed_data не содержит секцию '{key}'")
    
    # получаем данные о таблицах
    sample_schema = get_table_schema(cursor, "sample")
    measurement_schema = get_table_schema(cursor, "measurement")
    spectrum_schema = get_table_schema(cursor, "spectrum_file")
    
    
    # функция для проверки секции по полям и значениям полей
    def validate_section(section_data, schema, table_name, ignore_fields):
        for field, value in section_data.items():
            if field not in schema:
                raise ValueError(
                    f"Поле '{field}' отсутствует в таблице '{table_name}'"
                )

            normalized = normalize_value(value) # NA --> None

            if schema[field]["notnull"] and normalized is None:
                raise ValueError(
                    f"Поле '{table_name}.{field}' не может быть NULL"
                )

            # если тип данных колонки таблицы не совпадает с нормализированным значением (не рассматриваем None, ведь тогда сразу вернет True и мы пропустим условие, так как not True = False) вернет False. В итоге сработает условие.
            # напоминаю, что schema = { column_name: { "type": "...", "notnull": bool }, column_name_1: ... }
            # field = column_name
            if not python_type_matches_sql(normalized, schema[field]["type"]):
                raise TypeError(
                    f"Неверный тип для '{table_name}.{field}': "
                    f"{type(value)} → {schema[field]['type']}"
                )

        # Проверка обязательных NOT NULL полей
        for field, meta in schema.items():
            # если колонка имеет условие NOT NULL и эта колонка не находится в списке на игнорирование 
            if meta["notnull"] and field not in ignore_fields:
                # если колонка не находится в распарченных данных
                if field not in section_data:
                    raise ValueError(
                        f"Обязательное поле '{table_name}.{field}' отсутствует"
                    )

    validate_section(
        parsed_data["sample"],
        sample_schema,
        "sample",
        ignore_fields={"sample_id", "experiment_id"}
    )

    validate_section(
        parsed_data["measurement"],
        measurement_schema,
        "measurement",
        ignore_fields={"measurement_id", "sample_id"}
    )

    validate_section(
        parsed_data["spectrum_file"],
        spectrum_schema,
        "spectrum_file",
        ignore_fields={"spectrum_id", "measurement_id"}
    )


# ------------------------------------------------------------
# Основная функция вставки данных
# ------------------------------------------------------------

def insert_from_parsed_filename(
    parsed_data: Dict,
    file_path: str,
    db_name: str = "laboratory_data.db",
    experiment_id: int | None = None
) -> Tuple[int, int, int]:
    """
    Вставляет данные в таблицы sample, measurement и spectrum_file и возвращает (sample_id, measurement_id, spectrum_id).
    Связи создаются автоматически.
    """

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    try:
        validate_parsed_data(cursor, parsed_data)
        # ----------------------------------------------------
        # SAMPLE
        # ----------------------------------------------------
        sample_data = {
            k: normalize_value(v)
            for k, v in parsed_data["sample"].items()
        } # генератор словаря 

        cursor.execute(
            """
            INSERT INTO sample (
                experiment_id,
                concentration_value,
                concentration_unit,
                replicate_id,
                sample_semantics_json
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                sample_data["concentration_value"],
                sample_data["concentration_unit"],
                sample_data["replicate_id"],
                sample_data["sample_semantics_json"]
            )
        )

        sample_id = cursor.lastrowid # получение ID только что добавленной записи в таблицу sample

        # ----------------------------------------------------
        # MEASUREMENT
        # ----------------------------------------------------
        measurement_data = {
            k: normalize_value(v)
            for k, v in parsed_data["measurement"].items()
        }

        cursor.execute(
            """
            INSERT INTO measurement (
                sample_id,
                technique,
                instrument,
                mode
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                sample_id,
                measurement_data["technique"],
                measurement_data["instrument"],
                measurement_data["mode"]
            )
        )

        measurement_id = cursor.lastrowid # получение ID только что добавленной записи в таблицу measurement

        # ----------------------------------------------------
        # SPECTRUM FILE
        # ----------------------------------------------------
        spectrum_data = {
            k: normalize_value(v)
            for k, v in parsed_data["spectrum_file"].items()
        }

        cursor.execute(
            """
            INSERT INTO spectrum_file (
                measurement_id,
                spectrum_type,
                file_path,
                file_name,
                file_format
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                measurement_id,
                spectrum_data["spectrum_type"],
                file_path,
                spectrum_data["file_name"],
                spectrum_data["file_format"]
            )
        )

        spectrum_id = cursor.lastrowid # получение ID только что добавленной записи в таблицу spectrum
        
        conn.commit()
        
        return sample_id, measurement_id, spectrum_id
    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


# Так как Raman файл содержит данные одного спектра, то и запись у него одна, притом та, которая уже вмещает себя значения с имени файла, поэтому эта функция просто дополняет существующие записи и создает запись в таблице Raman
def enrich_raman_data(
    db_name: str,
    sample_id: int,
    measurement_id: int,
    spectrum_id: int,
    content_parsed: Dict[str, Any]
):
    """
    Дополняет уже созданные записи данными из файла.
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    try:
        # 1. Дополняем Sample (sample_name)
        if content_parsed.get("sample"):
            cursor.execute("UPDATE sample SET sample_name = ? WHERE sample_id = ?", 
                           (content_parsed["sample"]["sample_name"], sample_id))

        # 2. Дополняем Measurement (temperature)
        if content_parsed.get("measurement"):
            cursor.execute("UPDATE measurement SET temperature = ? WHERE measurement_id = ?", 
                           (content_parsed["measurement"]["temperature"], measurement_id))

        # 3. Обновляем Experiment (имя проекта) Пока что даже записи в этой таблицы не делается, поэтому это ложится на будущую доработку
        if content_parsed.get("experiment"):
            cursor.execute("UPDATE experiment SET name = ? WHERE experiment_id = (SELECT experiment_id FROM sample WHERE sample_id = ?)", 
                           (content_parsed["experiment"]["name"], sample_id))

        # 4. Обновляем первую запись spectrum_file
        if content_parsed.get("spectrum_file"):
            sf = content_parsed["spectrum_file"]
            # Здесь мы перечисляем все технические поля
            sql = """
                UPDATE spectrum_file SET 
                measurement_time = ?, 
                x_axis_name = ?, 
                x_axis_unit = ?, 
                y_axis_name = ?, 
                y_axis_unit = ?, 
                x_start = ?, 
                x_stop = ?, 
                resolution = ?, 
                acquisition_time = ?, 
                laser_wavelength = ?, 
                excitation_wavelength = ?, 
                instrument_metadata_json = ?, 
                instrument_config_json = ?
                WHERE spectrum_id = ?
            """
            cursor.execute(sql, (
                sf["measurement_time"], 
                sf["x_axis_name"], 
                sf["x_axis_unit"],
                sf["y_axis_name"], 
                sf["y_axis_unit"], 
                sf["x_start"], 
                sf["x_stop"],
                sf["resolution"], 
                sf["acquisition_time"], 
                sf["laser_wavelength"],
                sf["excitation_wavelength"], 
                sf["instrument_metadata_json"], 
                sf["instrument_config_json"], 
                spectrum_id
            ))

        # 5. Заполняем таблицу Raman
        if content_parsed.get("raman"):
            rd = content_parsed["raman"]
            columns = list(rd.keys()) + ["measurement_id"]
            placeholders = ", ".join(["?"] * len(columns))
            values = list(rd.values()) + [measurement_id]
            cursor.execute(f"INSERT INTO Raman ({', '.join(columns)}) VALUES ({placeholders})", values)

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def enrich_and_insert_uvvis_data(
    db_name: str,
    base_measurement_id: int,
    base_spectrum_id: int,
    filename_parsed: Dict[str, Any],
    content_list: List[Dict[str, Any]],
    file_path: str,
    experiment_id: int | None = None
):
    """
    Обрабатывает список спектров из UV-Vis файла.
    Первый спектр обновляет существующие ID, остальные создают новые записи,
    дублируя информацию из имени файла.
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    try:
        fn_meas = filename_parsed["measurement"]
        fn_spec = filename_parsed["spectrum_file"]

        for i, content in enumerate(content_list):
            
            # Извлекаем и сразу нормализуем основные блоки данных
            meas_content = {k: normalize_value(v) for k, v in content["measurement"].items()}
            sf = {k: normalize_value(v) for k, v in content["spectrum_file"].items()}
            uv = {k: normalize_value(v) for k, v in content["uvvis"].items()}
            sample_content = {k: normalize_value(v) for k, v in content["sample"].items()}
            
            if i == 0:
                 # --- ЭТАП 1: ОБНОВЛЯЕМ ОБЩИЕ ДАННЫЕ (только один раз для первого спектра) ---

                # Обновляем Measurement (ставим версию софта и оператора)
                cursor.execute(
                    "UPDATE measurement SET software_version = ?, operator = ? WHERE measurement_id = ?",
                    (meas_content["software_version"], 
                     meas_content["operator"], base_measurement_id)
                )

                # Обновляем ПЕРВУЮ запись spectrum_file (ту, что создана по имени файла)
                sql_sf = """
                    UPDATE spectrum_file SET 
                    measurement_time = ?, 
                    x_axis_name = ?, 
                    x_axis_unit = ?, 
                    y_axis_name = ?, 
                    y_axis_unit = ?, 
                    x_start = ?, 
                    x_stop = ?, 
                    resolution = ?, 
                    acquisition_time = ?, 
                    instrument_metadata_json = ?, 
                    instrument_config_json = ?
                    WHERE spectrum_id = ?
                """
                cursor.execute(sql_sf, (
                    sf["measurement_time"], 
                    sf["x_axis_name"], 
                    sf["x_axis_unit"],
                    sf["y_axis_name"], 
                    sf["y_axis_unit"], 
                    sf["x_start"], 
                    sf["x_stop"],
                    sf["resolution"], 
                    sf["acquisition_time"], 
                    sf["instrument_metadata_json"], 
                    sf["instrument_config_json"], 
                    base_spectrum_id
                ))

                # Вставляем запись в таблицу UVVis для этого первого спектра
                uv["measurement_id"] = base_measurement_id
                uv["uvvis_mode"] = normalize_value(fn_meas["mode"])
                cols = list(uv.keys())
                cursor.execute(
                    f"INSERT INTO UVVis ({', '.join(cols)}) VALUES ({', '.join(['?']*len(cols))})",
                    [normalize_value(uv[c]) for c in cols]
                )

            else:
                # --- ЭТАП 2: СОЗДАНИЕ НОВЫХ ЗАПИСЕЙ (ДОБАВЛЯЕМ НОВЫЕ СПЕКТРЫ К ТОМУ ЖЕ MEASUREMENT_ID) ---
                
                # Новая запись в spectrum_file
                cursor.execute(
                    """INSERT INTO spectrum_file (
                        measurement_id, 
                        spectrum_type, 
                        file_path, 
                        file_name, 
                        file_format,
                        measurement_time, 
                        x_axis_name, 
                        x_axis_unit, 
                        y_axis_name, 
                        y_axis_unit, 
                        x_start, 
                        x_stop, 
                        resolution, 
                        acquisition_time, 
                        instrument_metadata_json, 
                        instrument_config_json)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        base_measurement_id, # Тот же самый ID измерения!
                        fn_meas["mode"], 
                        file_path, 
                        fn_spec["file_name"], 
                        fn_spec["file_format"],
                        sf["measurement_time"], 
                        sf["x_axis_name"], 
                        sf["x_axis_unit"], 
                        sf["y_axis_name"], 
                        sf["y_axis_unit"], 
                        sf["x_start"], 
                        sf["x_stop"], 
                        sf["resolution"], 
                        sf["acquisition_time"], 
                        sf["instrument_metadata_json"], 
                        sf["instrument_config_json"]
                    )
                )

                # Новая запись в UVVis
                uv["measurement_id"] = base_measurement_id # Тот же самый ID измерения!
                uv["uvvis_mode"] = normalize_value(fn_meas["mode"])
                cols = list(uv.keys())
                cursor.execute(
                    f"INSERT INTO UVVis ({', '.join(cols)}) VALUES ({', '.join(['?']*len(cols))})",
                    [normalize_value(uv[c]) for c in cols]
                )

        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()