import sqlite3

def create_database(db_name="laboratory_data.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Включаем поддержку внешних ключей
    cursor.execute("PRAGMA foreign_keys = ON;")

    # 1. Слой доступа и пользователей
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS access_level (
        access_id INTEGER PRIMARY KEY AUTOINCREMENT,
        role_name TEXT NOT NULL,
        description TEXT
    )''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS system_user (
        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL UNIQUE,
        full_name TEXT,
        access_id INTEGER,
        FOREIGN KEY (access_id) REFERENCES access_level(access_id) ON DELETE CASCADE
    )''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS system_message (
        message_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        level TEXT,
        message TEXT,
        timestamp TEXT,
        FOREIGN KEY (user_id) REFERENCES system_user(user_id) ON DELETE CASCADE
    )''')

    # 2. Основной контекст эксперимента
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS experiment (
        experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        description TEXT,
        operator TEXT,
        organization TEXT,
        start_date TEXT,
        end_date TEXT,
        project_code TEXT, -- Из PDF Слой B
        location TEXT,     -- Из PDF Слой B
        notes TEXT
    )''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS test_sequence (
        sequence_id INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment_id INTEGER,
        name TEXT,
        description TEXT,
        version TEXT,
        created_by TEXT,
        created_at TEXT,
        FOREIGN KEY (experiment_id) REFERENCES experiment(experiment_id) ON DELETE CASCADE
    )''')

    # 3. Подготовка образцов
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sample_preparation_sequence (
        prep_sequence_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        description TEXT,
        protocol TEXT,
        created_at TEXT
    )''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS chemistry_lab_sequence (
        lab_sequence_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        description TEXT,
        procedure TEXT,
        safety_notes TEXT,
        created_at TEXT
    )''')

    # 4. Образцы (Sample)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sample (
        sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment_id INTEGER,
        sample_name TEXT,
        sample_type TEXT,
        material TEXT,
        batch_id TEXT,
        contamination_note TEXT,
        concentration_value REAL,
        concentration_unit TEXT,
        replicate_id TEXT,
        sample_semantics_json TEXT, -- JSON поле для matrix_code, analyte_code и т.д.
        FOREIGN KEY (experiment_id) REFERENCES experiment(experiment_id) ON DELETE CASCADE
    )''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sample_process_link (
        sample_id INTEGER,
        prep_sequence_id INTEGER,
        lab_sequence_id INTEGER,
        PRIMARY KEY (sample_id, prep_sequence_id, lab_sequence_id),
        FOREIGN KEY (sample_id) REFERENCES sample(sample_id) ON DELETE CASCADE,
        FOREIGN KEY (prep_sequence_id) REFERENCES sample_preparation_sequence(prep_sequence_id) ON DELETE CASCADE,
        FOREIGN KEY (lab_sequence_id) REFERENCES chemistry_lab_sequence(lab_sequence_id) ON DELETE CASCADE
    )''')

    # 5. Измерения (Measurement)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS measurement (
        measurement_id INTEGER PRIMARY KEY AUTOINCREMENT,
        sample_id INTEGER,
        technique TEXT,
        instrument TEXT,
        operator TEXT,
        temperature REAL,
        pressure REAL,
        software_name TEXT,
        software_version TEXT,
        mode TEXT,
        notes TEXT,
        FOREIGN KEY (sample_id) REFERENCES sample(sample_id) ON DELETE CASCADE
    )''')

    # 6. Файлы спектров
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS spectrum_file (
        spectrum_id INTEGER PRIMARY KEY AUTOINCREMENT,
        measurement_id INTEGER,
        spectrum_type TEXT,
        file_path TEXT,
        file_name TEXT,
        file_format TEXT,
        measurement_time TEXT,
        instrument_metadata_json TEXT,
        instrument_config_json TEXT,
        x_axis_name TEXT,
        x_axis_unit TEXT,
        y_axis_name TEXT,
        y_axis_unit TEXT,
        x_start REAL,
        x_stop REAL,
        resolution REAL,
        acquisition_time REAL, -- В секундах
        laser_wavelength REAL,
        excitation_wavelength REAL,
        FOREIGN KEY (measurement_id) REFERENCES measurement(measurement_id) ON DELETE CASCADE
    )''')

    # 7. Специфические методы (Raman, UVVis, FTIR)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Raman (
        raman_id INTEGER PRIMARY KEY AUTOINCREMENT,
        measurement_id INTEGER,
        laser_nm REAL,
        grating_gmm INTEGER,
        objective TEXT,
        slit_um REAL,
        hole_um REAL,
        filter_percent REAL,
        acq_time_s REAL,
        accumulations INTEGER,
        range_cm1_min REAL,
        range_cm1_max REAL,
        spectral_res_cm1 REAL,
        stage_x_um REAL,
        stage_y_um REAL,
        stage_z_um REAL,
        corrections_flags TEXT, -- Список флагов в формате текста или JSON
        FOREIGN KEY (measurement_id) REFERENCES measurement(measurement_id) ON DELETE CASCADE
    )''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS UVVis (
        uvvis_id INTEGER PRIMARY KEY AUTOINCREMENT,
        measurement_id INTEGER,
        uvvis_mode TEXT,
        x_unit TEXT,
        y_unit TEXT,
        wl_start_nm REAL,
        wl_stop_nm REAL,
        sbw_nm REAL,
        ave_time_s REAL,
        data_interval_nm REAL,
        scan_rate_nm_min REAL,
        beam_mode TEXT,
        baseline_correction TEXT,
        source_changeover_nm REAL,
        method_log TEXT,
        FOREIGN KEY (measurement_id) REFERENCES measurement(measurement_id) ON DELETE CASCADE
    )''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS FTIR (
        ftir_id INTEGER PRIMARY KEY AUTOINCREMENT,
        measurement_id INTEGER,
        ftir_accessory TEXT,
        x_unit TEXT,
        y_unit TEXT,
        background_id TEXT,
        method_name TEXT,
        energy_value REAL,
        FOREIGN KEY (measurement_id) REFERENCES measurement(measurement_id) ON DELETE CASCADE
    )''')

    # 8. Анализ и Контроль Качества (QC)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS analysis_run (
        analysis_id INTEGER PRIMARY KEY AUTOINCREMENT,
        measurement_id INTEGER,
        algorithm_name TEXT,
        algorithm_version TEXT,
        pipeline_name TEXT,
        parameters_json TEXT,
        feature_table_ref TEXT,
        executed_at TEXT,
        FOREIGN KEY (measurement_id) REFERENCES measurement(measurement_id) ON DELETE CASCADE
    )''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS analysis_metric (
        metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
        analysis_id INTEGER,
        metric_name TEXT,
        metric_value REAL,
        units TEXT,
        FOREIGN KEY (analysis_id) REFERENCES analysis_run(analysis_id) ON DELETE CASCADE
    )''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS qc_formula (
        formula_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        description TEXT,
        expression TEXT,
        units TEXT,
        standard TEXT,
        version TEXT
    )''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS qc_result (
        qc_id INTEGER PRIMARY KEY AUTOINCREMENT,
        analysis_id INTEGER,
        formula_id INTEGER,
        decision TEXT, -- pass/warn/fail
        threshold REAL,
        comment TEXT,
        evaluated_at TEXT,
        FOREIGN KEY (analysis_id) REFERENCES analysis_run(analysis_id) ON DELETE CASCADE,
        FOREIGN KEY (formula_id) REFERENCES qc_formula(formula_id) ON DELETE CASCADE
    )''')

    conn.commit()
    conn.close()
    # print(f"База данных '{db_name}' успешно создана.")

if __name__ == "__main__":
    create_database()