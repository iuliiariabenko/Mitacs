PRAGMA foreign_keys = ON;

-- ============================================================
-- 1. EXPERIMENT
-- ============================================================
-- Кампания / тест-план
-- ============================================================

CREATE TABLE experiment (
    experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT,
    operator TEXT,
    organization TEXT,
    start_date DATETIME NOT NULL,
    end_date DATETIME,
    notes TEXT
);

CREATE INDEX idx_experiment_start
    ON experiment(start_date);

-- ============================================================
-- 2. SAMPLE
-- ============================================================
-- Физический образец топлива / масла
-- ============================================================

CREATE TABLE sample (
    sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,

    sample_name TEXT NOT NULL,
    sample_type TEXT NOT NULL,          -- fuel, oil, reference
    material TEXT,                      -- Jet-A, Diesel, Oil-XYZ
    batch_id TEXT,
    contamination_note TEXT,

    FOREIGN KEY (experiment_id)
        REFERENCES experiment(experiment_id)
        ON DELETE CASCADE
);

CREATE INDEX idx_sample_experiment
    ON sample(experiment_id);

-- ============================================================
-- 3. MEASUREMENT
-- ============================================================
-- Один акт измерения образца
-- ============================================================

CREATE TABLE measurement (
    measurement_id INTEGER PRIMARY KEY AUTOINCREMENT,
    sample_id INTEGER NOT NULL,

    technique TEXT NOT NULL,             -- UV, Raman, FTIR, Fluorescence
    instrument TEXT,                     -- model / serial
    operator TEXT,

    measurement_time DATETIME NOT NULL,
    temperature REAL,                    -- °C
    pressure REAL,                       -- optional
    notes TEXT,

    FOREIGN KEY (sample_id)
        REFERENCES sample(sample_id)
        ON DELETE CASCADE
);

CREATE INDEX idx_measurement_sample
    ON measurement(sample_id);

CREATE INDEX idx_measurement_technique
    ON measurement(technique);

-- ============================================================
-- 4. SPECTRUM FILE
-- ============================================================
-- CSV-файл со спектром + метаданные
-- ============================================================

CREATE TABLE spectrum_file (
    spectrum_id INTEGER PRIMARY KEY AUTOINCREMENT,
    measurement_id INTEGER NOT NULL,

    spectrum_type TEXT NOT NULL,          -- UV, Raman, FTIR, Fluorescence
    file_path TEXT NOT NULL,              -- ./data/exp01/sampleA_uv.csv
    file_format TEXT DEFAULT 'CSV',

    x_axis_name TEXT,                     -- wavelength, wavenumber
    x_axis_unit TEXT,                     -- nm, cm-1
    y_axis_name TEXT,                     -- absorbance, intensity
    y_axis_unit TEXT,

    x_start REAL,
    x_stop REAL,
    resolution REAL,

    acquisition_time REAL,                -- seconds
    laser_wavelength REAL,                -- Raman
    excitation_wavelength REAL,           -- UV/Fluo

    FOREIGN KEY (measurement_id)
        REFERENCES measurement(measurement_id)
        ON DELETE CASCADE
);

CREATE INDEX idx_spectrum_measurement
    ON spectrum_file(measurement_id);

CREATE INDEX idx_spectrum_type
    ON spectrum_file(spectrum_type);

CREATE TABLE test_sequence (
    sequence_id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,

    name TEXT NOT NULL,
    description TEXT,
    version TEXT,
    created_by TEXT,
    created_at DATETIME NOT NULL,

    FOREIGN KEY (experiment_id)
        REFERENCES experiment(experiment_id)
        ON DELETE CASCADE
);

CREATE TABLE analysis_run (
    analysis_id INTEGER PRIMARY KEY AUTOINCREMENT,
    measurement_id INTEGER NOT NULL,

    algorithm_name TEXT NOT NULL,      -- BaselineCorr, PCA, ASTM_DXXX
    algorithm_version TEXT,
    parameters_json TEXT,
    executed_at DATETIME NOT NULL,

    FOREIGN KEY (measurement_id)
        REFERENCES measurement(measurement_id)
        ON DELETE CASCADE
);

CREATE TABLE qc_formula (
    formula_id INTEGER PRIMARY KEY AUTOINCREMENT,

    name TEXT NOT NULL,
    description TEXT,
    expression TEXT NOT NULL,          -- e.g. "OSNR > 35"
    units TEXT,

    standard TEXT,                     -- ASTM, ISO, internal
    version TEXT
);


CREATE TABLE analysis_metric (
    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER NOT NULL,

    metric_name TEXT NOT NULL,         -- OSNR, PeakShift, TAN
    metric_value REAL,
    units TEXT,

    FOREIGN KEY (analysis_id)
        REFERENCES analysis_run(analysis_id)
        ON DELETE CASCADE
);


CREATE TABLE qc_result (
    qc_id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER NOT NULL,
    formula_id INTEGER NOT NULL,

    decision TEXT NOT NULL,             -- pass, fail, warning
    threshold TEXT,
    comment TEXT,
    evaluated_at DATETIME NOT NULL,

    FOREIGN KEY (analysis_id)
        REFERENCES analysis_run(analysis_id)
        ON DELETE CASCADE,
    FOREIGN KEY (formula_id)
        REFERENCES qc_formula(formula_id)
);

CREATE TABLE access_level (
    access_id INTEGER PRIMARY KEY AUTOINCREMENT,
    role_name TEXT NOT NULL,            -- admin, chemist, viewer
    description TEXT
);

CREATE TABLE system_user (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    full_name TEXT,
    access_id INTEGER NOT NULL,

    FOREIGN KEY (access_id)
        REFERENCES access_level(access_id)
);

CREATE TABLE system_message (
    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,

    level TEXT NOT NULL,                -- INFO, WARNING, ERROR
    message TEXT NOT NULL,
    timestamp DATETIME NOT NULL,

    FOREIGN KEY (user_id)
        REFERENCES system_user(user_id)
);


CREATE TABLE sample_preparation_sequence (
    prep_sequence_id INTEGER PRIMARY KEY AUTOINCREMENT,

    name TEXT NOT NULL,
    description TEXT,
    protocol TEXT,
    created_at DATETIME NOT NULL
);


CREATE TABLE chemistry_lab_sequence (
    lab_sequence_id INTEGER PRIMARY KEY AUTOINCREMENT,

    name TEXT NOT NULL,
    description TEXT,
    procedure TEXT,
    safety_notes TEXT,
    created_at DATETIME NOT NULL
);

CREATE TABLE sample_process_link (
    sample_id INTEGER NOT NULL,
    prep_sequence_id INTEGER,
    lab_sequence_id INTEGER,

    PRIMARY KEY (sample_id),

    FOREIGN KEY (sample_id)
        REFERENCES sample(sample_id)
        ON DELETE CASCADE,
    FOREIGN KEY (prep_sequence_id)
        REFERENCES sample_preparation_sequence(prep_sequence_id),
    FOREIGN KEY (lab_sequence_id)
        REFERENCES chemistry_lab_sequence(lab_sequence_id)
);
