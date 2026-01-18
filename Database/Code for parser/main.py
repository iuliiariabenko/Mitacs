from create_db import create_database
from parser_filename import parse_filename
from db_insert import insert_from_parsed_filename, enrich_raman_data, enrich_and_insert_uvvis_data
from config import DB_NAME, DATA_DIR
from parser_raman import parse_raman_full
from parser_uvvis import parse_uvvis_full


import os


def main():
    # Гарантируем наличие БД
    create_database(DB_NAME)

    for file in os.listdir(DATA_DIR):
        
        full_path = os.path.join(DATA_DIR, file)

        try:
            name_data = parse_filename(file)
            s_id, m_id, spec_id = insert_from_parsed_filename(
                name_data, 
                file_path=DATA_DIR, 
                db_name=DB_NAME, 
                experiment_id=None
            )

            # Если это Раман, парсим контент и обогащаем данные
            if name_data["measurement"]["technique"].lower() == "raman":
                content_data = parse_raman_full(full_path)
                enrich_raman_data(DB_NAME, s_id, m_id, spec_id, content_data)
                
            if name_data["measurement"]["technique"].lower() == "uvvis":
                content_list = parse_uvvis_full(full_path)
                # Вызываем нашу новую функцию
                enrich_and_insert_uvvis_data(DB_NAME, m_id, spec_id, name_data, content_list, DATA_DIR)
                
            print(f"Обработан: {file}")

        except Exception as e:
            print(f"Ошибка в файле {file}: {e}")


if __name__ == "__main__":
    main()
