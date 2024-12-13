# src/encode_faces.py

import face_recognition
import os
import pickle


def encode_faces(dataset_dir='dataset', encodings_file='encodings.pkl'):
    # Списки для хранения кодировок и имен
    known_face_encodings = []
    known_face_names = []

    # Проход по всем файлам в датасете
    for image_name in os.listdir(dataset_dir):
        image_path = os.path.join(dataset_dir, image_name)

        # Проверка, что это файл изображения
        if not os.path.isfile(image_path) or not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # Извлечение имени сотрудника из названия файла
        try:
            # Предполагается, что имя сотрудника находится перед первым символом '_'
            name = image_name.split('_')[0]
            name = name.replace('-', ' ').replace('.', ' ').title()  # Форматирование имени
        except IndexError:
            print(f"Не удалось извлечь имя из названия файла {image_name}. Пропуск.")
            continue

        try:
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if len(encodings) == 0:
                print(f"Лицо не обнаружено на изображении {image_path}. Пропуск.")
                continue
            encoding = encodings[0]
            known_face_encodings.append(encoding)
            known_face_names.append(name)
            print(f"Обработано изображение {image_path} для сотрудника {name}.")
        except Exception as e:
            print(f"Ошибка при обработке {image_path}: {e}")

    # Сохранение кодировок и имен в файл
    with open(encodings_file, 'wb') as f:
        pickle.dump({'encodings': known_face_encodings, 'names': known_face_names}, f)

    print(f"Кодировки лиц успешно сохранены в {encodings_file}.")


if __name__ == "__main__":
    encode_faces()
