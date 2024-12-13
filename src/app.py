# src/app.py

import streamlit as st
import face_recognition
import cv2
import numpy as np
import pickle
from PIL import Image, ImageDraw, ImageFont
import os

# Установка конфигурации страницы должна быть первой командой Streamlit
st.set_page_config(page_title="Распознавание лиц Energo University", layout="wide")

# Загрузка кодировок лиц с использованием нового декоратора кэша
@st.cache_data
def load_encodings(encodings_path='encodings.pkl'):
    try:
        with open(encodings_path, 'rb') as f:
            data = pickle.load(f)
        return data['encodings'], data['names']
    except FileNotFoundError:
        st.error(f"Файл кодировок '{encodings_path}' не найден. Убедитесь, что он существует и путь указан правильно.")
        return [], []
    except Exception as e:
        st.error(f"Произошла ошибка при загрузке кодировок: {e}")
        return [], []

known_face_encodings, known_face_names = load_encodings()

def main():
    # Загрузка логотипа
    try:
        logo = Image.open("assets/logo_daukeev.png")
    except FileNotFoundError:
        st.warning("Логотип не найден. Убедитесь, что файл 'logo_daukeev.png' находится в папке 'assets/'.")
        logo = None
    except Exception as e:
        st.error(f"Произошла ошибка при загрузке логотипа: {e}")
        logo = None

    # Создание боковой панели с логотипом и меню
    with st.sidebar:
        if logo:
            try:
                # Ограничение размера логотипа до ширины 200 пикселей
                st.image(logo, width=200, use_container_width=False)
            except Exception as e:
                st.error(f"Произошла ошибка при отображении логотипа: {e}")
        menu = st.radio("Навигация", ["Главная", "Распознавание лица"])

    if menu == "Главная":
        st.title("Проект распознавания лиц и блюд")
        st.write("""
            Добро пожаловать в проект распознавания лиц и блюд, разработанный в **Energo University**.
            Это приложение использует технологии машинного обучения для распознавания лиц сотрудников.
            Также планируется добавить функциональность распознавания блюд.
        """)
        try:
            project_image = Image.open("assets/logo_labtop.jpg")
            # Ограничение размера изображения до ширины 600 пикселей для среднего размера
            st.image(project_image, width=600, use_container_width=False)
        except FileNotFoundError:
            st.warning("Изображение проекта не найдено. Убедитесь, что файл 'logo_labtop.jpg' находится в папке 'assets/'.")
        except Exception as e:
            st.error(f"Произошла ошибка при загрузке изображения проекта: {e}")

    elif menu == "Распознавание лица":
        st.title("Распознавание лиц")
        st.write("Используйте кнопку ниже, чтобы сделать снимок с веб-камеры и начать распознавание лиц.")

        # Захват изображения с веб-камеры
        captured_image = st.camera_input("Сделайте снимок")

        if captured_image:
            try:
                # Отображение исходного изображения
                st.image(captured_image, caption="Исходное изображение", use_container_width=True)

                # Конвертация изображения в формат, пригодный для обработки
                image = face_recognition.load_image_file(captured_image)
                face_locations = face_recognition.face_locations(image)
                face_encodings = face_recognition.face_encodings(image, face_locations)

                if not face_locations:
                    st.warning("Лица не обнаружены на изображении.")
                else:
                    # Создание объекта PIL для рисования
                    pil_image = Image.fromarray(image)
                    draw = ImageDraw.Draw(pil_image)

                    # Загрузка кастомного шрифта
                    font_path = os.path.join("fonts", "DejaVuSans.ttf")
                    try:
                        font = ImageFont.truetype(font_path, size=20)
                    except IOError:
                        st.error(f"Не удалось загрузить шрифт по пути: {font_path}")
                        st.warning("Используется шрифт по умолчанию, который может не поддерживать кириллицу.")
                        font = ImageFont.load_default()
                    except Exception as e:
                        st.error(f"Произошла ошибка при загрузке шрифта: {e}")
                        font = ImageFont.load_default()

                    # Установка максимальной дистанции для определения точности
                    max_distance = 0.6  # Соответствует стандартной толерантности face_recognition

                    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                        # Сравнение с известными лицами
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        name = "Неизвестно"
                        confidence = 0  # Изначально 0%

                        if matches[best_match_index]:
                            name = known_face_names[best_match_index]
                            # Вычисление процента точности
                            distance = face_distances[best_match_index]
                            confidence = max(0, min(100, int((1 - distance / max_distance) * 100)))
                            label = f"Сотрудник Energo University: {name} ({confidence}%)"
                            color = (0, 255, 0)  # Зеленый
                        else:
                            label = f"Неизвестно ({0}%)"
                            color = (255, 0, 0)  # Красный

                        # Рисование прямоугольника вокруг лица
                        draw.rectangle(((left, top), (right, bottom)), outline=color, width=2)

                        # Измерение размера текста
                        try:
                            bbox = font.getbbox(label)
                            text_width = bbox[2] - bbox[0]
                            text_height = bbox[3] - bbox[1]
                        except AttributeError:
                            st.error("Метод 'getbbox' не поддерживается используемым шрифтом.")
                            bbox = (0, 0, 0, 0)
                            text_width = 0
                            text_height = 0

                        # Рисование фона для текста
                        draw.rectangle(((left, bottom - text_height - 10), (left + text_width + 12, bottom)), fill=color, outline=color)

                        # Добавление текста под прямоугольником
                        draw.text((left + 6, bottom - text_height - 5), label, fill=(255, 255, 255, 255), font=font)

                    # Удаление объекта рисования
                    del draw

                    # Отображение обработанного изображения
                    st.image(pil_image, caption="Результат распознавания", use_container_width=True)

            except Exception as e:
                st.error(f"Произошла ошибка при обработке изображения: {e}")

if __name__ == "__main__":
    main()
