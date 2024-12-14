# src/app.py

import streamlit as st
import cv2
import numpy as np
import pickle
from PIL import Image, ImageDraw, ImageFont
import os

# Установка конфигурации страницы должна быть первой командой Streamlit
st.set_page_config(page_title="Распознавание лиц Energo University", layout="wide")

# Загрузка модели и меток
@st.cache_resource
def load_model(model_path='encodings.yml', labels_path='labels.pkl'):
    if not os.path.exists(model_path):
        st.error(f"Файл модели '{model_path}' не найден.")
        return None, {}
    if not os.path.exists(labels_path):
        st.error(f"Файл меток '{labels_path}' не найден.")
        return None, {}

    # Загрузка модели
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)

    # Загрузка словаря меток
    with open(labels_path, 'rb') as f:
        label_dict = pickle.load(f)

    # Инвертирование словаря для получения имен по меткам
    label_dict_inv = {v: k for k, v in label_dict.items()}

    return recognizer, label_dict_inv

recognizer, label_dict = load_model()

def main():
    if recognizer is None or not label_dict:
        st.stop()

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
                image = Image.open(captured_image).convert('RGB')
                open_cv_image = np.array(image)
                open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

                # Детектирование лиц
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

                if len(faces) == 0:
                    st.warning("Лица не обнаружены на изображении.")
                else:
                    # Создание объекта PIL для рисования
                    pil_image = Image.fromarray(cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB))
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

                    # Параметры для рисования
                    font_color = (255, 255, 255)  # Белый цвет текста

                    for (x, y, w, h) in faces:
                        face_roi = gray[y:y+h, x:x+w]
                        face_roi_resized = cv2.resize(face_roi, (200, 200))  # Приведение к размеру, ожидаемому моделью

                        # Распознавание лица
                        label, confidence = recognizer.predict(face_roi_resized)
                        if label in label_dict:
                            name = label_dict[label]
                            confidence_score = max(0, min(100, int(100 - confidence)))
                            label_text = f"Сотрудник Energo University: {name} ({confidence_score}%)"  # Изменено здесь
                            color = (0, 255, 0)  # Зеленый для известных лиц
                        else:
                            label_text = "Неизвестно"
                            confidence_score = 0
                            color = (255, 0, 0)  # Красный для неизвестных лиц

                        # Рисование прямоугольника вокруг лица
                        draw.rectangle(((x, y), (x + w, y + h)), outline=color, width=2)

                        # Измерение размера текста
                        try:
                            bbox = font.getbbox(label_text)
                            text_width = bbox[2] - bbox[0]
                            text_height = bbox[3] - bbox[1]
                        except AttributeError:
                            st.error("Метод 'getbbox' не поддерживается используемым шрифтом.")
                            bbox = (0, 0, 0, 0)
                            text_width = 0
                            text_height = 0

                        # Рисование фона для текста
                        draw.rectangle(((x, y + h - text_height - 10), (x + text_width + 12, y + h)), fill=color)

                        # Добавление текста под прямоугольником
                        draw.text((x + 6, y + h - text_height - 5), label_text, fill=font_color, font=font)

                    # Удаление объекта рисования
                    del draw

                    # Отображение обработанного изображения
                    st.image(pil_image, caption="Результат распознавания", use_container_width=True)

            except Exception as e:
                st.error(f"Произошла ошибка при обработке изображения: {e}")

if __name__ == "__main__":
    main()
