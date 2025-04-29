# 🐔🐄🐑🐎 Классификация животных по изображениям

## 📝 Описание проекта

Этот проект представляет собой систему классификации изображений с использованием модели глубокого обучения, позволяющей отличать **курицу**, **корову**, **овцу** и **лошадь**. Веб-интерфейс реализован с помощью **Streamlit**, серверная часть — на **FastAPI**. Приложение позволяет загружать изображение или рисовать его прямо в браузере, после чего оно отправляется на сервер, где выполняется предобработка и предсказание класса.

В качестве данных использован набор изображений, содержащий фотографии указанных животных ([Исходный датасет на Kaggle](https://www.kaggle.com/datasets/alessiocorrado99/animals10)). В представленном наборе данных содержится 10 классов изображений животных с разным разрешением и разным колличеством изображений на класс. Для удобства обучения моделей и для обеспечения равного количества данных на класс датасет был преобразован к нужному виду (выбрано 4 класса изображений с животными, уравновешенно количество данных — 1800 на класс).

### 🧠 Особенности:
- Поддержка загрузки и рисования изображений.
- Отображение предсказанного класса и вероятностей по всем категориям.
- Интерактивная визуализация.

---

## 📦 Используемые технологии

- Python 🐍
- TensorFlow / Keras 🤖
- FastAPI ⚡
- Streamlit 🌐
- OpenCV 📷
- NumPy 📊
- Pillow 🖼️

---

## 🔍 Сравниваемые модели

| Имя модели               | Точность | Полнота |      Precision      | F1-мера|
|--------------------------|----------|---------|---------------------|--------|
| **deep_cnn(work_№3)**    |  0.9181  | 0.9181  | 0.9182              | 0.9180 |
| **residual_cnn(work_№4)**|  0.8854  | 0.8854  | 0.8860              | 0.8852 |
| **simple_cnn(work_№3)**  |  0.8729  | 0.8729  | 0.8743              | 0.8728 |
| **simple_dnn(work_№2)**  |  0.5458  | 0.5458  | 0.5409              | 0.5378 |

**Лучшая модель по F1-мере: deep_cnn(work_№3) aka best_model_resnet_like.keras**

<p align="center"> <img src="https://i.imgur.com/VvGk8E3.png" alt="confusion_matrix" width="1000"/> </p>

[Ссылка на блокнот с сравнением моделей](https://github.com/Poziloi/neural_networks/blob/main/%D0%9F%D1%80%D0%B0%D0%BA%D1%82%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%B0%D1%8F_%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D0%B0_%E2%84%969_%D0%A1%D1%80%D0%B0%D0%B2%D0%BD%D0%B5%D0%BD%D0%B8%D0%B5_%D0%BC%D0%BE%D0%B4%D0%B5%D0%BB%D0%B5%D0%B9_%D0%BA%D0%BB%D0%B0%D1%81%D1%81%D0%B8%D1%84%D0%B8%D0%BA%D0%B0%D1%86%D0%B8%D0%B8_%D0%B8%D0%B7%D0%BE%D0%B1%D1%80%D0%B0%D0%B6%D0%B5%D0%BD%D0%B8%D0%B9_%D0%B8_%D1%80%D0%B0%D0%B7%D0%B2%D0%B5%D1%80%D1%82%D1%8B%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5_API_%D0%A1%D0%BC%D1%8B%D1%81%D0%BB%D0%BE%D0%B2_%D0%90%D0%BB%D0%B5%D0%BA%D1%81%D0%B5%D0%B9.ipynb)

---

## 📊 Визуализация результатов

Пример интерфейса **Streamlit**:

<p align="center"> <img src="https://i.imgur.com/fkbFT5t.png" alt="confusion_matrix" width="600"/> </p>

---

## 🚀 Развёртывание проекта локально

1. **Клонируйте репозиторий:**

```bash
git clone https://github.com/Poziloi/chicken_cow_horse_sheep-classification.git
cd chicken_cow_horse_sheep-classification
```

2. **Создайте виртуальное окружение и активируйте его:**

```bash
python -m venv .venv
source .venv/bin/activate     # Linux/macOS
.venv\Scripts\activate      # Windows
```

3. **Установите зависимости:**

```bash
pip install -r requirements.txt
```

4. **Запустите FastAPI-сервер:**

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

5. **Запустите Streamlit-приложение:**

```bash
streamlit run app.py
```

6. **Откройте приложение в браузере:**
[http://localhost:8501](http://localhost:8501)
(порт 8501 установлен по умолчанию)

---

## 🔗 Развёрнутые приложения

- 🎯 **Streamlit-интерфейс**: [https://chicken-cow-horse-sheep-classification.streamlit.app/](https://chicken-cow-horse-sheep-classification.streamlit.app/)
- ⚡ **API сервер (FastAPI)**: [https://chicken-cow-horse-sheep-classification.onrender.com/docs](https://chicken-cow-horse-sheep-classification.onrender.com/docs)

---

## 🧪 Примеры использования API

### Пример POST-запроса с изображением

```bash
curl -X POST "https://chicken-api.onrender.com/predict/" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@example.jpg"
```

### Ответ:

```json
{
  "predicted_class": "3",
  "probabilities": {
    "0": 1.52e-8,
    "1": 0.207,
    "2": 0.000045,
    "3": 0.793
  }
}
```

**Где:**
- `"0"` — курица 🐔  
- `"1"` — корова 🐄  
- `"2"` — овца 🐑  
- `"3"` — лошадь 🐎  

---
