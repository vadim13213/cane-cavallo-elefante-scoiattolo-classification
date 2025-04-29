
# 🐕🐎🐘🐿 Классификация животных по изображениям

## 📝 Описание проекта

Этот проект представляет собой систему классификации изображений с использованием модели глубокого обучения, позволяющей отличать **собаку (cane)**, **лошадь (cavallo)**, **слона (elefante)** и **белку (scoiattolo)**. Веб-интерфейс реализован с помощью **Streamlit**, серверная часть — на **FastAPI**. Приложение позволяет загружать изображение, после чего оно отправляется на сервер, где выполняется предобработка и предсказание класса.

В качестве данных использован набор изображений с сайта [Kaggle](https://www.kaggle.com/datasets/alessiocorrado99/animals10). Для обучения моделей был выбран только подмножество классов: собака, лошадь, слон и белка.

### 🧠 Особенности:
- Поддержка загрузки изображений.
- Отображение предсказанного класса и вероятностей по всем категориям.
- Интерактивная визуализация с помощью Streamlit.

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

| Имя модели               | Точность | Полнота | Precision | F1-мера |
|--------------------------|----------|---------|-----------|---------|
| **simple_cnn(work_№3)**   |  0.2710  | 0.2710  | 0.3595    | 0.2866  |
| **residual_cnn(work_№4)** |  0.2534  | 0.2534  | 0.3283    | 0.2754  |
| **deep_cnn(work_№3)**     |  0.2441  | 0.2441  | 0.3336    | 0.2609  |

**Лучшая модель по F1-мере: simple_cnn(work_№3) aka best_model_for_api.keras**

<p align="center"> <img src="https://i.postimg.cc/0QPpM1kH/2025-04-29-192229.png" alt="confusion_matrix" width="1000"/> </p>

[Ссылка на блокнот с сравнением моделей](https://github.com/vadim13213/neural_networks/blob/main/Практическая_работа_№9_Сравнение_моделей_классификации_изображений_и_развертывание_API.ipynb)

---

## 📊 Визуализация результатов

Пример интерфейса **Streamlit**:

<p align="center"> <img src="YOUR_SCREENSHOT_LINK" alt="confusion_matrix" width="600"/> </p>

---

## 🚀 Развёртывание проекта локально

1. **Клонируйте репозиторий:**

```bash
git clone https://github.com/vadim13213/cane-cavallo-elefante-scoiattolo-classification.git
cd cane-cavallo-elefante-scoiattolo-classification
```

2. **Создайте виртуальное окружение и активируйте его:**

```bash
python -m venv .venv
source .venv/bin/activate     # Linux/macOS
.venv\Scriptsctivate        # Windows
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

- 🎯 **Streamlit-интерфейс**: [https://cane-cavallo-elefante-scoiattolo-q9vw.onrender.com/](https://cane-cavallo-elefante-scoiattolo-q9vw.onrender.com/)
- ⚡ **API сервер (FastAPI)**: [https://cane-cavallo-elefante-scoiattolo.onrender.com/](https://cane-cavallo-elefante-scoiattolo.onrender.com/)

---

## 🧪 Примеры использования API

### Пример POST-запроса с изображением

```bash
curl -X POST "https://cane-cavallo-elefante-scoiattolo.onrender.com/predict/"   -H "accept: application/json"   -H "Content-Type: multipart/form-data"   -F "file=@example.jpg"
```

### Ответ:

```json
{
  "predicted_class": "0",
  "probabilities": {
    "0": 0.95,
    "1": 0.02,
    "2": 0.01,
    "3": 0.02
  }
}
```

**Где:**
- "0" — собака (cane) 🐕
- "1" — лошадь (cavallo) 🐎
- "2" — слон (elefante) 🐘
- "3" — белка (scoiattolo) 🐿

---
