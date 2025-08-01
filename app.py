from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import pickle
import numpy as np
import os
from catboost import CatBoostClassifier

app = Flask(__name__)

# Конфигурация
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MODEL_TYPE'] = 'CatBoost'  # Тип модели: 'CatBoost' или 'NeuralNetwork'

# Структура формы на русском языке
FORM_FIELDS = [
    {'name': 'gender', 'label': 'Пол', 'type': 'select', 'options': ['Мужской', 'Женский']},
    {'name': 'SeniorCitizen', 'label': 'Пенсионер', 'type': 'select', 'options': ['Нет', 'Да']},
    {'name': 'Partner', 'label': 'Есть партнер', 'type': 'select', 'options': ['Да', 'Нет']},
    {'name': 'Dependents', 'label': 'Есть иждивенцы', 'type': 'select', 'options': ['Да', 'Нет']},
    {'name': 'tenure', 'label': 'Срок обслуживания (месяцы)', 'type': 'number', 'min': '0'},
    {'name': 'PhoneService', 'label': 'Телефонная служба', 'type': 'select', 'options': ['Да', 'Нет']},
    {'name': 'MultipleLines', 'label': 'Несколько линий', 'type': 'select',
     'options': ['Да', 'Нет', 'Нет телефонной службы']},
    {'name': 'InternetService', 'label': 'Интернет-сервис', 'type': 'select',
     'options': ['DSL', 'Оптоволокно', 'Нет']},
    {'name': 'OnlineSecurity', 'label': 'Онлайн-безопасность', 'type': 'select',
     'options': ['Да', 'Нет', 'Нет интернет-сервиса']},
    {'name': 'OnlineBackup', 'label': 'Онлайн-резервное копирование', 'type': 'select',
     'options': ['Да', 'Нет', 'Нет интернет-сервиса']},
    {'name': 'DeviceProtection', 'label': 'Защита устройства', 'type': 'select',
     'options': ['Да', 'Нет', 'Нет интернет-сервиса']},
    {'name': 'TechSupport', 'label': 'Техническая поддержка', 'type': 'select',
     'options': ['Да', 'Нет', 'Нет интернет-сервиса']},
    {'name': 'StreamingTV', 'label': 'Трансляция ТВ', 'type': 'select',
     'options': ['Да', 'Нет', 'Нет интернет-сервиса']},
    {'name': 'StreamingMovies', 'label': 'Трансляция фильмов', 'type': 'select',
     'options': ['Да', 'Нет', 'Нет интернет-сервиса']},
    {'name': 'Contract', 'label': 'Тип контракта', 'type': 'select',
     'options': ['Помесячный', 'Годовой', 'Двухлетний']},
    {'name': 'PaperlessBilling', 'label': 'Безбумажный биллинг', 'type': 'select', 'options': ['Да', 'Нет']},
    {'name': 'PaymentMethod', 'label': 'Способ оплаты', 'type': 'select',
     'options': ['Электронный чек', 'Почтовый чек', 'Банковский перевод', 'Кредитная карта']},
    {'name': 'MonthlyCharges', 'label': 'Ежемесячные платежи ($)', 'type': 'number', 'step': '0.01', 'min': '0'},
    {'name': 'TotalCharges', 'label': 'Общие платежи ($)', 'type': 'number', 'step': '0.01', 'min': '0'}
]


def load_ml_model():
    try:
        with open('ml_model/model.pkl', 'rb') as f:
            model = pickle.load(f)
            # Определяем имя модели на основе типа
            model_name = 'CatBoost Classifier' if app.config['MODEL_TYPE'] == 'CatBoost' else 'Neural Network'
            return model, model_name
    except Exception as e:
        print(f"ML Model Error: {e}")
        return None, None


def preprocess_input(form_data):
    """Преобразование введенных данных для модели"""
    # Создаем DataFrame с правильным порядком признаков
    data = {
        'gender': form_data['gender'],
        'SeniorCitizen': form_data['SeniorCitizen'],
        'Partner': form_data['Partner'],
        'Dependents': form_data['Dependents'],
        'tenure': float(form_data['tenure']),
        'PhoneService': form_data['PhoneService'],
        'MultipleLines': form_data['MultipleLines'],
        'InternetService': form_data['InternetService'],
        'OnlineSecurity': form_data['OnlineSecurity'],
        'OnlineBackup': form_data['OnlineBackup'],
        'DeviceProtection': form_data['DeviceProtection'],
        'TechSupport': form_data['TechSupport'],
        'StreamingTV': form_data['StreamingTV'],
        'StreamingMovies': form_data['StreamingMovies'],
        'Contract': form_data['Contract'],
        'PaperlessBilling': form_data['PaperlessBilling'],
        'PaymentMethod': form_data['PaymentMethod'],
        'MonthlyCharges': float(form_data['MonthlyCharges']),
        'TotalCharges': float(form_data['TotalCharges'])
    }

    # Преобразуем в DataFrame
    df = pd.DataFrame([data])

    # Маппинг русских значений на английские аналоги (как в обучающих данных)
    value_mapping = {
        'gender': {'Мужской': 'Male', 'Женский': 'Female'},
        'SeniorCitizen': {'Да': 'Yes', 'Нет': 'No'},
        'Partner': {'Да': 'Yes', 'Нет': 'No'},
        'Dependents': {'Да': 'Yes', 'Нет': 'No'},
        'PhoneService': {'Да': 'Yes', 'Нет': 'No'},
        'MultipleLines': {'Да': 'Yes', 'Нет': 'No', 'Нет телефонной службы': 'No phone service'},
        'InternetService': {'DSL': 'DSL', 'Оптоволокно': 'Fiber optic', 'Нет': 'No'},
        'OnlineSecurity': {'Да': 'Yes', 'Нет': 'No', 'Нет интернет-сервиса': 'No internet service'},
        'OnlineBackup': {'Да': 'Yes', 'Нет': 'No', 'Нет интернет-сервиса': 'No internet service'},
        'DeviceProtection': {'Да': 'Yes', 'Нет': 'No', 'Нет интернет-сервиса': 'No internet service'},
        'TechSupport': {'Да': 'Yes', 'Нет': 'No', 'Нет интернет-сервиса': 'No internet service'},
        'StreamingTV': {'Да': 'Yes', 'Нет': 'No', 'Нет интернет-сервиса': 'No internet service'},
        'StreamingMovies': {'Да': 'Yes', 'Нет': 'No', 'Нет интернет-сервиса': 'No internet service'},
        'Contract': {'Помесячный': 'Month-to-month', 'Годовой': 'One year', 'Двухлетний': 'Two year'},
        'PaperlessBilling': {'Да': 'Yes', 'Нет': 'No'},
        'PaymentMethod': {
            'Электронный чек': 'Electronic check',
            'Почтовый чек': 'Mailed check',
            'Банковский перевод': 'Bank transfer (automatic)',
            'Кредитная карта': 'Credit card (automatic)'
        }
    }

    # Применяем маппинг значений
    for col in df.columns:
        if col in value_mapping:
            df[col] = df[col].map(value_mapping[col])

    # Заполняем TotalCharges если пусто (как при обучении)
    if df['TotalCharges'].isnull().any():
        df['TotalCharges'] = df['TotalCharges'].fillna(0)

    return df


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        form_data = request.form.to_dict()

        # Валидация
        errors = {}
        for field in FORM_FIELDS:
            value = form_data.get(field['name'], '')
            if not value:
                errors[field['name']] = 'Это поле обязательно для заполнения'
            elif field['type'] == 'number':
                try:
                    float_value = float(value)
                    if 'min' in field and float_value < float(field['min']):
                        errors[field['name']] = f'Значение должно быть не менее {field["min"]}'
                except ValueError:
                    errors[field['name']] = 'Введите корректное число'

        if errors:
            return render_template('index.html',
                                   form_fields=FORM_FIELDS,
                                   form_data=form_data,
                                   field_errors=errors)

        try:
            # Загрузка модели
            model, model_name = load_ml_model()
            if model is None:
                return render_template('index.html',
                                       form_fields=FORM_FIELDS,
                                       form_data=form_data,
                                       error="Не удалось загрузить модель")

            # Преобразование данных
            input_df = preprocess_input(form_data)

            # Предсказание
            prediction_proba = model.predict_proba(input_df)[0][1]  # Вероятность класса 1 (уход)
            prediction_percent = round(prediction_proba * 100, 2)
            prediction_class = "Да (клиент уйдет)" if prediction_proba > 0.5 else "Нет (клиент останется)"

            return render_template('result.html',
                                   prediction_class=prediction_class,
                                   prediction_percent=prediction_percent,
                                   input_data=form_data,
                                   model_name=model_name)

        except Exception as e:
            return render_template('index.html',
                                   form_fields=FORM_FIELDS,
                                   form_data=form_data,
                                   error=f"Ошибка обработки данных: {str(e)}")

    return render_template('index.html', form_fields=FORM_FIELDS)


@app.route('/result')
def result():
    return render_template('result.html',
                           prediction_class=request.args.get('prediction_class', 'Неизвестно'),
                           prediction_percent=request.args.get('prediction_percent', '0'),
                           model_name=request.args.get('model_name', 'Неизвестная модель'))


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)