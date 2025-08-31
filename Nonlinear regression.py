# 1. Импортирование бибилиотек
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy import optimize
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 2. Чтение файлов, анализ данных

# Чтение файлов
train = pd.read_csv('C:\\Users\\user\\Desktop\\Nonlinear regression\\train.csv')
k_test = pd.read_csv('C:\\Users\\user\\Desktop\\Nonlinear regression\\test.csv')

train.head(100) #проверка корректности показа данных
train.shape #проверка размерности данных

train.plot(kind='scatter', x = 'k', y = 'x', alpha = 0.9) #построение графика
plt.title("Train")
plt.grid()

df = pd.DataFrame(train) #разделение данных
x_full_train = df['x']
k_full_train = df['k']

#Вычисление параметров для угловой частоты
peak, _  = find_peaks(x_full_train, distance=10) #нахождение пиков
T = k_full_train[peak[1]] - k_full_train[peak[0]] #находим расстояние между первыми двумя пиками. Это период Т
f = 1/T #считаем частоту

# Анализ данных для предположения коэффициентов a, w, fi, b, c
def coefficient ():
    a = (x_full_train.max()-x_full_train.min())/2 #амплитуда
    w = 2*np.pi*f #угловая частота
    b, c = np.polyfit(k_full_train, x_full_train, 1) #линейный тренд и смещение
    fi = 0 #фаза
    return a, w, fi, b, c

# 3. Разделение данных, обучение модели, поиск коэффициентов и проверка модели на точность

#функция для сгенерированной формулы. Данные уже зашумлены
def training (k_train, a, w, fi, b, c):
    return a * np.sin(w * k_train + fi) + b * k_train + c

#Разделение данных для обучения и проверки
Train, Val = train_test_split(df, test_size=0.2, random_state=42)
x_train, k_train = Train['x'],Train['k']
x_val, k_val = Val['x'],Val['k']

# Метод curve_fit из-за простоты и быстроты исполнения. Нам известна аналитическая формула для расчета. Данный метод позволит быстро подобрать
# коэффициенты для нелинейной регрессии и провести предсказанием

#Модель для обучения, поиска коэффициентов и предсказаний
def fit_and_predict (k_values, x_values, k_pred):
    # Первоначальное предположение о параметрах [a, w, fi, b, c]
    a, w, fi, b, c = coefficient()
    initial_guess = [a, w, fi, b, c]

    # Подгонка к кривой
    params, covariance = curve_fit(training, k_values, x_values, p0=initial_guess)

    # Извлечение коэффициентовs
    a, w, fi, b, c = params

    print(f"Fitted parameters: a={round(a, 2)}, w={round(w, 2)}, fi={round(fi, 2)}, b={round(b, 2)}, c={round(c, 2)}")
    # Генерировать значения x, используя установленные параметры
    x_pred = training(k_pred, a, w, fi, b, c)
    return x_pred

#Обучение модели
x_pred = fit_and_predict(k_train, x_train, k_val)

#Проверка модели оценкой между фактическим и предсказанным значением
mse = mean_squared_error(x_val, x_pred)
print(f'MSE = {mse}')

# В результате расчеты была получена MSE = 0.14. Это показывает, что модель работает вполне корректно
# и имеет реалистичные предсказания для тестовых результатов. Чем меньше MSE, тем точнее модель делает предсказания.

# 4. Финальное обучение модели на всем train, получения предсказаний для файла test, заполнение таблицы и визуальная демонстрация предсказания

# Обучение на всем файле train
k = pd.DataFrame(k_test)
x_test = fit_and_predict(k_full_train, x_full_train, k_test['k'])
print(x_test)

#заполнение таблицы предсказаний и сохранение
k['x']= x_test
k.to_csv('pred.csv', index=False)

#визуальная демонстрация предсказания и тренировочных данных
ax = train.plot(kind='scatter', x='k', y='x', alpha=0.7, color='blue', label='train')
k.plot(kind='scatter', x='k', y='x', alpha=0.7, color='red', label='test', ax=ax)
plt.title("Prediction")
plt.grid()
plt.legend()
plt.show()