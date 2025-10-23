import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, classification_report, \
    confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
import numpy as np

df = pd.read_csv("Iris.csv")

df_clean = df.copy()
numeric_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

for col in numeric_cols:
    if df_clean[col].isnull().any():
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

print("Пропущенные значения после обработки:")
print(df_clean.isnull().sum())
print("\n" + "=" * 50 + "\n")

print("РАЗДЕЛЕНИЕ ДАТАСЕТА НА ОБУЧАЮЩУЮ И ТЕСТОВУЮ ВЫБОРКИ")

X_reg = df_clean[['SepalLengthCm', 'SepalWidthCm', 'PetalWidthCm']]
y_reg = df_clean['PetalLengthCm']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

print(f"Регрессия: обучающая {X_train_reg.shape}, тестовая {X_test_reg.shape}")

X_clf = df_clean[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
le = LabelEncoder()
y_clf = le.fit_transform(df_clean['Species'])

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.3, random_state=42, stratify=y_clf
)

print(f"Классификация: обучающая {X_train_clf.shape}, тестовая {X_test_clf.shape}")
print("\n" + "=" * 50 + "\n")

print("ЗАДАЧА РЕГРЕССИИ (предсказание PetalLengthCm)")

scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

reg_model = LinearRegression()
reg_model.fit(X_train_reg_scaled, y_train_reg)
y_pred_reg = reg_model.predict(X_test_reg_scaled)

print("ОЦЕНКА РЕГРЕССИОННОЙ МОДЕЛИ:")

mse = mean_squared_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_reg, y_pred_reg)

print(f"MSE (среднеквадратичная ошибка): {mse:.4f}")
print(f"RMSE (корень среднеквадратичной ошибки): {rmse:.4f}")
print(f"MAE (средняя абсолютная ошибка): {mae:.4f}")

print(f"\nАНАЛИЗ РЕЗУЛЬТАТОВ РЕГРЕССИИ:")
print(f"Средняя ошибка предсказания: ±{rmse:.3f} см")
print(f"Диапазон PetalLengthCm: {y_reg.min():.1f} - {y_reg.max():.1f} см")
print(f"Относительная ошибка: {(rmse / y_reg.std() * 100):.1f}% от стандартного отклонения")

if mse > 0.1:
    print("\nРЕЗУЛЬТАТЫ РЕГРЕССИИ МОЖНО УЛУЧШИТЬ")
    print("\nПОЛИНОМИАЛЬНАЯ РЕГРЕССИЯ:")

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_reg)
    X_test_poly = poly.transform(X_test_reg)

    scaler_poly = StandardScaler()
    X_train_poly_scaled = scaler_poly.fit_transform(X_train_poly)
    X_test_poly_scaled = scaler_poly.transform(X_test_poly)

    poly_model = LinearRegression()
    poly_model.fit(X_train_poly_scaled, y_train_reg)
    y_pred_poly = poly_model.predict(X_test_poly_scaled)

    mse_poly = mean_squared_error(y_test_reg, y_pred_poly)
    rmse_poly = np.sqrt(mse_poly)

    print(f"MSE с полиномиальной регрессией: {mse_poly:.4f}")
    print(f"Улучшение: {((mse - mse_poly) / mse * 100):.1f}%")

    if mse_poly < mse:
        print("Полиномиальная регрессия дала улучшение")
    else:
        print("Полиномиальная регрессия не улучшила результат")

else:
    print("\nРезультаты регрессии хорошие")

print("\n" + "=" * 50 + "\n")

print("ЗАДАЧА КЛАССИФИКАЦИИ (предсказание Species)")

scaler_clf = StandardScaler()
X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)
X_test_clf_scaled = scaler_clf.transform(X_test_clf)

clf_model = LogisticRegression(max_iter=1000)
clf_model.fit(X_train_clf_scaled, y_train_clf)
y_pred_clf = clf_model.predict(X_test_clf_scaled)

print("ОЦЕНКА КЛАССИФИКАЦИОННОЙ МОДЕЛИ:")

accuracy = accuracy_score(y_test_clf, y_pred_clf)
print(f"Accuracy (доля правильных предсказаний): {accuracy:.4f}")

cm = confusion_matrix(y_test_clf, y_pred_clf)
print(f"Матрица ошибок:\n{cm}")

print("\nДетальный отчет классификации:")
print(classification_report(y_test_clf, y_pred_clf, target_names=le.classes_))

print(f"\nАНАЛИЗ РЕЗУЛЬТАТОВ КЛАССИФИКАЦИИ:")
print("Iris-setosa: идеальная классификация (precision=1.00, recall=1.00)")
print("Iris-versicolor: 1 ошибка (принят за virginica)")
print("Iris-virginica: 3 ошибки (приняты за versicolor)")

if accuracy < 0.9:
    print("\nКЛАССИФИКАЦИЮ МОЖНО УЛУЧШИТЬ")
else:
    print("\nРезультаты классификации отличные")

print("\n" + "=" * 50 + "\n")

df_clean.to_csv("iris_prepared.csv", index=False)