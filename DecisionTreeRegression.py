import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

allData = pd.read_csv('encoded_data_csv.csv', encoding="utf-8")
df = pd.DataFrame(allData)
df.head()

input_data = df.drop(columns=['Price'])
output_data = df['Price']


x_train, x_test, y_train, y_test = train_test_split(input_data , output_data, test_size=0.2, random_state=42)
dt = DecisionTreeRegressor()

# train model
dt.fit(x_train, y_train)

#accuracy
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Modelinizin test verileri üzerinde tahmin yapması
y_pred = dt.predict(x_test)

# Gerçek değerlerle tahmin edilen değerlerin karşılaştırılması
r2 = r2_score(y_test, y_pred)

print(f"Modelin R^2 skoru: {r2}")

#Ortalama Kare Hatası (Mean Squared Error - MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Modelin Ortalama Kare Hatası (MSE): {mse}")

# Ortalama Mutlak Hata (Mean Absolute Error - MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f"Modelin Ortalama Mutlak Hatası (MAE): {mae}")

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squanred Error (RMSE): {rmse}")

rae = mean_absolute_error(y_test, y_pred) / mean_absolute_error(y_test, [np.mean(y_test)]*len(y_test))
print(f"Relative Absolute Error (RAE): {rae}")

rse = mean_squared_error(y_test, y_pred) / mean_squared_error(y_test, [np.mean(y_test)]*len(y_test))
print(f"Relative Squared Error (RSE): {rse}")


# Modelin R^2 skoru: 0.9933245421284941
# Modelin Ortalama Kare Hatası (MSE): 688145443.3260274
# Modelin Ortalama Mutlak Hatası (MAE): 5421.068914646997
# Root Mean Squanred Error (RMSE): 26232.526438107852
# Relative Absolute Error (RAE): 0.02444722098493048
# Relative Squared Error (RSE): 0.00667545787150588
