import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score , mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

# Veri setini yükleme
allData = pd.read_csv('encoded_data_csv.csv', encoding="utf-8")
df = pd.DataFrame(allData)

# Girdi ve çıktı verilerini ayarlama
input_data = df.drop(columns=['Price'])
output_data = df['Price']

# Veriyi eğitim ve test setlerine bölme
x_train, x_test, y_train, y_test = train_test_split(input_data , output_data, test_size=0.2, random_state=42)

# Polynomial özellikler oluşturma
pr = PolynomialFeatures(degree=4)
x_train_poly = pr.fit_transform(x_train)  # Eğitim verisi için polinomiyel dönüşüm uygula
x_test_poly = pr.transform(x_test)  # Test verisi için aynı dönüşümü uygula

# Lineer Regresyon modelini eğitme
lr2 = LinearRegression()
lr2.fit(x_train_poly, y_train)

# Test verileri üzerinde tahmin yapma
y_pred = lr2.predict(x_test_poly)

# Metriklerin hesaplanması
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
rae = mae / mean_absolute_error(y_test, [np.mean(y_test)]*len(y_test))
rse = mse / mean_squared_error(y_test, [np.mean(y_test)]*len(y_test))

# Sonuçların yazdırılması
print(f"Modelin R^2 skoru: {r2}")
print(f"Modelin Ortalama Kare Hatası (MSE): {mse}")
print(f"Modelin Ortalama Mutlak Hatası (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Relative Absolute Error (RAE): {rae}")
print(f"Relative Squared Error (RSE): {rse}")

#DERECE 2 İKEN SONUÇLAR
# Modelin R^2 skoru: 0.9024092997149259
# Modelin Ortalama Kare Hatası (MSE): 10060223134.479958
# Modelin Ortalama Mutlak Hatası (MAE): 66488.53634512264
# Root Mean Squared Error (RMSE): 100300.66367915996
# Relative Absolute Error (RAE): 0.29984122441277616
# Relative Squared Error (RSE): 0.0975907002850741

#DERECE 3 İKEN SONUÇLAR
# Modelin R^2 skoru: 0.9024092997149259
# Modelin Ortalama Kare Hatası (MSE): 10060223134.479958
# Modelin Ortalama Mutlak Hatası (MAE): 66488.53634512264
# Root Mean Squared Error (RMSE): 100300.66367915996
# Relative Absolute Error (RAE): 0.29984122441277616
# Relative Squared Error (RSE): 0.0975907002850741

#DERECE 4 İKEN SONUÇLAR
# Modelin R^2 skoru: 0.9024092997149259
# Modelin Ortalama Kare Hatası (MSE): 10060223134.479958
# Modelin Ortalama Mutlak Hatası (MAE): 66488.53634512264
# Root Mean Squared Error (RMSE): 100300.66367915996
# Relative Absolute Error (RAE): 0.29984122441277616
# Relative Squared Error (RSE): 0.0975907002850741

#DERECE 5 İKEN SONUÇLAR
# Modelin R^2 skoru: 0.20311120417583806
# Modelin Ortalama Kare Hatası (MSE): 82147982091.93958
# Modelin Ortalama Mutlak Hatası (MAE): 90146.73802800119
# Root Mean Squared Error (RMSE): 286614.69273563
# Relative Absolute Error (RAE): 0.40653185936941516
# Relative Squared Error (RSE): 0.7968887958241619

