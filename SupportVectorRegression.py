import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score , mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Veri yükleme
allData = pd.read_csv('encoded_data_csv.csv', encoding="utf-8")
df = pd.DataFrame(allData)

# Giriş ve çıkış verilerini hazırlama
input_data = df.drop(columns=['Price'])
output_data = df['Price']

# Veriyi eğitim ve test setlerine ayırma
x_train, x_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.2, random_state=42)

# StandardScaler ile ölçeklendirme
sc = StandardScaler()
x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.transform(x_test)

# y_train'i 2D yapısına dönüştürme
y_train_scaled = sc.fit_transform(y_train.values.reshape(-1, 1)).flatten()

# SVR modelini eğitme
svr = SVR(kernel="sigmoid")
# 'linear','poly','rbf','sigmoid
svr.fit(x_train_scaled, y_train_scaled)

# Test verileri üzerinde tahmin yapma
y_pred_scaled = svr.predict(x_test_scaled)
y_pred = sc.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

# Doğruluk metrikleri
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
rae = mae / mean_absolute_error(y_test, [np.mean(y_test)]*len(y_test))
rse = mse / mean_squared_error(y_test, [np.mean(y_test)]*len(y_test))

# Metrikleri yazdırma
print(f"Modelin R^2 skoru: {r2}")
print(f"Modelin Ortalama Kare Hatası (MSE): {mse}")
print(f"Modelin Ortalama Mutlak Hatası (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Relative Absolute Error (RAE): {rae}")
print(f"Relative Squared Error (RSE): {rse}")

# "rbf" için
# Modelin R^2 skoru: 0.9496308272882853
# Modelin Ortalama Kare Hatası (MSE): 5192350450.389271
# Modelin Ortalama Mutlak Hatası (MAE): 27019.74187545865
# Root Mean Squared Error (RMSE): 72057.96590516048
# Relative Absolute Error (RAE): 0.12185006517817547
# Relative Squared Error (RSE): 0.050369172711714705

#linear için
# Modelin R^2 skoru: 0.7326653215840713
# Modelin Ortalama Kare Hatası (MSE): 27558430348.306667
# Modelin Ortalama Mutlak Hatası (MAE): 74363.51470680498
# Root Mean Squared Error (RMSE): 166007.32016482487
# Relative Absolute Error (RAE): 0.3353547622944709
# Relative Squared Error (RSE): 0.26733467841592873

#poly
# Modelin R^2 skoru: 0.9104394626844343
# Modelin Ortalama Kare Hatası (MSE): 9232426725.154999
# Modelin Ortalama Mutlak Hatası (MAE): 49091.039305168844
# Root Mean Squared Error (RMSE): 96085.51777013537
# Relative Absolute Error (RAE): 0.22138428881262803
# Relative Squared Error (RSE): 0.08956053731556571

#sigmoid
# Modelin R^2 skoru: -44407.56928493628
# Modelin Ortalama Kare Hatası (MSE): 4577896405952947.0
# Modelin Ortalama Mutlak Hatası (MAE): 28447206.362725925
# Root Mean Squared Error (RMSE): 67660153.75354202
# Relative Absolute Error (RAE): 128.2874560908929
# Relative Squared Error (RSE): 44408.569284936275