import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

allData = pd.read_csv('encoded_data_csv.csv', encoding="utf-8")
df = pd.DataFrame(allData)
df.head()

input_data = df.drop(columns=['Price'])
output_data = df['Price']


x_train, x_test, y_train, y_test = train_test_split(input_data , output_data, test_size=0.2, random_state=42)
model = LinearRegression() # model creation

# train model
model.fit(x_train, y_train)

predict = model.predict(x_test)


input_data_model = pd.DataFrame([[24,281,1372,2014,225000.0,2,2,0,3,1461,110,0,4.4,60,0,1,0,0]],
 columns = ['Brand','Series','Model','Year','Mileage','Transmission Type','Fuel Type','Body Type','Color','Engine Volume','Engine Power','Drive','Average Fuel Consumption','Fuel Tank','Exchangeable','From Whom','paint','changed'])

# print (input_data_model)
#
#print(model.predict(input_data_model))





#accuracy
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Modelinizin test verileri üzerinde tahmin yapması
y_pred = model.predict(x_test)

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

# Price" kolonunun veri setinizdeki istatistiksel özeti aşağıdaki gibidir: (ESKİiii)

# Ortalama (mean): 669,193.8 TL
# Standart Sapma (std): 1,583,747 TL
# Minimum Değer (min): 123,456 TL
# Maksimum Değer (max): 338,000,000 TL
# Medyan (50% quantile): 600,000 TL


# Modelin R^2 skoru: 0.7688722965654105
# Modelin Ortalama Kare Hatası (MSE): 23826002501.464832
# Modelin Ortalama Mutlak Hatası (MAE): 85973.99616906734
# Root Mean Squanred Error (RMSE): 154356.7377909524
# Relative Absolute Error (RAE): 0.3877141789553546
# Relative Squared Error (RSE): 0.23112770343458952
