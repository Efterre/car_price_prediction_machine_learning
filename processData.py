import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

allData = pd.read_csv('imputed_last_data_csv.csv', delimiter=";", encoding="utf-8")
df = pd.DataFrame(allData)
df.head()

#tekrar eden rowları siliyor
print(len(df))
df = df.drop_duplicates(subset=['Ad No'])
print(len(df))

df = df.drop(columns=['Ad No','Ad Date', 'Vehicle Condition'])  # Bu columnlar fiyat tahmininde kullanılmayacak

 # Dönüşüm yapmak için months'ı tanımladım
months = { 'Oca': '1','Şub': '2', 'Mar': '3', 'Nis': '4', 'May': '5', 'Haz': '6', 'Tem': '7', 'Ağu': '8', 'Eyl': '9', 'Eki': '10', 'Kas': '11', 'Ara': '12'} 

# Average Fuel Consumption için Dönüşüm fonksiyonu
def convert_date_AFC(date):
    if '.' in date:
        day, month = date.split('.')
        month_num = months.get(month, month)
        return f"{day}.{month_num}"
    return date

# DataFrame oluşturma ve dönüşüm uygulama
df['Average Fuel Consumption'] = df['Average Fuel Consumption'].apply(convert_date_AFC)

def convert_date_Model(date):
    for ay, num in months.items():
        if f'1.{ay}' in date:
            return date.replace(f'1.{ay}', f'1.{num}')
    return date

# Dönüşümü uygulama
df['Model'] = df['Model'].apply(convert_date_Model)




# az unique value'ya sahip columnların encoding halleri rakamları değiştirebilriz.
df['Exchangeable'] = df['Exchangeable'].replace({'Takasa Uygun': 0, 'Takasa Uygun Değil': 1})
df['From Whom'] = df['From Whom'].replace({ 'Yetkili Bayiden' : 0,'Sahibinden': 1, 'Galeriden': 2})
df['Drive'] = df['Drive'].replace({ 'Önden Çekiş' : 0,'Arkadan İtiş': 1, '4WD (Sürekli)': 2,'AWD (Elektronik)':3})
df['Transmission Type'] = df['Transmission Type'].replace({'Düz': 0, 'Otomatik': 2, 'Yarı Otomatik': 1})
df['Fuel Type'] = df['Fuel Type'].replace({'LPG & Benzin': 0, 'Benzin': 1, 'Dizel': 2, 'Hibrit': 3})
df['Body Type'] = df['Body Type'].replace({'Sedan': 0, 'Hatchback/3': 1, 'Hatchback/5': 2, 'Station wagon': 3, 'Coupe': 4, 'Cabrio': 5, 'Roadster': 6, 'MPV': 7, 'SUV': 8, 'Pick-up': 9})
def encode_column(df, column_name):  # çok unique value içieren columndaki değerleri encode ediyor
    unique_values = df[column_name].unique().tolist()
    encoding = {value: idx + 1 for idx, value in enumerate(unique_values)}
    df[column_name] = df[column_name].map(encoding)
    return encoding

brands = df['Brand'].unique().tolist()
series = df['Series'].unique().tolist()
models = df['Model'].unique().tolist()
colors = df['Color'].unique().tolist()

brands_encoding = encode_column(df, 'Brand')
series_encoding = encode_column(df, 'Series')
model_encoding = encode_column(df, 'Model')
color_encoding = encode_column(df, 'Color')

# df['Price'] = df['Price'].str.replace('.', '') #regression için .'ları sildik
# df['Price'] = df['Price'].astype(float)  #regression için float'a çevirdik
# df['Mileage'] = df['Mileage'].str.replace('.', '') #regression için .'ları sildik
# df['Mileage'] = df['Mileage'].astype(float) #regression için float'a çevirdik

for column in df.columns:
    print("Unique Values of "+ column)
    print(df[column].unique())
    print("/////////////////////////////////////\n")
    
df.to_csv('encoded_data_csv.csv', index = False)

# allData = pd.read_csv('encoded_data_csv.csv', encoding="utf-8")
# df = pd.DataFrame(allData)
# df.head()
#
# input_data = df.drop(columns = ['Price'])
# output_data = df['Price']
#
# x_train, x_test, y_train, y_test = train_test_split(input_data , output_data, test_size=0.2, random_state=42)
# model = LinearRegression() # model creation
#
# # train model
# model.fit(x_train, y_train)
#
# predict = model.predict(x_test)
#
# print(predict)
#
# print(x_train.head(1))
#
# input_data_model = pd.DataFrame([[24,281,1372,2014,225000.0,2,2,0,3,1461,110,0,4.4,60,0,1,0,0]],
# columns = ['Brand','Series','Model','Year','Mileage','Transmission Type','Fuel Type','Body Type','Color','Engine Volume','Engine Power','Drive','Average Fuel Consumption','Fuel Tank','Exchangeable','From Whom','paint','changed'])
#
# print (input_data_model)
#
# print(model.predict(input_data_model))