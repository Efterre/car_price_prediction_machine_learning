import pandas as pd 
import numpy as np



data = pd.read_excel('cleaned_merged_data.xlsx') # Read the data into a dataframe
print(data.isnull().sum()) # Check for missing values

data = data.replace('Belirtilmemiş', np.nan) # Replace 'Belirtilmemiş' with NaN in order to impute the missing values
print(data.isnull().sum()) # Check for missing values

grouped_data = data.groupby(['Brand', 'Series']) # Group the data by 'Brand' and 'Series' columns to impute the missing values based on the group



#Impute the missing values with the mode of the groups
data['Year'] = grouped_data['Year'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else data['Year'].mode()[0]))
data['Engine Power'] = grouped_data['Engine Power'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else data['Engine Power'].mode()[0]))
data['Average Fuel Consumption'] = grouped_data['Average Fuel Consumption'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else data['Average Fuel Consumption'].mode()[0]))
data['Fuel Tank'] = grouped_data['Fuel Tank'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else data['Fuel Tank'].mode()[0]))
data['Price'] = grouped_data['Price'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else data['Price'].mode()[0]))
data['Engine Volume'] = grouped_data['Engine Volume'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else data['Engine Volume'].mode()[0]))
data['Drive'] = grouped_data['Drive'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else data['Drive'].mode()[0]))
data['Vehicle Condition'] = grouped_data['Vehicle Condition'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else data['Vehicle Condition'].mode()[0]))
data['Exchangeable'] = grouped_data['Exchangeable'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else data['Exchangeable'].mode()[0]))
data['paint'] = grouped_data['paint'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else data['paint'].mode()[0]))
data['changed'] = grouped_data['changed'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else data['changed'].mode()[0]))



print(data.isnull().sum()) # Check for missing values

data.to_excel('imputed_data.xlsx', index=False) # Save the imputed data to an Excel file 


