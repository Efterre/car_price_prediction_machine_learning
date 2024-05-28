import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def kaydir(row, index):
    if (len(row) == 20):
        for i in range(len(row)-1, index, -1):
            row[i] = row[i-1]
    else:
        for i in range(len(row)-2, index, -1):
            row[i] = row[i-1]
    return row

def modelToVolume(row):
    if isinstance(row, str) and "." in row:
        stringArr = row.split(" ")
        for i in range(len(stringArr)):
            if "." in stringArr[i]:
                volume = stringArr[i]
                volume = volume.replace('.', '')
                if "D" in volume:
                    volume = volume.replace('D', '')
                if int(volume) // 100 > 0:
                    volume = int(volume)
                    volume = str(volume) + "0 cc"
                elif int(volume) // 10 > 0:
                    volume = int(volume)
                    volume = str(volume) + "00 cc"
                elif int(volume) < 10:
                    volume = int(volume)
                    volume = str(volume) + "00 cc"
                return str(volume)
    return "Belirtilmemiş"

def engineVolume(row):
    if ("İkinci" in str(row[14])):
        row[11] = modelToVolume(row[4])
        return row
    else:
        kaydir(row, 11)
        row[11] = modelToVolume(row[4])
        return row

def enginePower(row):
    kaydir(row, 12)
    row[12] = np.nan # average
    return row

def drive(row):
    kaydir(row, 13)
    row[13] = "Belirtilmemiş"
    return row

def vehicleCondition(row):
    kaydir(row,14)
    row[14] = "Belirtilmemiş"
    return row

def avrFuelConsume(row):
    kaydir(row,15)
    row[15] = np.nan # will update averag
    return row

def fuelTank(row):
    kaydir(row,16)
    row[16] = np.nan # will update average 
    return row

def isIntAvrFuelConsume(data):
    data = data.split(" ")
    if (len(data) == 2 and "lt" in data[1] and int(data[0]) // 10 < 1):
        return True
    
def fromWhom(row):
    kaydir(row,18)
    row[18] = "Belirtilmemiş"
    return row
        
def changeData(row, index, newData):
    row[index] = newData
    return row   

def dataManipulation(pathFile):
    dataFile = pd.read_excel(pathFile)   
    
    for i in range(0, len(dataFile)):
        if (dataFile.loc[i].isnull().all()):
            continue
        else:
            if ("cc" not in dataFile.iloc[i, 11] and "cm3" not in dataFile.iloc[i, 11]):
                dataFile.loc[i] = engineVolume(dataFile.loc[i])
            if ("hp" not in dataFile.iloc[i, 12] and "HP" not in dataFile.iloc[i, 12]):
                if( "-" in dataFile.iloc[i, 12] and "İkinci"  in dataFile.iloc[i,14]):
                    continue
                dataFile.loc[i] = enginePower(dataFile.loc[i])
            # Önden, Arkadan ile sınırla
            if ("Önden" not in dataFile.iloc[i, 13] and "Arkadan" not in dataFile.iloc[i, 13] and "WD" not in dataFile.iloc[i, 13] and "-" not in dataFile.iloc[i, 13]):
                dataFile.loc[i] = drive(dataFile.loc[i])
            if ("İkinci" not in dataFile.iloc[i, 14] and "ikinci" not in dataFile.iloc[i,14]):
                dataFile.loc[i] = vehicleCondition(dataFile.loc[i])
            #lt & ,
            if ("," not in dataFile.iloc[i, 15] and "-" not in dataFile.iloc[i, 15]):
                if (dataFile.iloc[i,19] == "Sahibinden" or dataFile.iloc[i,19] == "Galeriden"):
                    continue
                elif (isIntAvrFuelConsume(dataFile.iloc[i, 15]) == True):
                    continue
                dataFile.loc[i] = avrFuelConsume(dataFile.loc[i])
            elif ("lt" not in dataFile.iloc[i, 15] and "-" not in dataFile.iloc[i, 15]):
                if (dataFile.iloc[i,19] == "Sahibinden" or dataFile.iloc[i,19] == "Galeriden" or  dataFile.iloc[i,19] == "Yetkili Bayiden"):
                    continue
                dataFile.loc[i] = avrFuelConsume(dataFile.loc[i])
                
            #lt
            if ("lt" not in dataFile.iloc[i,16] and "-" not in dataFile.iloc[i, 16]):
                dataFile.loc[i] = fuelTank(dataFile.loc[i])
                
            # control peakness
            
            if ("den" in dataFile.iloc[i, 18] ):
                dataFile.loc[i] = fromWhom(dataFile.loc[i])
    if (dataFile.shape[1] == 21):
        dataFile = dataFile.dropna(subset=[dataFile.columns[20]])
    
    dataFile = dataFile.dropna(subset=[dataFile.columns[19]])
    pathFile = "updated_"+pathFile
    # save it in a new excel file
    dataFile.to_excel(pathFile, index=False)

