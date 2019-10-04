#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import datetime
import time
import numpy as np
import iexfinance
from iexfinance.stocks import get_historical_intraday
from collections import Counter
import os
import copy
from sklearn.externals import joblib


# In[2]:


#формирование списка sp_500
def get_sp500(k=3):
    
    sp500=pd.read_excel('C:/Users/user/Documents/Модуль биржа по фракталам/sp500_symbols.xlsx')
    sp500=list(sp500['symbol'])
    return sp500[1:len(sp500):k]

#Тренировочные данные - загрузка
def traindata_download(k=1):

    companies={}
    path='C:/Users/user/Documents/Модуль биржа по фракталам/minute_by_andrey/'
    for root, dirs, files in os.walk(path):
        files=[files[i] for i in range(len(files)) if i%k==0] #БЕРЕМ ПОКА ТОЛЬКО КАЖДЫЙ 30-ЫЙ ФАЙЛ
        for _file in files:
            try:
                text=str(_file)
                splitted_text = text.split(".")
                tiker=splitted_text[0]
                if tiker in sp500:

                    companies[tiker]=pd.read_csv('minute_by_andrey/'+_file)
            except:
                pass
    return companies

#Тренировочные данные - приведение к нужному виду
def traindata_making_standard(companies):
    DF={}
    step=0
    for i in companies.keys():
        try:

            df=companies[i].copy(deep=True)

            df.drop(['OPEN','CLOSE','VOLUME'],axis=1,inplace=True)

            df.index=pd.to_datetime(df['Date'])
            df.drop('Date',axis=1,inplace=True)

            df.rename(columns={'HIGH': 'High', 'LOW': 'Low'}, inplace=True)
            df['company']=str(i)
            df=df[df['High'].notnull()]
            
            if step == 0:
                indexes=pd.date_range(start=df.index[0], end=df.index[-1], freq='1Min')
                dfs = pd.DataFrame(index=indexes)
                dfs['day_number'] = dfs.index.date
                dti=pd.date_range(start=df.index[0], end=df.index[-1], freq='B')
                dfs_filtered=(dfs.loc[dfs['day_number'].isin(pd.DataFrame(index=dti).index.date)])
                dfs_filtered_between_time=dfs_filtered.between_time(start_time='09:30', end_time='16:00')
            step = step + 1
            df_filtered = (df.index.isin(dfs_filtered_between_time.index))
            
            
                
                
            
            
            DF[i]=df[df_filtered]

        except:
            pass
    return DF

#Данные: ресэмплирование на нужные минутные интервалы перед подсчетом фрактальных точек
def resampling_standard_data(DF, minutes=5):
    
    resample_time=str(minutes)+str('Min')

    DF_1={}
    for i in DF.keys():
        try:

            df1=DF[i].copy(deep=True)
            df=pd.DataFrame()
            df['High'], df['Low']=df1['High'].resample(resample_time).max(), df1['Low'].resample(resample_time).min()
            df['company']=str(i)
            df=df[df['High'].notnull()]


            DF_1[i]=df

        except:
            pass
    return DF_1

#Данные: после ресэмплирования, получение фрактальных среднесрочных точек для одной фирмы
def get_fractal_points_for_one_df(df):
    
    hh=(df['High']>df['High'].shift(1))&(df['High']>df['High'].shift(-1))
    ll=(df['Low']<df['Low'].shift(1))&(df['Low']<df['Low'].shift(-1))

    df_new_hh=pd.DataFrame({'High':df[hh].High})
    frac_hh=(df_new_hh['High']>df_new_hh['High'].shift(1))&(df_new_hh['High']>df_new_hh['High'].shift(-1))

    df_new_ll=pd.DataFrame({'Low':df[ll].Low})
    frac_ll=(df_new_ll['Low']<df_new_ll['Low'].shift(1))&(df_new_ll['Low']<df_new_ll['Low'].shift(-1))

    df_newest=pd.DataFrame({'Srhigh':df_new_hh[frac_hh].High,'Srlow':df_new_ll[frac_ll].Low})
    
    return df_newest


# In[3]:


#ПОДСЧЕТ МОМЕНТОВ УВЕРЕННОГО РАЗВОРОТА ПО РЕСЭМПЛИРОВАННЫМ ФРАКТАЛЬНЫМ ТОЧКАМ - по одной фирме.
# Возвращается список таймстэмпов
def get_high_low_timestamps_for_one_df(df_newest):
    spisok_low=[]
    spisok_high=[]
    for y in range(10, len(df_newest)):
        try:
            if df_newest['Srlow'].iat[y] >= 0:
                y_point=y
                y=y-1
                while not df_newest['Srlow'].iat[y] >= 0:
                    y=y-1
                srlow3=y

                y=y-1
                while not df_newest['Srlow'].iat[y] >= 0:
                    y=y-1
                srlow2=y

                y=y_point-1                     
                while not df_newest['Srhigh'].iat[y] >= 0:
                    y=y-1
                srhigh=y                    

                if srhigh<srlow3:
                    if df_newest['Srlow'].iat[y_point]>df_newest['Srlow'].iat[srlow3] and df_newest['Srlow'].iat[srlow2]>df_newest['Srlow'].iat[srlow3]:
                        moment_low=df_newest.index[srlow3]
                        spisok_low.append(moment_low)
        except:
            pass
    # а теперь по моментам хай    
    for y in range(10, len(df_newest)):
        try:
            if df_newest['Srhigh'].iat[y] >= 0:
                y_point=y
                y=y-1
                while not df_newest['Srhigh'].iat[y] >= 0:
                    y=y-1
                srhigh3=y

                y=y-1
                while not df_newest['Srhigh'].iat[y] >= 0:
                    y=y-1
                srhigh2=y

                y=y_point-1                     
                while not df_newest['Srlow'].iat[y] >= 0:
                    y=y-1
                srlow=y                    

                if srlow<srhigh3:
                    if df_newest['Srhigh'].iat[y_point]<df_newest['Srhigh'].iat[srhigh3] and df_newest['Srhigh'].iat[srhigh2]<df_newest['Srhigh'].iat[srhigh3]:
                        moment_high=df_newest.index[srhigh3]
                        spisok_high.append(moment_high)
        except:
            pass
    return spisok_high, spisok_low


# In[4]:


#Получение полного списка таймстэмпов по всем фирмам
def get_timestamps_for_all(traindata_resampled):
    companies = traindata_resampled.keys()
    spisok_high_global = []
    spisok_low_global = []
    for company in companies:
        df=traindata_resampled[company].copy(deep=True)
        df_newest = get_fractal_points_for_one_df(df)
        spisok_high, spisok_low = get_high_low_timestamps_for_one_df(df_newest)
        spisok_high_global.append(spisok_high)
        spisok_low_global.append(spisok_low)
    spisok_high_all = []
    spisok_low_all = []    
    for item in spisok_high_global:
        for i in item:
            spisok_high_all.append(i)
    for item in spisok_low_global:
        for i in item:
            spisok_low_all.append(i)            
            
    
    return spisok_high_all, spisok_low_all


# In[5]:


#получение чернового датасета по минутам
def get_unsampled_dataset(traindata_standard, times = [3,7,15]):
    step = 0
    for minute in times:
        traindata_resampled = resampling_standard_data(traindata_standard, minutes=minute)
        spisok_high_all, spisok_low_all = get_timestamps_for_all(traindata_resampled)
        if step == 0:
            max_time = max(max(spisok_high_all), max(spisok_low_all))
            min_time = min(min(spisok_high_all), min(spisok_low_all))
            indexes=pd.date_range(start=min_time, end=max_time, freq='1Min')
            
            
        
            
            dfs = pd.DataFrame(index=indexes)
            dfs['day_number'] = dfs.index.date
            dti=pd.date_range(start=min_time, end=max_time, freq='B')
            dfs_filtered=(dfs.loc[dfs['day_number'].isin(pd.DataFrame(index=dti).index.date)])
            dfs_filtered_between_time=dfs_filtered.between_time(start_time='09:30', end_time='16:00')
            
            
            unsampled_dataset = pd.DataFrame(index=dfs_filtered_between_time.index)
        step = step + 1
            
            
            
            
            
            
            
        
        slovar_high = Counter(spisok_high_all)
        slovar_low = Counter(spisok_low_all)

        unsampled_dataset['highes'+str(minute)] = pd.Series(data=list(slovar_high.values()), index=list(slovar_high.keys()))
        unsampled_dataset['lowes'+str(minute)] = pd.Series(data=list(slovar_low.values()), index=list(slovar_low.keys()))

    return unsampled_dataset


# # Предподготовка traindata

# In[6]:


#traindata = traindata_download(1)


# In[7]:


#traindata_standard = traindata_making_standard(traindata)


# # Предподготовка testdata

# In[8]:


#Загрузка тестовых данных
def testdata_download(start_day=datetime.datetime(2019,4,1), end_day=datetime.datetime(2019,6,18)):
    dti=pd.date_range(start_day,end_day, freq='B')
    companies={}
    for i in sp500:

        get=pd.DataFrame()
        for date in dti:
            try:
                get_current=pd.read_excel('C:/Users/user/Documents/Модуль биржа по фракталам/iex_intraday/'+str(i)+'_'+str(date)[:-9]+'.xlsx', dtype={'date':object})
                get=get.append(get_current)
            except:
                continue

        companies[i]=get
    return companies

#Тестовые данные: преобразование в формат готовности перед ресэмплированием
def testdata_making_standard(companies):
    traindata_standard={}
    for company in companies.keys():
        if len(companies[company])>10:
            example=companies[company]

            example=example[['high','low','close']]
            example=example[example['close'].notnull()]
            example.drop('close',axis=1,inplace=True)
            #massiv=[0]*len(example)
            #for i in range(len(example)):
                #vremya=str(example.iloc[i].date)+str(example.iloc[i].minute)
                #massiv[i]=datetime.datetime.strptime(vremya,"%Y%m%d%H:%M")
            #example.index=np.asarray(massiv)
            #example.drop(['date','minute'], axis=1,inplace=True)
            example.rename(columns={'high': 'High', 'low': 'Low'}, inplace=True)
            example.index.name='Date'
            example['company']=str(company)
            traindata_standard[company]=example
    return traindata_standard

def get_traindata_standard_with_mart_2019(traindata_standard):
    for company in traindata_standard.keys():
        try:
            df = pd.read_excel('C:/Users/user/Documents/Модуль биржа по фракталам/test_data_mart_2019/'+str(company)+'_'+'mart_2019.xlsx', index_col='Date')
            traindata_standard[company]=df.append(traindata_standard[company])
        except:
            pass
    return traindata_standard

def traindata_standard_together(traindata_standard_previous, traindata_standard_current):
    traindata_standard={}
    for company in traindata_standard_previous.keys():
        try:
            df = traindata_standard_previous[company]
            traindata_standard[company]=df.append(traindata_standard_current[company])
        except:
            pass
    return traindata_standard


# # Общая работа с данными для получения ready_dataset_for_marking

# In[9]:


#ресэмплирование к периоду 1 час всех периодов (3, 7, 15 минут) - методом суммы по каждому периоду
#оставляем только время от 09.00 до 16.00
def get_resampled_dataset(unsampled_dataset):
    resampled_dataset = unsampled_dataset.resample('H', label = 'right').sum()
    return resampled_dataset


# In[10]:


#формирование датасета из цен фирм - для получения индекса рынка по часам
def get_dataset_for_market_index(traindata_standard):
    step = 0
    companies = traindata_standard.keys()
    DF={}
    for company in companies:
        df=traindata_standard[company].copy(deep=True)
        df['price'] = (df['High'] + df['Low']) / 2
        #df.drop(columns=['High', 'Low', 'company'], inplace=True)
        DF[company] = df['price']
    pddf=pd.DataFrame(DF)
    df=pddf.resample('H', label = 'right').mean()

    if step == 0:
        indexes=pd.date_range(start=df.index[0], end=df.index[-1], freq='H')
        dfs = pd.DataFrame(index=indexes)
        dfs['day_number'] = dfs.index.date
        dti=pd.date_range(start=df.index[0], end=df.index[-1], freq='B')
        dfs_filtered=(dfs.loc[dfs['day_number'].isin(pd.DataFrame(index=dti).index.date)])
        dfs_filtered_between_time=dfs_filtered.between_time(start_time='09:30', end_time='16:00')
    df_filtered = (df.index.isin(dfs_filtered_between_time.index))
    pddf=df[df_filtered]
    
    
    step = step + 1    
    

    
    #pddf=pddf.between_time(start_time='09:00', end_time='16:00')
    #pddf=pddf.between_time(start_time='08:00', end_time='16:00')
    dataset_for_market_index=pddf/pddf.shift(1)-1
    dataset_for_market_index=(1+dataset_for_market_index).cumprod()
    return dataset_for_market_index


# In[11]:


#получение столбца для добавления - индекс рынка по часам
def get_market_index_by_hour(dataset_for_market_index):
    return dataset_for_market_index.transpose().mean()


# In[12]:


def get_market_index_rolling(market_index_by_hour):
    market_index_by_hour.dropna(inplace=True)
    market_index_rolling = market_index_by_hour - market_index_by_hour.rolling(32).mean()
    return market_index_rolling


# In[13]:


def get_market_index_for_plotting(market_index_by_hour):
    market_index_by_hour.dropna(inplace=True)
    market_index_for_plotting = market_index_by_hour
    return market_index_for_plotting


# In[14]:


#получение датасета (с индексом рынка и только по бизнес-дням) для вычисления финального датасета
def get_dataset_for_calculation(resampled_dataset, market_index_rolling):
    resampled_dataset['market_index_rolling'] = market_index_rolling
    mask = (resampled_dataset['market_index_rolling'].notnull())
    dataset_for_calculation = resampled_dataset[mask]
    return dataset_for_calculation


# In[15]:


#получаем имена столбцов датасета
def get_224__columnnames_for_dataset():
    columns_mask = ['highes3','lowes3','highes7','lowes7','highes15','lowes15','market_index_rolling']
    global_mask = []
    for i in range(32):
        mask = []
        for j in columns_mask:
            imya = str(i) + '_' + str(j)
            mask.append(imya)
        global_mask = global_mask + mask
    return global_mask


# In[16]:


#простираем вправо на 32 часа прежний датасет, с именами столбцов. датасет уже для маркировки
def get_ready_dataset_for_marking(dataset_for_calculation):
    calculated_dataset_for_marking = []
    dataset_for_marking_indexes = []
    for row in range(31, len(dataset_for_calculation)):
        row_list=[]
        for item in range(32):
            row_list = row_list + list(dataset_for_calculation.iloc[row - item].values)
        calculated_dataset_for_marking.append(row_list)
        dataset_for_marking_indexes.append(dataset_for_calculation.index[row])
    global_mask = get_224__columnnames_for_dataset()
    ready_dataset_for_marking = pd.DataFrame(data=calculated_dataset_for_marking, index=dataset_for_marking_indexes, columns=global_mask)
    return ready_dataset_for_marking


# In[17]:


sp500 = get_sp500(3)
traindata_standard=joblib.load('traindata_standard.pkl')


# In[22]:


unsampled_dataset = get_unsampled_dataset(traindata_standard, times = [3,7,15])
resampled_dataset = get_resampled_dataset(unsampled_dataset)
dataset_for_market_index = get_dataset_for_market_index(traindata_standard)
market_index_by_hour=get_market_index_by_hour(dataset_for_market_index)
market_index_rolling = get_market_index_rolling(market_index_by_hour)
dataset_for_calculation = get_dataset_for_calculation(resampled_dataset, market_index_rolling)
ready_dataset_for_marking = get_ready_dataset_for_marking(dataset_for_calculation)
ready_dataset_for_marking.to_excel('ready_dataset_for_marking.xlsx')

def df_function():
    import pandas as pd
    import numpy as np
    from sklearn.externals import joblib
    ready_dataset_for_marking = pd.read_excel('ready_dataset_for_marking_last.xlsx')

    scaler = joblib.load('scaler_best.pkl')
    X_real_test = scaler.transform(ready_dataset_for_marking)

    xgb_clf_from_disk = joblib.load('xgb_clf_best.pkl')
    forest_clf_from_disk = joblib.load('forest_clf_best.pkl')
    gboost_clf_from_disk = joblib.load('gboost_clf_best.pkl')
    bernoulli_clf_from_disk = joblib.load('bernoulli_clf_best.pkl')
    xgb_clf_for_df_predictions_from_disk = joblib.load('xgb_clf_for_df_predictions_best.pkl')

    models = [xgb_clf_from_disk, forest_clf_from_disk, gboost_clf_from_disk, bernoulli_clf_from_disk]
    dict_predict = {}
    for model in models:
        y_test_pred = model.predict_proba(X_real_test)
        dict_predict[str(model)[:9]+' proba_0'] = np.array([i[0] for i in y_test_pred])
        dict_predict[str(model)[:9]+' proba_1'] = np.array([i[1] for i in y_test_pred])
        dict_predict[str(model)[:9]+' proba_2'] = np.array([i[2] for i in y_test_pred])
    df_predictions = pd.DataFrame(dict_predict, index=ready_dataset_for_marking.index)

    y_test_pred = xgb_clf_for_df_predictions_from_disk.predict(df_predictions)
    y_test_pred_proba = xgb_clf_for_df_predictions_from_disk.predict_proba(df_predictions)

    df=pd.DataFrame({'predictions':y_test_pred,'predictions_proba_for_2':[i[2] for i in y_test_pred_proba]}, index=ready_dataset_for_marking.index)
    #df[df.predictions==2]
    return df
