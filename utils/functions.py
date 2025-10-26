import numpy as np
import pandas as pd #для анализа и предобработки данных

from utils.reader_config import config_reader 
from models.models_collection import ModelRandomForest


from sklearn import preprocessing #предобработка
from sklearn.model_selection import train_test_split #сплитование выборки
from sklearn import metrics #метрики


# Импортируем константы из файла config
config = config_reader('../config/config.json')

def education_level(arg):
    """
    Function takes first three words of the argument and sort it among 4 cathegories: 'higher', 'higher_unfinished', 'secondary_professional' and 'secondary'
    """
    arg.lower()
    arg.split(' ', 2)[:3]
    if 'Высшее' in arg:
        return 'higher'
    elif 'Неоконченное высшее' in arg:
        return 'higher_unfinished'
    elif 'Среднее специальное' in arg:
        return 'secondary_professional'
    elif 'Среднее образование' in arg:
        return 'secondary'
    else:
        pd.NA

def get_gender(arg:pd.Series)->str:

    """Function get_gender takes string whch includes user's gender as the argument. 
    Arguments:
        _pd.Series_: value from a row of a DataFrame
    Returns:
        _str_: returnes user's gender either 'm' or 'f'.
    """    
  
    arg.split(' ') # Method split() divides the sring by space

    # If the string contains 'Мужчина', it returns letter 'M' else 'F'
    return 'M' if 'Мужчина' in arg else 'F'

def get_age(arg):
    """
    Function get_age takes third element of the argument and returns int number of user's age.
    """
    # Method split() divides the sring by space, and method int() transform the number into an integer 
    return int(arg.split(' ')[3]) 

def get_experience(arg):
    """
    The get_experience function takes int value of monts and/or years of user experience as the argument. It returns int value in monts.
    """
    year_description = ['лет', 'года', 'год']
    month_description = ['месяц','месяца', 'месяцев']
    months = 0
    
    # If user experience value is absent, return np.NaN
    if arg == 'Не указано' or arg is None or arg is np.NaN:
        return np.NaN
      
    else:
        arg_splitted = arg.split(' ')[:6] # Method divides the string by spaces
        # If the string contains years features, it will be recalculated in months
        if arg_splitted[3] in year_description:

            # If the string contains months features, its value will be represented as an int number
            if arg_splitted[5] in month_description:
                months = int(arg_splitted[2])*12 + int(arg_splitted[4])

            # If the string does not contains months features, its value will be calculated from years value
            else:
                months = int(arg_splitted[2])*12       
        
        # If the string does not contains year features, its value is just int months number 
        else:
            months = int(arg_splitted[2])
        return months
        
def get_city(arg):
    """
   The get_city function return 1 of 4 cathegories of city: 1)Moscow 2)Petersburg 3)Megapolis 4)other. 
   The function argument is a first word of the feature 'Город, переезд, командировки'.
    """
    arg = arg.split(' ') # Method divides the string by spaces

    # The list of megapolis cities
    million_cities = ['Новосибирск', 'Екатеринбург','Нижний Новгород','Казань', \
        'Челябинск','Омск', 'Самара', 'Ростов-на-Дону', 'Уфа', 'Красноярск', \
        'Пермь', 'Воронеж','Волгоград']

    if arg[0] == 'Москва':
        return 'Moscow'
    elif arg[0] == 'Санкт-Петербург':
        return 'Petersburg'
    elif arg[0] in million_cities:
        return 'megapolis'
    else:
        return 'other'


# Definition of relocation willingness
def get_relocation(arg):
    """
    The get_relocation function returns relocation willingness (True/False). The function argument is the feature string 'Город, переезд, командировки'.
    """
    if ('не готов к переезду' in arg) or ('не готова к переезду' in arg): # not ready
        return False 
    elif 'хочу переехать' in arg: # ready
        return True
    else: return True
            
def get_bisiness_trips(arg):
    """
    The get_bisiness_trips function retuens business trip readiness (True/False). The function argument is the feature string 'Город, переезд, командировки'.
    """
    if ('командировка' in arg):
        if ('не готов к командировкам' in arg) or('не готова к командировкам' in arg): # not ready
            return False
        else: 
            
            return True
    else:               # ready
        return False
    
    
# Definition of currency
def get_currency_in_ISO_format(arg):
    arg_splitted = arg.split(' ')[1].replace('.',"") #
    
    #  international currecy codes
    currencies_in_ISO_dict = {
        'грн':'UAH', 
        'USD':'USD', 
        'EUR':'EUR', 
        'белруб':'BYN', 
        'KGS':'KGS', 
        'сум':'UZS', 
        'AZN':'AZN', 
        'KZT':'KZT'
    }
    
    if arg_splitted == 'руб':
        return 'RUB'
    else:
        return currencies_in_ISO_dict[arg_splitted]
    
    
def get_aggregated_status(arg1, arg2):
    """
    The get_aggregated_status function group user by wilingness for business trips and by relocation.
    """
    if arg1 is True and arg2 is True:
        return 'ready_for_relocation_and_business_trips'
    elif arg1 is False and arg2 is True:
        return 'ready_for_relocation' 
    elif arg1 is True and arg2 is False:
        return 'ready_for_business_trips'
    else:
        return 'not_ready'

def outliers_z_score_mod(data, feature, left=3, right=3, log_scale=False):
    """
    The outliers_z_score_mod function filters values from outliers using z-method. Input:
    - DataFrame;
    - feature where we are looking for outliers. 
    - arguments 'left' and 'right' - are sigma multipliers; both are equal to 3 by default;
    - when the argument log_scale is True, it scales values to logarithmic.
    """
    if log_scale:
        x = np.log(data[feature])
    else:
        x = data[feature]
    mu = x.mean()
    sigma = x.std()
    lower_bound = mu - left * sigma
    upper_bound = mu + right * sigma
    outliers = data[(x < lower_bound) | (x > upper_bound)]
    cleaned = data[(x > lower_bound) & (x < upper_bound)]
    return outliers, cleaned

def get_profession(arg:str)->str:
    """Function for unifying of profession titles
    """
    arg = arg.lower().replace("-"," ").replace("веб","web") #.split(" ") 
    
    #programmer = ['программист', 'frontend', 'web', 'разработчик']
    
    if 'программист' in arg or 'разработчик' in arg : #or arg==('frontend разработчик' or 'веб разработчик')
        return "programmer"
    
    elif 'дизайнер' in arg:
        return 'designer'
    
    elif 'aналитик' in arg or ('системный' and 'аналитик') in arg:
        return 'analyst'
    
    elif ('главный' and 'специалист') in arg:
        return 'leading specialist'
    
    elif 'продавец' in arg:
        return 'salesman'
    
    elif ('системный' and 'администратор') in arg:
        return 'sys admin'
    
    elif 'менеджер по продажам' in arg:
        return 'sales manager'
    
    elif 'ведущий инженер' in arg:
        return 'leading engineer'
    
    elif ('руководитель' or 'менеджер' and 'проекта' or 'проектов') in arg:
        return 'project manager'
    
    elif ('начальник' or 'руководитель' or 'заведующий') in arg:
        return 'unit head'
    
    elif ('менеджер' or ('заместитель' and 'руководителя'))in arg:
        return 'manager'
    
    elif 'директор' in arg:
        return 'director'
    
    elif 'инженер' in arg:
        return 'engineer'
    
    elif 'маркетолог' in arg:
        return 'marketing specialist'
   
    elif ('техник' or 'монтажник') in arg or arg=='монтажник':
        return 'technicien'
    
    elif ('администратор' or 'administrator' or 'reception' or '') in arg:
        return 'administrator'
    
    else:
        return 'other' #arg 
    
    
    
    
# Define metrics
def print_metrics(y_train, y_train_predict, y_test, y_test_predict):
    print('Train R^2: {:.3f}'.format(metrics.r2_score(y_train, y_train_predict)))
    print('Train MAE: {:.0f} rub.'.format(metrics.mean_absolute_error(y_train, y_train_predict)))
    print('Train MAPE: {:.0f} %'.format(metrics.mean_absolute_percentage_error(y_train, y_train_predict)*100))
    print('\n')
    print('Test R^2: {:.3f}'.format(metrics.r2_score(y_test, y_test_predict)))
    print('Test MAE: {:.0f} rub.'.format(metrics.mean_absolute_error(y_test, y_test_predict)))
    print('Train MAPE: {:.0f} %'.format(metrics.mean_absolute_percentage_error(y_test, y_test_predict)*100))
    
    
    
    
      
def get_result():
    
    # get data
    data = pd.read_csv('data/dst-3.0_16_1_hh_database.zip', sep=';')
    
    # Apply the function to the DataFrame
    data['Education'] = data['Образование и ВУЗ'].apply(education_level)
    
    # delete the feature 'Образование и ВУЗ'
    data.drop(['Образование и ВУЗ'], axis=1, inplace=True)
    
    # Creation of a new feature 'Gender' 
    data['Gender'] = data['Пол, возраст'].apply(get_gender)

    # Creation of a new feature 'Age'
    data['Age'] = data['Пол, возраст'].apply(get_age)
    
    # Delete original feature 
    data.drop(['Пол, возраст'], axis=1, inplace=True)
    
    data["User_experience(months)"] = data['Опыт работы'].apply(get_experience)

    # Deleting the original feature 'Опыт работы'
    data.drop(['Опыт работы'], axis=1, inplace=True)
    
    # City cathegorization using the function 'get_city'
    data['City'] = data['Город, переезд, командировки'].apply(get_city)
    
    # relocation willingness using the function 'get_relocation'    
    data['Relocation'] = data['Город, переезд, командировки'].apply(get_relocation)
    
    data['Business_trip'] = data['Город, переезд, командировки'].apply(get_bisiness_trips)
    
    # Delete the original feature 'Город, переезд, командировки'
    data.drop(['Город, переезд, командировки'], axis=1, inplace=True)

    
    # Features for employment type: full time, part time, project,  volunteering, internship.
    employment_types = ['полная занятость', 'частичная занятость', 'проектная работа', 'волонтерство', 'стажировка']

    for i in employment_types:
        data[i] = data['Занятость'].apply(lambda x: True if i in x else False)

    data = data.rename(columns={
        'полная занятость':'full_time', 
        'частичная занятость':'part_time',
        'проектная работа':'project',
        'стажировка':'internship',
        'волонтерство':'volunteering'
    })
    
    # Features for schedule types: full_time, flexible, remote, daily_shifts, long_shifts.
    schedule =  ['полный день', 'гибкий график', 'удаленная работа', 'сменный график', 'вахтовый метод']

    # Features for schedule
    for j in schedule:
        data[j] = data['График'].apply(lambda x: True if j in x else False)

    data = data.rename(columns={
        'полный день':'full_time',     
        'гибкий график': 'flexible',
        'удаленная работа':'remote',   
        'сменный график': 'daily_shifts',
        'вахтовый метод':'long_shifts' 
    })
    
    # Delete original features
    data.drop(['Занятость', 'График' ], axis=1, inplace=True)
    
    # Change feature «Обновление резюме» format to datetime.
    data['date'] = pd.to_datetime(data['Обновление резюме'], dayfirst=False).dt.date

    # Creation of a new features: 'currency' and 'salary_national' (for salary expectations in national currencies) 
    data['currency'] =  data['ЗП'].apply(get_currency_in_ISO_format)
    data['salary_national'] = data['ЗП'].apply(lambda x: x.split(' ')[0]).astype('int64')
    
    # Reading the currency base
    Exchange_Rates = pd.read_csv('data/ExchangeRates.zip')
    # change date format to datetime 
    Exchange_Rates['date'] = pd.to_datetime(Exchange_Rates['date'], dayfirst=True).dt.date

    # merging of database 'date' and columns of 'Exchange_Rates':'currency', 'date', 'close', 'proportion'
    data_merged = data.merge(
        Exchange_Rates[['currency', 'date', 'close', 'proportion']], 
        left_on=['date','currency'], 
        right_on=['date','currency'], 
        how='left'
    )

    # Filling ruble to ruble rate as 1.
    data_merged['close'] = data_merged['close'].fillna(1)
    data_merged['proportion'] = data_merged['proportion'].fillna(1)

    # Calculation of salary in rubles
    data_merged['salary(rub)'] = data_merged['salary_national'] * data_merged['close'] / data_merged['proportion']

    # Delete original features 'ЗП_сумма', 'ЗП', 'currency', 'Обновление резюме', 'close', 'proportion'
    data_merged.drop(['salary_national', 'ЗП', 'currency', 'Обновление резюме', 'close', 'proportion'], axis=1, inplace=True)
    
    data_merged['Relocation_and_business_trip_status'] = data_merged[['Relocation','Business_trip']].apply(lambda x: get_aggregated_status(*x), axis=1)
    
    #--------------Data cleaning--------
    data = data_merged
    
    print(f'Inintal number of entries: {data.shape[0]}')
    dupl_columns = list(data.columns)
    
    mask_duplicated = data.duplicated(subset=dupl_columns)
    print(f'Number of repeating lines: {data[mask_duplicated].shape[0]}')

    # Delete repeating lines
    data_deduplicated = data_merged.drop_duplicates(subset=dupl_columns)
    print(f'Number of rows without duplicates: {data_deduplicated.shape[0]}')
    
    
    # Estimation of missing values
    cols_null_sum = data_deduplicated.isnull().sum()
    cols_with_null = cols_null_sum[cols_null_sum > 0].sort_values(ascending=False)
    
    # Remove rows in features 'Последнее/нынешнее место работы' and 'Последняя/нынешняя должность'.
    data_deduplicated = data_deduplicated.dropna(subset=['Последнее/нынешнее место работы','Последняя/нынешняя должность'], how='any', axis=0)

    # The dictionnary to fill missing values
    values = {
        'User_experience(months)': data_deduplicated['User_experience(months)'].median()
    }
    #Fill the missing values
    data_deduplicated = data_deduplicated.fillna(values)
        
        
    # Filtering salaries lower than 1000 and higher than 1 million rubles
    mask_salary_filter = (data_deduplicated['salary(rub)'] > 1e6) | (data_deduplicated['salary(rub)'] < 1e3)

    # Outliers
    print(f"Number of outliers: {data_deduplicated[mask_salary_filter].shape[0]}")

    # Filter entries using "mask_salary_filter"
    data_deduplicated.drop(data_deduplicated[mask_salary_filter].index, axis=0, inplace=True)
    
    # Filtering entries where user experience exceed user age
    mask_experience_equal_to_age = (data_deduplicated['User_experience(months)'] / 12 > data_deduplicated['Age']) & (data_deduplicated['Age'] / 12 < data_deduplicated['User_experience(months)'])

    # Outliers that stands outside of the boundary
    print(f"Outliers: {data_deduplicated[mask_experience_equal_to_age].shape[0]}")

    # Filtering outliers by "mask_experience_equal_to_age"
    data_deduplicated.drop(data_deduplicated[mask_experience_equal_to_age].index, axis=0, inplace=True)
    
    
    #remove candidates whose age exceed the range [-3*sigma; +4*sigma]
    log_data = data_deduplicated['Age'] #normal scale

    left = 3
    right = 4
    lower_bound = log_data.mean() - left * log_data.std()
    upper_bound = log_data.mean() + right * log_data.std()

    outliers, cleaned = outliers_z_score_mod(data_deduplicated, 'Age', left=3, right=4, log_scale=False)

    # Delete outliers
    data_deduplicated.drop(outliers.index, axis='index', inplace=True)


    # Data encoding----------------------------------

    # Let us split users by possession of auto feature: set '1' who does have an auto and '0' for those who does not
    data_deduplicated['auto'] = data_deduplicated['Авто'].apply(lambda x: 1 if x.find('Имеется собственный автомобиль')>=0 else 0)  

    # let us drop original feature
    data_deduplicated.drop(['Авто'], axis=1, inplace=True)


    # Let us identify most frequent user positions and delete minor deviations in titles
    data_deduplicated['position'] = data_deduplicated['Последняя/нынешняя должность'].apply(get_profession)

    #delete original feature
    data_deduplicated.drop(['Последняя/нынешняя должность', 'Последнее/нынешнее место работы','Ищет работу на должность:','date'], axis=1, inplace=True)

    #delete original features without preprocessing
    #data_deduplicated.drop(['Последнее/нынешнее место работы','Ищет работу на должность:','date'], axis=1, inplace=True)


    # Encoding-----
    data_encoded = pd.get_dummies(data_deduplicated, columns=['Education', 'Gender', 'City',  'Relocation_and_business_trip_status','position',
    'Business_trip', 'full_time', 'part_time', 'project', 'volunteering', 'internship', 'full_time', 'flexible', 'remote', 'daily_shifts', 'long_shifts', 'Relocation', 
    ])
    
    # Initiate the RobustScaler()
    r_scaler = preprocessing.RobustScaler()

    # copy original dataset
    df_r = r_scaler.fit_transform(data_encoded[['Age', 'User_experience(months)']]) #'salary(rub)'

    #Transform the features for visualization
    df_r = pd.DataFrame(df_r, columns=['Age_n', 'User_experience(months)_n']) #'salary(rub)'

    # Add transformed features to the Dataframe
    data_encoded = data_encoded.join(df_r, how='left') #on='mukey'

    # Delere original features without normalization
    data_encoded.drop(['Age', 'User_experience(months)'], axis=1, inplace=True)
    
    
    # Delete rows with missed data
    data_encoded = data_encoded.dropna(how='any', axis=0)

    # copy dataframe
    data_prepared = data_encoded.copy()
    
    
    #--------Models---------
    
    # Create two matrixes: features and target
    X, y = data_prepared.drop('salary(rub)', axis=1, ), data_prepared['salary(rub)']

    # Split the data in a ratio 80/20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=config.random_seed)

    # check the shape
    print(f'X_Train: {X_train.shape} y_train: {y_train.shape}')
    print(f' X_Test: {X_test.shape},  y_test: {y_test.shape}')
    
    
    # # Get a logarythm of train data
    # y_train_log = np.log(y_train)


    # # Creation of an isntance of the linear model class wtith the L2-regularization with the best alpha-coefficient
    # ridge_lr = linear_model.Ridge(alpha=config.alpha)
    # # Train the model to predict log target values
    # ridge_lr.fit(X_train_scaled_poly, y_train_log)
    # #Make a prediction for train and test samples and get expanential data
    # y_train_pred = np.exp(ridge_lr.predict(X_train_scaled_poly))
    # y_test_pred = np.exp(ridge_lr.predict(X_test_scaled_poly))
    # # Calculate metrics
    # print_metrics(y_train, y_train_pred, y_test, y_test_pred)