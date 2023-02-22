# Data Cleaning and Preprocessing -

import os
import pandas as pd
import numpy as np
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

#import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

train_data = pd.read_csv('/Users/vidhiwaghela/Desktop/Yield-Prediction-main/train_data.csv', nrows=300)
train_weather = pd.read_csv('/Users/vidhiwaghela/Desktop/Yield-Prediction-main/train_weather-1646897968670.csv', nrows=300)
farm_data = pd.read_csv('/Users/vidhiwaghela/Desktop/Yield-Prediction-main/farm_data-1646897931981.csv', nrows=300)

train_data.dtypes

train_data.nunique()

train_data.head()

train_data.tail()

farm_data.head()

farm_data.dtypes

train_weather.head()

train_weather.dtypes

# sns.set_style("whitegrid")
# sns.set_context("poster")

# plt.figure(figsize = (12, 6))
# plt.hist(train_data['yield'])
# plt.title('Histogram of Yield Prediction in the training set')
# plt.xlabel('Count')
# plt.ylabel('Target value')
# plt.show()

"""### Data Cleansing

Train Data -
"""

#Convert date Object to datetime format
train_data['date'] = pd.to_datetime(train_data['date'], format='%Y-%m-%d %H:%M:%S')

#Type Conversion
train_data['farm_id'] = train_data['farm_id'].astype('str')

#Check for duplicates
train_data.duplicated(subset=None, keep= 'first').sum()

#Drop duplicates
train_data1 = train_data.drop_duplicates(subset=None, keep= 'first', inplace=False, ignore_index=False)

"""Farm Data - """

#Drop 'operations_commencing_year' column from the farm data
farm_data.drop('operations_commencing_year', axis=1, inplace=True)

#Type Casting
farm_data['farm_id'] = farm_data['farm_id'].astype('str')

#Check for Null Values
farm_data.isnull().sum()

#Fill missing values with median 
farm_data['num_processing_plants'].fillna(farm_data['num_processing_plants'].median(), inplace=True)

#Type Conversion
farm_data['num_processing_plants'] = farm_data['num_processing_plants'].astype('int64')

"""Weather Data -"""

#Convert date Object to datetime format
train_weather['timestamp'] = pd.to_datetime(train_weather['timestamp'], format='%Y-%m-%d %H:%M:%S')

#Dropping 'cloudiness' column
train_weather.drop('cloudiness', axis = 1, inplace =  True)

"""### Merging the Datasets"""

#Merging train_data and farm_data on 'farm_id' column
train_data_merged = pd.merge(train_data1,farm_data, on = 'farm_id')

train_data_merged.head()

#Renaming date column as timestamp
train_data_merged.rename(columns = {'date':'timestamp'},inplace = True)

#Check for Value counts
train_weather.deidentified_location.value_counts()

#Drop deidentified_location column from the merged dataset
train_data_merged.drop('deidentified_location', axis=1, inplace=True)

train_data_merged.shape

train_weather.shape

train_data_merged.isnull().sum()

train_data_merged.nunique()

train_data_merged.ingredient_type.value_counts(normalize = True)*100

train_data_merged.farming_company.value_counts(normalize = True)*100

train_weather.nunique()

train_weather.head()

train_weather.dtypes

#Drop 'timestamp' from weather data
train_weather.drop('timestamp', axis=1, inplace=True)

train_weather.shape

train_data_merged.shape

#Final Merged Dataset 
train_final_merged = pd.merge(train_data_merged,train_weather,left_index = True, right_index = True)

train_final_merged.nunique()

train_final_merged.ingredient_type.value_counts(normalize = True)*100

train_final_merged.farming_company.value_counts(normalize = True)*100

train_final_merged.deidentified_location.value_counts(normalize = True)*100

train_final_merged.shape

train_final_merged.isnull().sum()

train_final_merged.head()

train_final_merged.tail()

train_final_merged.drop('farm_id', axis=1, inplace=True)

train_final_merged.head()

train_final_merged.tail()

#Calculating Hour difference between the timestamps
train_final_merged['Hours'] = train_final_merged['timestamp'].dt.hour

#train_final_merged['Minutes'] = train_final_merged['timestamp'].dt.minute

#train_final_merged['Seconds'] = train_final_merged['timestamp'].dt.second

train_final_merged['timestamp'].min(), train_final_merged['timestamp'].max(), (train_final_merged['timestamp'].max() -train_final_merged['timestamp'].min())

"""Unix time is a system for describing a point in time. It is the number of seconds that have elapsed since the Unix epoch"""

#Converting Timestamp to integer
train_final_merged['Unix Sec'] = pd.to_datetime(train_final_merged['timestamp']).astype(int)/ 10**9

train_final_merged.head()

train_final_merged.dtypes

#Type Conversion float to int
train_final_merged['Unix Sec'] = train_final_merged['Unix Sec'].astype('int64')

train_final_merged.dtypes

train_final_merged.drop('timestamp', axis=1, inplace=True)

"""###  Practices in  Data Cleaning -

* I have cleaned the data by checking for duplicate values and dropped them from that trainset.

* Performed Type casting on the columns respectively wherever needed and also checked for misssing values in the data.

* Determined if all of the columns in test exist in train, and vice versa w.r.t target, 
Did some initial plotting .

* Merged all the three datasets into the final training set which is given to the model to learn and can predict on much larger data.

* To save complexity , removed/dropped few columns from the data .

### Label Encoding for Categorical Columns
"""

#Necessary Import
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

train_final_merged.farming_company = le.fit_transform(train_final_merged.farming_company)

train_final_merged['farming_company'] = train_final_merged['farming_company'].astype('category')

train_final_merged.deidentified_location = le.fit_transform(train_final_merged.deidentified_location)

train_final_merged['deidentified_location'] = train_final_merged['deidentified_location'].astype('category')

train_final_merged.ingredient_type = le.fit_transform(train_final_merged.ingredient_type)

train_final_merged['ingredient_type'] = train_final_merged['ingredient_type'].astype('category')

train_final_merged.dtypes

train_final_merged.nunique()

train_final_merged.farming_company.value_counts()

train_final_merged.head()

train_final_merged.deidentified_location.value_counts()

train_final_merged.ingredient_type.value_counts()

train_final_merged.nunique()

#Descriptive Stats of Final Dataset
train_final_merged.describe()

#Check for null values in the merged dataset
train_final_merged.isnull().sum()

train_final_merged.dtypes

train_final_merged.drop('Hours', axis=1, inplace=True)

train_final_merged.num_processing_plants.value_counts()

train_final_merged.head()

#Seperated the numeric columns from the final merged dataset
num_col = ['yield','farm_area','temp_obs', 'wind_direction', 'dew_temp', 'pressure_sea_level', 
       'precipitation', 'wind_speed','Unix Sec']

num_col

#Correlation Plot on the numeric data
plt.figure(figsize=(22,10))
data = train_final_merged

df = pd.DataFrame(data)

# corrMatrix = df.corr(method = 'pearson',min_periods = 1)
# sns.heatmap(corrMatrix, annot=True)
# plt.show()

"""### Imputing the Numeric Columns"""

from sklearn.impute import SimpleImputer

#Imputation
num_imputer = SimpleImputer(strategy = 'median')

imputed_data = pd.DataFrame(num_imputer.fit_transform(train_final_merged[num_col]),
                               columns = num_col)

imputed_data.dtypes

imputed_data.head()

imputed_data.tail()

#Type Conversions
col = ['temp_obs','dew_temp','wind_speed','wind_direction','Unix Sec','farm_area']

imputed_data[col] = imputed_data[col].astype('int64')

imputed_data.head()

imputed_data.tail()

imputed_data.dtypes

imputed_data.shape

imputed_data.insert(0, 'id', imputed_data.index)

imputed_data.head()

cat_col = ['ingredient_type','farming_company', 'deidentified_location']

Final_data = pd.merge(imputed_data,
                         train_final_merged[cat_col],
                         left_index=True,
                         right_index=True)

Final_data.head()

Final_data.dtypes

"""# Model Building and Predictions

###  Train Test Validation Split
"""

#Necessary Import
from sklearn.model_selection import train_test_split

y = Final_data['yield']
X = Final_data.loc[:, Final_data.columns != 'yield']

X_train, X_test, y_train, y_test =   train_test_split(X, y, test_size=0.20,random_state = 123)

print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

"""### Building a Base Model"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

"""* Because we are attempting to predict a value we will look at Regression methods. Because of the medium size of our data we will begin RandomForestRegressors,
and then move towards ensemble methods like boosted trees such as GradientBoostingRegressor.



* To give ourself a baseline our first submission will simply use all the data and no hyperparameter tuning.
"""

# First create the base model to tune
rf = RandomForestRegressor()

rf.fit(X_train, y_train)

train_pred1 = rf.predict(X_train)
test_pred1 = rf.predict(X_test)

train_pred1

test_pred1

print(rf.score(X_train, y_train))
print(rf.score(X_test, y_test))

#Defining Custom Function 
def convert_for_sklearn(label_list):
    return [1 if i == 1 else 0 for i in label_list]


def rmse_metric(y_true, y_pred):
    
    y_test_scoring = convert_for_sklearn(y_true)
    test_pred_scoring = convert_for_sklearn(y_pred)

    mse = mean_squared_error(y_true= y_test, y_pred = test_pred1)
    rmse = np.sqrt(mse)
    
    print("Test RMSE_score: ",rmse)

rmse_metric(y_true = y_test, y_pred = test_pred1)

"""### Hyper Parameter Tuning and Cross Validation

* Hyperparameter tuning on the all features data .
"""

from pprint import pprint
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(rf.get_params())

"""* Random search is a technique where random combinations of the hyperparameters are used to find the best solution for the built model. It is similar to grid search, and yet it has proven to yield better results comparatively."""

#import randomized search
from sklearn.model_selection import RandomizedSearchCV

#Use the random grid to search for best hyperparameters


# Number of trees in random forest
n_estimators = [100,150,200]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [5,10,15,20,25,30]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]
# Method of selecting samples for training each tree
# bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

pprint(random_grid)

# Random search of parameters, using 3 fold cross validation
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 5, 
                               cv = 3, verbose=2, random_state=42, n_jobs = 1)

rf_random.fit(X_train, y_train)

#rf_random.best_params_

"""### Hyperparameter Tuned Random Forest Regressor

* Best parameters set for the Random Forest Regression.
"""

RF_reg = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
          max_features='auto', max_leaf_nodes=None,      
                               min_samples_leaf=2, min_samples_split=2,
          min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
          oob_score=False, random_state=None, verbose=0, warm_start=False)

# Commented out IPython magic to ensure Python compatibility.
# %%time
RF_reg.fit(X_train,y_train)

train_pred = RF_reg.predict(X_train)
test_pred = RF_reg.predict(X_test)

train_pred

test_pred

print(RF_reg.score(X_train, y_train))
print(RF_reg.score(X_test, y_test))

#Defining Custom Function 
def convert_for_sklearn(label_list):
    return [1 if i == 1 else 0 for i in label_list]


def rmse_metric(y_true, y_pred):
    
    y_test_scoring = convert_for_sklearn(y_true)
    test_pred_scoring = convert_for_sklearn(y_pred)

    mse = mean_squared_error(y_true= y_test, y_pred = test_pred)
    rmse = np.sqrt(mse)
    
    print("Test RMSE_score: ",rmse)

rmse_metric(y_true = y_test, y_pred = test_pred)

"""# Predictions on the Test Data

Reading the Test Data Files
"""

# Reading the test data files
test_data = pd.read_csv('/Users/vidhiwaghela/Desktop/Yield-Prediction-main/test_data-1664552867678.csv', nrows=50)
test_weather = pd.read_csv('/Users/vidhiwaghela/Desktop/Yield-Prediction-main/test_weather-1646897984996-1664552604982.csv', nrows=50)

test_data.head()

test_weather.head()

test_data.shape

"""* Performed all the Data Cleaning practices on the Test datasets as well """

test_data['date'] = pd.to_datetime(test_data['date'], format='%Y-%m-%d %H:%M:%S')

test_data['farm_id'] = test_data['farm_id'].astype('str')

#test_data.duplicated(subset=None, keep= 'first').sum()

#test_data1 = test_data.drop_duplicates(subset=None, keep= 'first', inplace=False, ignore_index=False)

test_weather['timestamp'] = pd.to_datetime(test_weather['timestamp'], format='%Y-%m-%d %H:%M:%S')

test_weather.drop('cloudiness', axis = 1, inplace =  True)

test_data_merged = pd.merge(test_data,farm_data, on = 'farm_id')

test_data_merged.head()

test_data_merged.rename(columns = {'date':'timestamp'},inplace = True)

#test_data_merged.drop('timestamp', axis=1, inplace=True)

#test_data_merged.drop('deidentified_location', axis=1, inplace=True)

test_weather.drop('timestamp', axis=1, inplace=True)

test_weather.drop('deidentified_location', axis=1, inplace=True)

#test_data.head()

test_data_merged.shape

test_weather.head()

test_data_merged.tail()

test_data_merged.head()

#test_data_merged.nunique()

#test_weather.nunique()

test_weather.insert(0, 'id', test_weather.index)

test_weather.head()

test_data_merged.head()

test_data_merged.tail()

#test_weather.drop('deidentified_location', axis=1, inplace=True)

test_data_merged.shape

test_weather.shape

test_final_merge = pd.merge(test_data_merged,test_weather,on = 'id',how = 'left')

test_final_merge.shape

test_final_merge.head()

test_final_merge.drop('farm_id', axis=1, inplace=True)

test_final_merge['timestamp'].min(), test_final_merge['timestamp'].max(), (test_final_merge['timestamp'].max() -test_final_merge['timestamp'].min())

test_final_merge.isnull().sum()

test_final_merge['Unix Sec'] = pd.to_datetime(test_final_merge['timestamp']).astype(int)/ 10**9

test_final_merge['Unix Sec'] = test_final_merge['Unix Sec'].astype('int64')

test_final_merge['Hours'] = test_final_merge['timestamp'].dt.hour

test_final_merge.drop('timestamp', axis=1, inplace=True)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

test_final_merge.farming_company = le.fit_transform(test_final_merge.farming_company)

test_final_merge['farming_company'] = test_final_merge['farming_company'].astype('category')

test_final_merge.deidentified_location = le.fit_transform(test_final_merge.deidentified_location)

test_final_merge['deidentified_location'] = test_final_merge['deidentified_location'].astype('category')

test_final_merge.ingredient_type = le.fit_transform(test_final_merge.ingredient_type)

test_final_merge['ingredient_type'] = test_final_merge['ingredient_type'].astype('category')

test_final_merge.drop('Hours', axis=1, inplace=True)

test_final_merge.tail()

Test_Data = test_final_merge

Test_Data.head()

Test_Data.nunique()

Test_Data.tail()

num_col = ['id','farm_area','temp_obs', 'wind_direction', 'dew_temp', 'pressure_sea_level', 
       'precipitation', 'wind_speed','Unix Sec']

from sklearn.impute import SimpleImputer

num_imputer = SimpleImputer(strategy = 'median')
#
imputed_data_test = pd.DataFrame(num_imputer.fit_transform(Test_Data[num_col]),
                               columns = num_col)

imputed_data_test.head()

col = ['id','temp_obs','dew_temp','wind_speed','wind_direction','Unix Sec','farm_area']

imputed_data_test[col] = imputed_data_test[col].astype('int64')

cat_col = ['ingredient_type','farming_company', 'deidentified_location']

Final_data_test = pd.merge(imputed_data_test, 
                         Test_Data[cat_col], 
                         left_index=True,
                         right_index=True)

Final_data_test.shape

#Final_data_test.duplicated(subset=None, keep= 'first').sum()

type(Test_Data)

#Test_Data.shape

#Final_data_test1 = Final_data_test.drop_duplicates(subset=None, keep= 'first', inplace=False, ignore_index=False)

test_pred = RF_reg.predict(Final_data_test)

type(test_pred)

test_pred

test_pred_1 = np.round(test_pred,3)

test_pred_1[0:10]

"""# Submissions"""

submission  = pd.DataFrame({'id': test_data.id, 'yield': test_pred_1})

submission.head()

submission.tail()

submission['yield'].plot()

#Converting the Dataframe to a csv file.

submission.to_csv('//Users/vidhiwaghela/Desktop/Yield-Prediction-main/submission.csv', index = None)

#Pickle dump
import pickle 
pickle.dump(RF_reg, open("model.pkl","wb"))