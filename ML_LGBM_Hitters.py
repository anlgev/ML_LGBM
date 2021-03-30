####################################
# Data Information

# This dataset was originally taken from the StatLib library which is maintained at Carnegie Mellon University.
# This is part of the data that was used in the 1988 ASA Graphics Section Poster Session.
# The salary data were originally from Sports Illustrated, April 20, 1987.
# The 1986 and career statistics were obtained from The 1987 Baseball Encyclopedia Update published by Collier Books,
#  Macmillan Publishing Company, New York.

# Format
# A data frame with 322 observations of major league players on the following 20 variables.
# AtBat Number of times at bat in 1986
# Hits Number of hits in 1986
# HmRun Number of home runs in 1986
# Runs Number of runs in 1986
# RBI Number of runs batted in in 1986
# Walks Number of walks in 1986
# Years Number of years in the major leagues
# CAtBat Number of times at bat during his career
# CHits Number of hits during his career
# CHmRun Number of home runs during his career
# CRuns Number of runs during his career
# CRBI Number of runs batted in during his career
# CWalks Number of walks during his career
# League A factor with levels A and N indicating player’s league at the end of 1986
# Division A factor with levels E and W indicating player’s division at the end of 1986
# PutOuts Number of put outs in 1986
# Assists Number of assists in 1986
# Errors Number of errors in 1986
# Salary 1987 annual salary on opening day in thousands of dollars
# NewLeague A factor with levels A and N indicating player’s league at the beginning of 1987


import warnings
import pandas as pd
# !pip install catboost
from catboost import CatBoostRegressor
# !pip install lightgbm
# conda install -c conda-forge lightgbm
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
# !pip install xgboost
from xgboost import XGBRegressor
from sklearn.neighbors import LocalOutlierFactor

from helpers.data_prep import *
from helpers.eda import *

pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


def load_hit():
    data = pd.read_csv("dataset/hitters.csv")
    return data

hit = load_hit()
hit.head()

############################
# EDA
############################


check_data(hit)

cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(hit)

for col in num_cols:
    sns.displot(hit[col])
    plt.show()

for col in ['Years', 'Errors','Assists', 'PutOuts', 'CWalks',
            'CRBI', 'CRuns', 'CHmRun', 'CHits', 'CAtBat', 'HmRun']:
    hit[col] = np.log(hit[col])

from sklearn.preprocessing import RobustScaler

for col in num_cols:
    transformer = RobustScaler().fit(hit[[col]])
    hit[col] = transformer.transform(hit[[col]])

def hitters_prep(dataframe):
    # OUTLIER
    cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(dataframe)
    for col in num_cols:
        if check_outlier(dataframe, col, q1=0.10, q3=0.90):
            replace_with_thresholds(dataframe, col, q1=0.10, q3=0.90)

        # MISSINIG VALUES

    # if dataframe.isnull().sum().any():
        # dataframe['Salary'] = dataframe['Salary'].\
          #  fillna(dataframe.groupby(['League', 'Division', 'NewLeague'])['Salary'].transform("mean"))

    dataframe.dropna(inplace=True)


    num_df = dataframe[num_cols]

    clf = LocalOutlierFactor(n_neighbors=20)
    clf.fit_predict(num_df)
    df_scores = clf.negative_outlier_factor_
    esik_deger = np.sort(df_scores)[4]
    dataframe.drop(axis=0, labels=dataframe[df_scores < esik_deger].index, inplace=True)

    for col in ['Years', 'Errors', 'Assists', 'PutOuts', 'CWalks',
                'CRBI', 'CRuns', 'CHmRun', 'CHits', 'CAtBat', 'HmRun']:
        dataframe[col] = np.log(dataframe[col])

    # FEATURE ENGINEERING
    dataframe.columns = [col.upper() for col in dataframe.columns]

    dataframe['NEW_ATBAT_RATIO'] = dataframe['ATBAT'] / dataframe['CATBAT']
    dataframe['NEW_HITS_RATIO'] = dataframe['HITS'] / dataframe['CHITS']
    dataframe['NEW_HMRUN_RATIO'] = dataframe['HMRUN'] / dataframe['CHMRUN']
    dataframe['NEW_RUNS_RATIO'] = dataframe['RUNS'] / dataframe['CRUNS']
    dataframe['NEW_RBI_RATIO'] = dataframe['RBI'] / dataframe['CRBI']
    dataframe['NEW_WALKS_RATIO'] = dataframe['WALKS'] / dataframe['CWALKS']

    dataframe['NEW_CATBAT_Y'] = dataframe['CATBAT'] / dataframe['YEARS']
    dataframe['NEW_HITS_Y'] = dataframe['CHITS'] / dataframe['YEARS']
    dataframe['NEW_HMRUN_Y'] = dataframe['CHMRUN'] / dataframe['YEARS']
    dataframe['NEW_CRUNS_Y'] = dataframe['CRUNS'] / dataframe['YEARS']
    dataframe['NEW_CRBI_Y'] = dataframe['CRBI'] / dataframe['YEARS']
    dataframe['NEW_CWALKS_Y'] = dataframe['CWALKS'] / dataframe['YEARS']

    dataframe['NEW_ATBAT_ERR'] = dataframe['ATBAT'] / dataframe['ERRORS']
    dataframe['NEW_HITS_ERR'] = dataframe['HITS'] / dataframe['ERRORS']

    dataframe['NEW_SUCCCESS_LAST'] = ((dataframe['ATBAT'] + dataframe['HITS'] + dataframe['HMRUN']
                                      + dataframe['RUNS'] + dataframe['RBI'] + dataframe['WALKS']
                                      + dataframe['PUTOUTS'] + dataframe['ASSISTS'] - dataframe['ERRORS'])
                                      / dataframe['YEARS'])

    dataframe.fillna(0, inplace=True)

    dataframe['NEW_ATBAT_HITS_R'] = dataframe['ATBAT'].div(dataframe['HITS']).replace(np.inf, 0)

    dataframe['NEW_YEARS'] = pd.qcut(dataframe['YEARS'], 3, labels=['rookie', 'senior', 'proof'])

    # ENCODING
    binary_cols = [col for col in dataframe.columns if dataframe[col].nunique() == 2 and dataframe[col].dtypes == 'O']
    for col in binary_cols:
        label_encoder(dataframe, col)

    ohe_cols = [col for col in dataframe.columns if 10 >= dataframe[col].nunique() > 2]

    dataframe = one_hot_encoder(dataframe, ohe_cols)

    return dataframe


hit = hitters_prep(hit)
y = hit["SALARY"]
X = hit.drop(["SALARY"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


#######################################
# LightGBM: Model & Tahmin
#######################################

lgb_model = LGBMRegressor().fit(X_train, y_train)
y_pred = lgb_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))


#######################################
# Model Tuning
#######################################

lgb_model = LGBMRegressor()

lgbm_params = {"learning_rate": [0.001, 0.01, 0.05, 0.03],
               "n_estimators": [200, 500],
               "max_depth": [3, 5, 7],
               "colsample_bytree": [1, 0.8, 0.5, 0.2, 0.1]}

lgbm_cv_model = GridSearchCV(lgb_model,
                             lgbm_params,
                             cv=10,
                             n_jobs=-1,
                             verbose=2).fit(X_train, y_train)

lgbm_cv_model.best_params_

#######################################
# Final Model
#######################################

lgbm_tuned = LGBMRegressor(**lgbm_cv_model.best_params_).fit(X_train, y_train)
y_pred = lgbm_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(lgbm_tuned, X_train)

# correlation graph
fig, ax = plt.subplots(1, 1, figsize=(20, 15))
vmax = 0.5
vmin = -0.5
sns.heatmap(hit.corr(), annot=True, fmt='.2f', cmap='Spectral', mask=np.triu(hit.corr()), vmax=vmax, vmin=vmin,
            ax=ax)
plt.tight_layout()
plt.show()