# importing needed libraries/modules
import os
from re import L
import pandas as pd
import numpy as np
import datetime
from math import sqrt

# importing visualization libraries 
import seaborn as sns
import matplotlib.pyplot as plt

# importing sql 
import env
from env import user, password, host, get_connection


# sklearn library for data science
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, RFE, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler


'''function that will either 
1. import the zillow dataset from MySQL or 
2. import from cached .csv file'''
def get_zillow_dataset():
    # importing "cached" dataset
    filename = "zillow_regression.csv"
    if os.path.isfile(filename):
        return pd.read_csv(filename, index_col=[0])

    # if not in local folder, let's import from MySQL and create a .csv file
    else:
        # query necessary to pull the 2017 properties table from MySQL
        query = '''
        SELECT *
            FROM properties_2017
                JOIN predictions_2017 USING (id)
                    JOIN propertylandusetype USING (propertylandusetypeid)
                        WHERE transactiondate = 2017
                            AND propertylandusedesc = "Single Family Residential"'''
        db_url = f'mysql+pymysql://{user}:{password}@{host}/zillow'
        # creating the zillow dataframe using Pandas' read_sql() function
        df = pd.read_sql(query, db_url)
        df.to_csv(filename)

        return df


'''Preparing/cleaning zillow dataset
focus is dropping Null values and changing column types'''
def clean_zillow_dataset(df):
    # cols needed for initial exploration & hypothesis testing
    df = df[[
    'taxvaluedollarcnt',
    'bathroomcnt',
    'bedroomcnt',
    'calculatedfinishedsquarefeet',
    'fips',
    'latitude',
    'longitude',
    'lotsizesquarefeet',
    'regionidcity',
    'regionidcounty',
    'regionidzip',
    'yearbuilt',
    'transactiondate']]

    # renaming cols
    df = df.rename(columns = {
    'taxvaluedollarcnt': "home_value",
    'bathroomcnt': "bathroom_count",
    'bedroomcnt': "bedroom_count",
    'calculatedfinishedsquarefeet': "living_sq_feet",
    'fips': "fips_code",
    'lotsizesquarefeet': "property_sq_feet",
    'parcelid': "property_id",
    'regionidcity': "city_id",
    'regionidcounty': "county_id",
    'regionidzip': "zip_code",
    'transactiondate': "purchase_date",
    'yearbuilt': "year_built",})

    # dropping remaining Nulls
    df = df.dropna()

    # converting the following cols to proper int type
    int_cols = ["bedroom_count", 
                "city_id", 
                "county_id", 
                "zip_code",
                "year_built"]
    df[int_cols] = df[int_cols].astype(int)

    # converting purchase date to datetime type
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])

    # lastly, return the cleaned dataset
    return df


# function for handling outliers in the dataset
def zillow_outliers(df):
    df = df[df["home_value"] <= 900_000]
    df = df[df["living_sq_feet"] <= 8_000]
    df = df[(df["bedroom_count"] > 0) & (df["bedroom_count"] <= 6)]
    df = df[(df["bathroom_count"] > 0) & (df["bathroom_count"] <= 6.5)]

    return df


'''Function for generating categorial/feature dummy columns'''
def clean_months(df):
    # renaming month-year column to months only
    year_and_month = df["purchase_month"].sort_values().unique().tolist()
    month_lst = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September']

    df["purchase_month"] = df["purchase_month"].replace(
        year_and_month,
        month_lst)

    return df 


# generating dummy variables
def get_dummy_cols(df):
    dummy_df = pd.get_dummies(df[[
                'fips_code', \
                'county_id', \
                'purchase_month']])

    return df


# dropping redundant columns
def drop_after_dummy(df):
    # dropping the following features/columns since they are being either 1. being bucketed or 2. have dummy variables created
    df = df.drop(columns = [
        'purchase_date',
        'year_built',
        'bedroom_count', 
        'bathroom_count'])
    
    return df

# function establishes a baseline for train and validate - will be used for model comparison:
def establish_baseline(train, validate):
    baseline = round(train["home_value"].mean(), 2)

    train['baseline'] = baseline
    validate['baseline'] = baseline

    train_rmse = sqrt(mean_squared_error(train.home_value, train.baseline))
    validate_rmse = sqrt(mean_squared_error(validate.home_value, validate.baseline))

    print('Train baseline RMSE: {:.2f}'.format(train_rmse))
    print('Validation baseline RMSE: {:.2f}'.format(validate_rmse))

    train = train.drop(columns = "baseline")
    validate = validate.drop(columns = "baseline")
    print()
    print(f'train shape: {train.shape}')
    print(f'validate shape: {validate.shape}')

    return train, validate

'''Function created to split the initial dataset into train, validate, and test datsets'''
def train_validate_test_split(df):
    train_and_validate, test = train_test_split(
    df, test_size = 0.2, random_state = 123)
    
    train, validate = train_test_split(
        train_and_validate,
        test_size = 0.3,
        random_state = 123)

    print(f'train shape: {train.shape}')
    print(f'validate shape: {validate.shape}')
    print(f'test shape: {test.shape}')

    return train, validate, test
    

def age_of_homes(df):
    # creating a column for age of the home
    year_built = df["year_built"]
    curr_year = datetime.datetime.now().year

    # placing column/series back into main df
    df["home_age"] = (curr_year - year_built).astype("int")

    return df

# function to create and plot a new column/home purchase by 2017 month
def get_months_and_plot(df):
    # creating a lambda function to isolate the dates by month and year
    df['purchase_month'] = df['purchase_date'].map(lambda dt: dt.strftime('%Y-%m'))

    grouped_df = df.sort_values("purchase_month").groupby('purchase_month').size().to_frame("count").reset_index()

    sns.set(font_scale = .5, style = "darkgrid")
    ax = sns.countplot(x = "purchase_month",
                    data = df,
                    order = grouped_df["purchase_month"],
                    palette = "crest")

    ax.bar_label(ax.containers[0])

    plt.xlabel(None)
    plt.title("2017 Home Purchases by Month")
    plt.show()

    return df

# function returns True if bathroom contains half bath as denoted by ".5"
def get_half_baths(df):
    df["half_bathroom"] = df["bathroom_count"].astype("str").str.contains(".5").astype(bool)
    
    return df

# function bins bathrooms and bedrooms
def bin_bath_and_beds(df):
    # binning bathrooms
    df["1_to_3.5_baths"] = df["bathroom_count"] <= 3.5
    df["4_to_6.5_baths"] = (df["bathroom_count"] > 3.5) | (df["bathroom_count"] <= 6.5)

    # binning bedrooms
    df["1_to_2_bedrooms"] = df["bedroom_count"] <= 2
    df["3_to_4_bedrooms"] = (df["bedroom_count"] > 2) | (df["bedroom_count"] <= 4)
    df["5_to_6_bedrooms"] = (df["bedroom_count"] > 4) | (df["bedroom_count"] <= 6)

    return df


'''Function takes in a dataset and cleans/drops unneeded cols for modeling'''
def clean_for_features(df):
    df = df.drop(columns = [
        '4_to_6.5_baths',
        '3_to_4_bedrooms',
        '5_to_6_bedrooms',
        'purchase_month_February', 
        'purchase_month_June',
        "city_id", 
        "property_sq_feet",
        "longitude",
        "zip_code"])

    return df


'''Function takes in a df and scales continuous cols'''
def scaled_data(df): 
    # creating a copy of the original zillow/dataframe
    scaled_cols = [ 
        'living_sq_feet',
        'latitude',
        'home_age']

    scaler = RobustScaler()

    # fitting and transforming cols to needed scaled values
    df[scaled_cols] = scaler.fit_transform(df[scaled_cols])

    # returning newly created dataframe with scaled data
    return df

def generate_df(model, x_val, feature_lst, y_val):
    
    X_var = pd.DataFrame(x_val[feature_lst])
    y_var = pd.DataFrame(y_val)

    # concatenating the two tables
    df = pd.concat([X_var, y_var], axis = 1)

    # generating validation data baseline predictions
    baseline_mean_predictions = round(y_val.mean(), 2)
    df["baseline_mean_predictions"] = baseline_mean_predictions

    # generating model predictions
    lr_predictions = model.predict(x_val)
    df["linear_predictions"] = lr_predictions.round(2)

    return df


'''Function to retrieve final validate RMSE report'''
def get_validate_report():
    report = {
    'Models': ['Baseline Predictions', 'Linear Regression Predictions', 'Laso Lars Predictions'],
    'Explained Sum of Squares (ESS)': [0.3, 121233107350265.9, 116879751553174.2],
    'Mean Sum Error (MSE)': [41914677866.4, 32010289438.0, 32028755060.62],
    'Root Mean Squared Error (RMSE)': [204730.7, 178914.2, 178965.79]}

    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    final_validate_report = pd.DataFrame(report)
    
    return final_validate_report


# function retrieves final readout on test dataset
def final_rmse():
    final_rmse = pd.DataFrame({
    "Test": ["Baseline", "Train", "Validate", "Final"],
    "RMSE": [204730.70,178296.33,178914.20,177111.14],
    "Relative Diff.": [0, .15, 0, .01]})

    return final_rmse

# plotting functions

'''Function plots home transactions/purchases by months in 2017'''
def plot_transactions_by_month(df, month_col):
        # creating a lambda function to isolate the dates by month and year
        months_lst = df[month_col].map(lambda dt: dt.strftime('%Y-%m'))

        # adding the new 2017 month series to main df as new column
        df[month_col] = months_lst.astype("str")

        grouped_df = df.sort_values(month_col).groupby(month_col).size().to_frame("count").reset_index()

        sns.set(font_scale = .5, style = "darkgrid")
        ax = sns.countplot(x = month_col,
                        data = df,
                        order = grouped_df[month_col],
                        palette = "crest")

        ax.bar_label(ax.containers[0])

        plt.xlabel(None)
        plt.title("2017 Home Purchases by Month")
        plt.show()


'''Function takes in a dataframe and plots all variables against one another using sns.pairplot function. 
This function also shows the line-of-best-fit for ea. plotted variables'''
def plot_variable_pairs1(df):
    plt.figure(figsize = (15, 6))
    sns.set(font_scale = 1)
    sns.pairplot(df.sample(1000, random_state = 123), corner = True, kind="reg", diag_kind = "kde", plot_kws={'line_kws':{'color':'red'}})
    
    plt.show()


'''function takes in a dataframe and list, and plots them against target variable w/'line-of-best-fit'''
def plot_variable_pairs2(train_df, x_features_lst):
    for col in x_features_lst:
        plt.figure(figsize = (10, 4))
        sns.set(font_scale = 1)

        # plotting ea. feature against target variable with added "independent jitter" for easier visual
        ax = sns.regplot(train_df[col].sample(2000), \
        train_df["home_value"].sample(2000), \
        x_jitter = 1, # adding superficial noise to independent variables
        line_kws={
            "color": "red", 'linewidth': 1.5})
        
        ax.figure.set_size_inches(18.5, 8.5)
        sns.despine()
        # removing scientific notations
        ax.ticklabel_format(style = "plain")
        
        # removing x_axis label
        ax.set_xlabel(None)

        plt.title(col)
        plt.show()



'''function for plotting categorical or discrete/low feature option columns'''
def plot_discrete(df):
    discrete_vars = [
    'bathroom_count',
    'bedroom_count',
    'fips_code']

    for col in discrete_vars:
        plt.figure(figsize=(12, 6))
        sns.set(font_scale = 1)
        ax = sns.countplot(x = df[col], 
                        data = df,
                        palette = "crest_r",
                        order = df[col].value_counts().index)
        
        ax.bar_label(ax.containers[0])
        ax.set(xlabel = None)
        plt.title(col)
        plt.show()


'''function for plotting continuous/high feature option columns'''
def plot_continuous(df):
    # setting and plotting continuous features/variables
    feature_lst = [
    'living_sq_feet', \
    'latitude', \
    'longitude', \
    'property_sq_feet', \
    'city_id', \
    'county_id', \
    'zip_code', \
    'year_built']


    for col in feature_lst:
        plt.figure(figsize=(12, 6))
        ax = sns.histplot(df[col], 
                        bins = 50,
                        kde = True)
        
        ax.set(xlabel = None)
        ax.ticklabel_format(style = "plain")

        plt.title(col)
        plt.show()

'''plotting the target variable'''
def plot_target(df):
    plt.figure(figsize = (15, 8))
    sns.set(font_scale = 1)
    ax = sns.histplot(df, bins = 20, kde = True)

    ax.ticklabel_format(style = "plain") # removing axes scientific notation 
    ax.bar_label(ax.containers[0])

    plt.axvline(df.mean(), linewidth = 2, color = 'purple', ls = ':', label = "mean")
    plt.axvline(df.median(), linewidth = 2, color = 'red', alpha = 0.5, label = "median")
    plt.title("Home Value Averages and Median")
    plt.xlabel(None)
    plt.legend()
    plt.show()


'''Plotting features against target variable w/line-of-best-fit'''
def features_and_target(df):
    cols = df.columns.to_list()
    for col in cols:
        plt.figure(figsize = (10, 4))
        sns.set(font_scale = 1)

        # plotting ea. feature against target variable with added "independent jitter" for easier visual
        ax = sns.regplot(df[[col]].sample(2000), \
        df["home_value"].sample(2000), \
        
        # adding superficial noise to independent variables to help visualize the individual plots
        x_jitter = 1, \
        line_kws={
            "color": "red", 'linewidth': 1.5})
        
        ax.figure.set_size_inches(18.5, 8.5)
        sns.despine()
        # removing scientific notations
        ax.ticklabel_format(style = "plain")
        
        # removing x_axis label
        ax.set_xlabel(None)

        plt.title(col)
        plt.show()


'''Function to plot model residuals against actual (y_variable):
furthermore, model takes in two (2) dataframes or series (y and y_hat) - where y_hat = model preditions
and calculates the model residula (y - y_hat)'''
def plot_residuals(y, y_hat):
    y = y.sample(5000, random_state = 123)
    y_hat = y_hat.sample(5000, random_state = 123)
    residuals = y - y_hat
    plt.figure(figsize = (12,6))
    ax = sns.scatterplot(x = y, y = residuals, 
                alpha = 1/5)

    plt.axhline(y = 0, ls = ':', color = "red", linewidth = 2)
    plt.xlabel('y_variable')
    plt.ylabel('Residual')
    plt.title('Model Residuals')
    # removing axes scientific notation
    plt.ticklabel_format(style = "plain") 

    # making individual plots more readable
    ax.figure.set_size_inches(18, 8)



# compare functions

'''Function to compare Model vs. Baseline Sum-of-Squares'''
# note: the lower the SSE, the lower the predicted error from actual observations & the better the model represents the "actual" predictions
def compare_sum_of_squares(SSE_baseline, SSE_model):
    if SSE_model >= SSE_baseline:
        print("Model DOES NOT outperform baseline.")
    else:
        print("Model outperforms baseline!")



'''Function that takes in y_variable and y_hat (predictions) and returns whether or not the created model 
has a lower sum of squared error than baseline'''
def better_than_baseline(y, y_hat):
    df_model = y.sample(1000, random_state = 123) - y_hat.sample(1000, random_state = 123)
    df_model["residual^2"] = df_model.round(2) ** 2

    # calculating a baseline
    baseline = round(y.sample(1000, random_state = 123).mean(), 2)

    # creating an empty DataFrame with n_rows as y
    df = pd.DataFrame(index = range(len(y)))

    # setting the n_values for all indices in df
    df["baseline"] = baseline

    df["baseline_residuals"] = y - df["baseline"]
    df["baseline_residual^2"] = df["baseline_residuals"].round(2) ** 2

    # generating sum of squared error
    SSE_model = sum(df_model["residual^2"])
    SSE_baseline = sum(df["baseline_residual^2"])


    if SSE_model < SSE_baseline:
        return True
    else:
        return False


# feature engineering functions

def select_kbest(X_train, y_train, number_of_top_features):
    # using Select-K-Best to select the top number of features for predicting y variable 
    # parameters: f_regression stats test, all features
    f_selector = SelectKBest(f_regression, k = number_of_top_features)

    # find the top number of independent variables (X's) correlated with y
    f_selector.fit(X_train, y_train)

    # boolean mask of whether the column was selected or not
    feature_mask = f_selector.get_support()

    # get list of top (2) K features. 
    f_feature = X_train.iloc[:,feature_mask].columns.tolist()
    
    return pd.DataFrame(f_feature)


def recursive_feature_eng(X_train, y_train, number_of_top_features):

    # initialize the ML algorithm
    lm = LinearRegression()

    rfe = RFE(lm, n_features_to_select = number_of_top_features)

    # fit the data using RFE
    rfe.fit(X_train,y_train) 

    # get the mask of the columns selected
    feature_mask = rfe.support_

    # get list of the column names. 
    rfe_features = X_train.iloc[:,feature_mask].columns.tolist()

    # view list of columns and their ranking
    # get the ranks using "rfe.ranking" method
    variable_ranks = rfe.ranking_

    # get the variable names
    variable_names = X_train.columns.tolist()

    # combine ranks and names into a df for clean viewing
    rfe_ranks_df = pd.DataFrame({'Feature': variable_names, 'Ranking': variable_ranks})

    # sort the df by rank
    return rfe_ranks_df.sort_values('Ranking')


# Function returns rasidual/error reports for model predictions
def get_error_report(y, y_hat):
    # importing math.sqrt module for calculations
    from math import sqrt
    
    # generating model residuals and residuals squared
    df = y - y_hat
    df["residual^2"] = df.round(2) ** 2

    # generating sum of squared error
    SSE = sum(df["residual^2"])

    # generating explained sum of squares
    ESS = sum((y_hat - y.mean()) ** 2)

    # generating total sum of squares error
    TSS = ESS + SSE

    # generating mean squared error
    MSE = SSE/len(y)

    # generating root mean squared error
    RMSE = sqrt(MSE)

    print(f'{y_hat.name} SSE: {SSE}')
    print(f'{y_hat.name} ESS: {ESS}')
    print(f'{y_hat.name} TSS: {TSS}')
    print(f'{y_hat.name} MSE: {MSE}')
    print(f'{y_hat.name} RMSE: {RMSE}')

    return SSE, ESS, TSS, MSE, RMSE

# Model Residual (error) Plot
def plot_model_residuals(df):
    plt.figure(figsize=(16,8))
    plt.axhline(label='No Error', 
                color = 'purple',
                ls = ':')

    # plotting linear model
    plt.scatter(df['home_value'].sample(300, random_state = 123), 
                df['linear_predictions'].sample(300, random_state = 123) - df['home_value'].sample(300, random_state = 123), 
                alpha=0.5,
                color='red', 
                s=100, 
                label='Linear Regression')

    # plotting lasso lars model
    plt.scatter(df['home_value'].sample(300, random_state = 123), 
                df['lars_predictions'].sample(300, random_state = 123) - df['home_value'].sample(300, random_state = 123), 
                alpha=0.5,
                color='yellow',
                s=100, 
                label='Lasso Lars')

    # plotting tweedie model
    plt.scatter(df['home_value'].sample(300, random_state = 123), 
                df['glm_predictions'].sample(300, random_state = 123) - df['home_value'].sample(300, random_state = 123), 
                alpha=0.5,
                color='green',
                s=100, 
                label='Tweedie Regressor')

    plt.legend()
    plt.xlabel('Actual Home Value')
    plt.ylabel('Residual Error')
    plt.title('Model Residual Plot')

    plt.show()



# plotting actual home values, baseline_mean_predictions predictions, and model predictions
def plot_models(df):
    plt.figure(figsize = (16, 10))
    plt.plot(df['home_value'].sample(300, 
            random_state = 123), 
            df['baseline_mean_predictions'].sample(300, random_state = 123), 
            alpha=0.5,
            color='red', 
            ls = ':', 
            label='_nolegend_')

    # plotting home value line of best fit
    plt.plot(df['home_value'].sample(300, random_state = 123), 
            df['home_value'].sample(300, random_state = 123), 
            alpha=0.5,
            color='purple', 
            label='_nolegend_')

    # linear model plot
    plt.scatter(df['home_value'].sample(300, random_state = 123), 
                df['linear_predictions'].sample(300, random_state = 123), 
                color='red',
                s=100,
                label='Linear Regression')

    # lasso lars plot
    plt.scatter(df['home_value'].sample(300, random_state = 123), 
                df['lars_predictions'].sample(300, random_state = 123), 
                color='yellow', 
                s=100, 
                label='Lasso Lars')

    # tweedie/glm plot
    plt.scatter(df['home_value'].sample(300, random_state = 123), 
                df['glm_predictions'].sample(300, random_state = 123), 
                color = 'green',
                s=100, 
                label='Tweedie Regressor')


    plt.legend()
    plt.xlabel("Actual Home Value")
    plt.ylabel('Predicted Home Value')
    plt.title('Actual Home Values vs Model Predicted Home Values')

    plt.show()

def model_distributions(df):
    # Distribution of my model predictions (linear & tweedie)
    plt.figure(figsize = (16, 8))
    sns.set(style = "darkgrid")

    plt.hist(df['home_value'], color='blue', alpha=0.4, label='Actual Home Value')
    plt.hist(df['linear_predictions'], color='red', alpha=0.4, label='Linear Regression')
    plt.hist(df['lars_predictions'], color='green', alpha=0.4, label='Laso Lars')


    plt.xlabel('Home Value')
    plt.ylabel('Frequency')
    plt.title('Frequency of Home Values by Predictive Model')
    plt.legend()

    plt.show()