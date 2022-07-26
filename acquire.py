# importing needed libraries/modules
import os
import pandas as pd
import numpy as np

# importing visualization libraries 
import seaborn as sns
import matplotlib.pyplot as plt

# importing sql 
import env
from env import user, password, host, get_connection

# sklearn train, test, and split function
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, RFE, f_regression
from sklearn.linear_model import LinearRegression


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
    # dropping null values in dataset (where <=1% makeup nulls in ea. feature/column)
    df = df.dropna()

    # converting "bedroom_count" "year_built", and "fips" columns to int type
    df["bedroom_count"] = df["bedroom_count"].astype("int").round()
    df["year_built"] = df["year_built"].astype("int")
    df["fips"] = df["fips"].astype("int")
    
    to_interger = ["bedroom_count", "city_id", "county_id", "zip_code"]
    df[to_interger] = df[to_interger].astype("int")
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])

    # rearranging columns for easier readibility
    df = df[[
        'bedroom_count',
        'bath_count',
        'finished_sq_feet',
        'year_built',
        'fips',
        'tax_amount',
        'home_value']]

    # lastly, return the cleaned dataset
    return df


# function for handling outliers in the dataset
def zillow_outliers(df):
    df = df[df["home_value"] <= 900_000]
    df = df[df["living_sq_feet"] <= 8_000]
    df = df[df["bedroom_count"] <= 6]
    df = df[df["bathroom_count"] <= 6.5]

    return df

'''Function created to split the initial dataset into train, validate, and test sub-datsets'''
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
    


# plotting functions

'''Function takes in a dataframe and plots all variables against one another using sns.pairplot function. 
This function also shows the line-of-best-fit for ea. plotted variables'''
def plot_variable_pairs1(df):
    g = sns.pairplot(data = df.sample(1000), corner = True, kind="reg", diag_kind = "kde", plot_kws={'line_kws':{'color':'red'}})
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
def plot_discrete(df, feature_lst):
    for col in feature_lst:
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
def plot_continuous(df, feature_lst):
    for col in feature_lst:
        plt.figure(figsize=(12, 6))
        ax = sns.distplot(x = df[col], 
                        bins = 50,
                        kde = True)
        
        ax.set(xlabel = None)
        ax.ticklabel_format(style = "plain")

        plt.axvline(df[col].mean(), linewidth = 2, color = 'purple', ls = ':', label = "mean")
        plt.axvline(df[col].median(), linewidth = 2, color = 'red', alpha = 0.5, label = "median")
        plt.title(col)
        plt.legend()
        plt.show()


'''plotting the target variable'''
def plot_target(df):
    plt.figure(figsize = (12, 5))
    sns.set(font_scale = .8)
    ax = sns.histplot(df, bins = 20, kde = True)

    ax.ticklabel_format(style = "plain") # removing axes scientific notation 
    ax.bar_label(ax.containers[0])

    plt.axvline(df.mean(), linewidth = 2, color = 'purple', ls = ':', label = "mean")
    plt.axvline(df.median(), linewidth = 2, color = 'red', alpha = 0.5, label = "median")
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
    ax = plt.scatter(x = y, y = residuals, 
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
    
    return f_feature


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