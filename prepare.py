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

# sklearn df, test, and split function
from sklearn.model_selection import train_test_split


'''function for plotting categorical or discrete/low feature option columns'''
def plot_discrete(df, feature_lst):
    for column in df[[feature_lst]]:
        plt.figure(figsize=(12, 6))
        sns.set(font_scale = 1)
        ax = sns.countplot(x = column, 
                        data = df,
                        palette = "crest_r",
                        order = df[column].value_counts().index)
        ax.bar_label(ax.containers[0])
        ax.set(xlabel = None)
        plt.title(column)
        plt.show()


'''function for plotting continuous/high feature option columns'''
def plot_continuous(df, feature_lst):
    for column in df[[feature_lst]]:
        plt.figure(figsize=(12, 6))
        ax = sns.distplot(x = df[feature_lst], 
                        bins = 50,
                        kde = True)
        ax.set(xlabel = None)
        plt.axvline(df[column].median(), linewidth = 2, color = 'purple', alpha = 0.4, label = "median")
        plt.title(column)
        plt.legend()
        plt.show()


'''plotting the target variable'''
def plot_target(df):
    plt.figure(figsize = (12, 5))
    sns.set(font_scale = .8)
    ax = sns.histplot(df, bins = 20, kde = True)

    ax.ticklabel_format(style = "plain") # removing axes scientific notation 
    ax.bar_label(ax.containers[0])

    plt.axvline(df.median(), linewidth = 2, color = 'purple', alpha = 0.4, label = "median")
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


'''Function that takes in any pd.method and returns it without row/columns cut-offs'''
def display_all(pd_function):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        return display(pd_function)