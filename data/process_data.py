import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Performs the "load" part of the ETL pipeline. 
    Given paths to 2 CSV files, one for messages and another for categories, load those files
    into dataframes and then concatenate into a single dataframe.
    """
    # read the csv files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # index both dataset by id, will also help with doing concat
    messages = messages.set_index('id', drop=True)
    categories = categories.set_index('id', drop=True)

    # join columns by index (id)
    df = pd.concat([messages, categories], axis=1)
    
    return df


def clean_data(df):
    """
    Performs the "transform" part of the ETL pipeline.
    Given a dataframe which contains both messages and categories, ensure that 
    category fields are binary columns.
    """
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(";", expand=True)
    
    # select the first row of the categories dataframe
    row = categories.head(1).values[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = [cat.split('-')[0] for cat in row]

    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])
        # convert column from string to numeric
        categories[column] = categories[column].astype('int')

    # drop the original categories column from `df`
    df = df.drop(labels=['categories'], axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename, database_tablename='tweets'):
    """
    Saves a dataframe into a table in an sqlite file.
    """
    engine = create_engine("sqlite:///{}".format(database_filename))
    df.to_sql(database_tablename, engine, index=False, if_exists='replace')
    return True


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()