import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Load raw data to pre-process.
    
    params: 
        - messages_filepath: messages filepath
        - categories_filepath: categories filepath
    return:
        - Pandas dataframe of messages and categories
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    return (messages, categories)


def clean_data(messages, categories):
    """
    Normalize the categories data and join it with messages.
    
    params:
        - messages: messages pandas dataframe
        - categories: categories pandas dataframe
    return:
        - processed pandas dataframe
    
    """
    # clean messages
    messages = messages.drop_duplicates()
    
    # clean categories
    categories = categories.drop_duplicates()
    
    # merge two datasets
    df = categories.merge(messages, on='id', how='left')
    
    # clean categories column to prepare a categories dataframe
    prepared_cat = categories.categories.str.split(';', expand=True)
    
        # prepare header row
    first_row = prepared_cat.loc[0, :]
    header = first_row.apply(lambda x: x.split('-')[0])
    prepared_cat.columns = header
    
        # prepare categories
    prepared_cat = prepared_cat.apply(lambda x: x.str.split('-').str[1].astype(bool))
    
    # prepare final df
    df = df.drop(columns=['categories'])
    df = pd.concat([df, prepared_cat], axis=1)
    df = df.drop_duplicates()
    df = df.dropna(how='any', axis=0)
    
    return df


def save_data(df, database_filename):
    """
    Save data to a SQLite database.
    
    params: 
        - df: dataframe that needs saving
        - database_filename: file name of database
    return: None
    
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('cleaned_data', engine, index=False, if_exists='replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        messages, categories = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(messages, categories)
        
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