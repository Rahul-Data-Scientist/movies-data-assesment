import numpy as np
import pandas as pd
import ast


def safe_literal_eval(val):
    if isinstance(val, str):  # Apply only if it's a string
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return ["Unknown"]  # Return ["Unknown"] instead of NaN
    return val  # Return the original value if it's not a string


def preprocess(df):
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)

    list_columns = ['directors', 'writers', 'stars', 'genres', 'countries_origin',
                    'filming_locations', 'production_companies', 'Languages']

    for col in list_columns:
        df[col] = df[col].apply(safe_literal_eval)

    # Upon checking, it is found that one movie whose original name is 2.O is written as '2'. So, correcting it
    df.loc[df['Title'] == '2', 'Title'] = '2.O'
    return df


def convert_votes(value):
    if 'K' in value:
        return float(value.replace('K', '')) * 1000
    elif 'M' in value:
        return float(value.replace('M', '')) * 1000000
    else:
        return float(value)


def preprocess_uncleaned(df):
    df = preprocess(df)
    df.loc[df['directors'].apply(len) == 0, 'directors'] = 'Unknown'
    df.loc[df['stars'].apply(len) == 0, 'stars'] = 'Unknown'
    df.loc[df['writers'].apply(len) == 0, 'writers'] = 'Unknown'
    df.loc[df['genres'].apply(len) == 0, 'genres'] = 'Unknown'
    df.loc[df['countries_origin'].apply(len) == 0, 'countries_origin'] = 'Unknown'
    df.loc[df['filming_locations'].apply(len) == 0, 'filming_locations'] = 'Unknown'
    df.loc[df['production_companies'].apply(len) == 0, 'production_companies'] = 'Unknown'
    df.loc[df['Languages'].apply(len) == 0, 'Languages'] = 'Unknown'
    df['Votes'] = df['Votes'].apply(convert_votes)
    return df