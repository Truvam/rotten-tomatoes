import pandas as pd
import numpy as np
from math import sqrt


def convert_rating_aux(rating):
    """
    Converting all ratings to one single scale /10
    For any v/m , being v the value and m the max value in the scale,
    the new rating will be (v * 10)/m
    Note: We won't to be able to convert single numbers like ['1', '6', ...]
          since we don't know the corresponding scale.
          Wee also need to verify if the v is bigger than m
    """
    grade_letters = {'A+': 12, 'A': 11, 'A-': 10, 'B+': 9, 'B': 8, 'B-': 7,
                     'C+': 6, 'C': 5, 'C-': 4, 'D+': 3, 'D': 2, 'D-': 1, 'F': 0}

    if '/' in str(rating) and ' ' not in str(rating):
        i = rating.index('/')
        value = float(rating[:i])
        scale_max = float(rating[i+1:])
        if value < scale_max:
            new_rating = value*10/scale_max
        else:
            new_rating = np.nan
    elif str(rating) in grade_letters:
        new_rating = grade_letters[rating]*10/12
    else:
        new_rating = np.nan
    return new_rating


def convert_ratings(reviews):
    reviews['rating'] = reviews['rating'].apply(convert_rating_aux)


def create_rs_df(reviews):
    """
    Create new dataset to work with recommender systems
    Associated each critic to an id
    Has 3 columns: critics_id, movie_id and rating
    """
    critics = reviews['critic'].unique().tolist()
    critic_uid = {}
    uid = 0
    for critic in critics:
        critic_uid[critic] = uid
        uid += 1

    critics = reviews['critic'].tolist()
    new_column = []
    for critic in critics:
        new_column.append(critic_uid[critic])

    reviews_rs = reviews.copy()
    reviews_rs.insert(loc=0, column='critic_uid', value=new_column)

    reviews_rs.drop(columns=['review', 'top_critic', 'publisher', 'date',
                             'fresh', 'critic'], inplace=True)

    return reviews_rs, critics, critic_uid


def creat_ratings_mean_count(reviews):
    # Create DataFrame with average rating for each movie:
    ratings_mean_count = pd.DataFrame(reviews.groupby('id')['rating'].mean())

    # Add number of ratings per movie to DataFrame:
    ratings_mean_count['rating_counts'] = pd.DataFrame(
        reviews.groupby('id')['rating'].count())
    return ratings_mean_count
