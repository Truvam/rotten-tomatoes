from sklearn.metrics.pairwise import cosine_similarity
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from rake_nltk import Rake
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy import sparse
import pandas as pd
import numpy as np
import surprise
import re


class Binary:
    def __init__(self, reviews, reviews_rs, critic_uid):
        self.reviews = reviews
        self.reviews_rs = reviews_rs

        # create binary dataset, that contains which movies the critics have reviewed
        self.binary_dataset = pd.get_dummies(reviews_rs.set_index(
            'critic_uid')['id'].astype(str)).max(level=0).sort_index()

        self.critic_uid = critic_uid

    def binary_popularity_based(self, top_n=5):
        # Number of reviews for each movie
        recommended_movies = self.reviews.groupby('id')['rating'].count(
        ).sort_values(ascending=False).head(top_n)
        for i, (movie_id, nratings) in enumerate(recommended_movies.items()):
            print("movie_id: {0} with number of rating: {1}".format(movie_id,
                                                                    nratings))

    def ar_create_df(self):
        critics_id = self.reviews_rs['critic_uid'].unique().tolist()
        dataset = []
        for i in critics_id:
            dataset.append(
                self.reviews_rs.loc[self.reviews_rs['critic_uid'] == i,
                                    'id'].tolist())

        te = TransactionEncoder()
        te_ary = te.fit(dataset).transform(dataset)
        df = pd.DataFrame(te_ary, columns=te.columns_)
        return df

    def frequent_itemsets(self, df, min_support=0.03, verbose=0):
        frequent_itemsets = apriori(df, min_support=min_support,
                                    use_colnames=True,
                                    verbose=verbose)
        return frequent_itemsets

    def association_rules(self, frequent_itemsets, metric, min_threshold):
        return association_rules(frequent_itemsets, metric=metric,
                                 min_threshold=min_threshold)

    def get_ar_recommendation(self, df, rules, critic_id, critics, top_n=5):
        res = rules[['antecedents', 'consequents', 'support']]

        movies_watched = df.loc[critic_id]
        movies_watched = [k for k, v in movies_watched.items() if v]

        recommended_movies = []
        for movie_id in movies_watched:
            cs = res.loc[res['antecedents'] == {movie_id}, 'consequents']
            cs = [list(x) for x in cs.values]
            recommended_movies.append(cs)

        recommended_movies = sum(recommended_movies, [])[:top_n]
        print('Top movies for reviewer {0}: {1}'.format(critic_id, critics[
            critic_id]))
        for movie_id in recommended_movies:
            if movie_id not in movies_watched:
                print("movie_id: {0}".format(movie_id[0]))

    def normalize_dataset(self):
        dataset_norm = self.binary_dataset.copy()
        # normalize the data for all users.
        # to make sure that users with more reviews don't influentiate users with less reviews
        magnitude = np.sqrt(np.square(dataset_norm).sum(axis=1))
        dataset_norm = dataset_norm.divide(magnitude, axis='index')
        return dataset_norm

    def calculate_similarity(self, dataset):
        """
        Calculate the column-wise cosine similarity for a sparse
        matrix. Return a new dataframe matrix with similarities.
        """
        sparse_matrix = sparse.csr_matrix(dataset)
        similarities = cosine_similarity(sparse_matrix.transpose())
        sim = pd.DataFrame(data=similarities,
                           index=self.binary_dataset.columns,
                           columns=self.binary_dataset.columns)
        return sim

    def binary_collab_item_based(self, item, top_n=5):
        # normalize data
        dataset_norm = self.normalize_dataset()

        # Build similarity matrix
        similarity_matrix = self.calculate_similarity(dataset_norm)
        predictions = similarity_matrix.loc[str(item)].nlargest(top_n + 1)

        print("Top " + str(
            top_n) + " movie recommendation based on movie_id = " + str(
            item) + ":")
        for index, row in predictions.iteritems():
            print("movie_id: " + str(index) + " with similarity: " + str(
                round(row, 3)))
        # print(similarity_matrix.loc[str(item)].nlargest(top_n+1))

    def binary_collab_user_based(self, user, top_n=5):
        # normalize data
        dataset_norm = self.normalize_dataset()

        # Build similarity matrix
        similarity_matrix = self.calculate_similarity(dataset_norm)

        # Get movies that user liked
        user_id = self.critic_uid[user]
        movies_user_liked = dataset_norm.loc[user_id]
        movies_user_liked = movies_user_liked[
            movies_user_liked > 0].index.values

        # Users likes for all items as a sparse vector.
        user_rating_vector = dataset_norm.loc[user_id]

        # Calculate the score.
        score = similarity_matrix.dot(user_rating_vector).div(
            similarity_matrix.sum(axis=1))

        # Remove the user liked movies from the recommendation.
        score = score.drop(movies_user_liked)

        predictions = score.nlargest(top_n)

        print("Top " + str(
            top_n) + " movie recommendation based on critic_id = " + str(
            user) + ":")
        for index, row in predictions.iteritems():
            print("movie_id: " + str(index) + " with similarity: " + str(
                round(row, 3)))
        # print(score.nlargest(top_n))


class NonBinary:
    def __init__(self, reviews, reviews_rs, critic_uid, critics):
        self.reviews = reviews
        self.reviews_rs = reviews_rs
        self.critic_uid = critic_uid
        self.critics = critics

    def nbinary_popularity_based(self, top_n=5):
        # Average rating for each movie in descending order:
        recommended_movies = self.reviews.groupby('id')['rating'].mean(
        ).sort_values(ascending=False).head(top_n)
        for i, (movie_id, nratings) in enumerate(recommended_movies.items()):
            print("movie_id: {0} with average rating: {1}".format(movie_id,
                                                                  nratings))

    def nb_collaborative_filtering(self, critic_id, top_n=5):
        lower_rating = self.reviews_rs['rating'].min()
        upper_rating = self.reviews_rs['rating'].max()

        reader = surprise.Reader(rating_scale=(0.0, 10.0))
        data = surprise.Dataset.load_from_df(self.reviews_rs, reader)

        alg = surprise.SVDpp()
        output = alg.fit(data.build_full_trainset())

        # Get a list of all unique movies
        movies_id = self.reviews_rs['id'].unique()

        # Get a list of movies_id that reviewer 0 has rated
        movies_id_critic = self.reviews_rs.loc[
            self.reviews_rs['critic_uid'] == critic_id, 'id']

        # Remove the movie_id that reviewer 0 has rated
        movies_ids_to_pred = np.setdiff1d(movies_id, movies_id_critic)

        testset = [[critic_id, movie_id, 10.0] for movie_id in
                   movies_ids_to_pred]
        predictions = alg.test(testset)

        pred_ratings = np.array([pred.est for pred in predictions])

        # Find the index of the maximum predicted rating
        i_max = np.argpartition(pred_ratings, -top_n)[-top_n:]

        # Use this to find the corresponding movie_id to recommend
        print('Top movies for reviewer {0}: {1}'.format(critic_id,
                                                        self.critics[
                                                            critic_id]))
        for i in i_max:
            movie_id = movies_ids_to_pred[i]
            print('movie_id: {0} with predicted rating: {1}'.format(movie_id,
                                                                    pred_ratings[
                                                                        i]))


class ContextAware:
    def __init__(self, movie_info, reviews_rs, critics):
        self.reviews_rs = reviews_rs
        self.critics = critics
        # Using Rake to extract the most relevant words from synopsis
        ca_df = movie_info.copy()

        # Removing movies with no synopsis
        ca_df = ca_df.loc[~ca_df['synopsis'].isnull()]

        # Removing movies with no genre
        ca_df = ca_df.loc[~ca_df['genre'].isnull()]

        # Removing movies with no director
        ca_df = ca_df.loc[~ca_df['director'].isnull()]

        # Create new column with the key words from synopsis for each movie
        ca_df['key_words'] = ""

        ca_df['key_words'] = ca_df['synopsis'].apply(generate_key_words)

        ca_df.drop(
            columns=['synopsis', 'rating', 'writer', 'theater_date', 'dvd_date',
                     'currency',
                     'box_office', 'runtime', 'studio'], inplace=True)

        ca_df['genre'] = ca_df['genre'].apply(split_genre)
        ca_df['director'] = ca_df['director'].apply(remove_space_director)

        self.ca_df = ca_df

    def merge_columns(self):
        # Merge all columns into one containing all words
        self.ca_df['all_words'] = self.ca_df[['genre', 'director',
                                              'key_words']].apply(
            lambda x: ' '.join(x), axis=1)

        self.ca_df.drop(columns=['genre', 'director', 'key_words'],
                        inplace=True)

        # Changing index to movie id
        self.ca_df = self.ca_df.set_index('id')

    def create_similarity_matrix(self):
        # instantiating and generating the count matrix
        count = CountVectorizer()
        count_matrix = count.fit_transform(self.ca_df['all_words'])

        # generating the cosine similarity matrix
        cosine_sim = cosine_similarity(count_matrix, count_matrix)

        """
        all the numbers on the diagonal are 1 because, of course, every movie is 
        identical to itself. 
        The matrix is also symmetrical because the similarity between A and B is 
        the same as the similarity between B and A.
        """
        return cosine_sim

    def get_recommendations(self, movie_id, cosine_sim, top_n=5, user=False):
        recommended_movies = []

        # creating a Series for the movie titles so they are associated to an
        # ordered numerical list I will use in the function to match the indexes
        indices = pd.Series(self.ca_df.index)
        # getting the index of the movie that matches the movie id
        idx = indices[indices == movie_id].index[0]

        # creating a Series with the similarity socres in descending order
        score_series = pd.Series(cosine_sim[idx]).sort_values(ascending=False)

        top_n_indexs = list(score_series.iloc[1:top_n + 1].index)

        for i in top_n_indexs:
            recommended_movies.append(self.ca_df.index[i])
        if not user:
            print("Top movies similar to movie_id: {0}".format(movie_id))
            for movie_id in recommended_movies:
                print("movie_id: {0}".format(movie_id))
        else:
            return recommended_movies

    def get_recommendation_user(self, critic_id, cosine_sim, top_n=5):
        recommended_movies = []

        df = self.reviews_rs.loc[self.reviews_rs['critic_uid'] == critic_id]

        # Get top 5 rated movies by a specific critic
        df = df.nlargest(5, ['rating'])

        # Get movies id from top 5
        movies_ids = list(df['id'])

        for movie_id in movies_ids:
            if movie_id in self.ca_df.index:
                recommended_movies.append(self.get_recommendations(movie_id,
                                                                   cosine_sim,
                                                                   user=True))

        recommended_movies = sum(recommended_movies, [])[:top_n]
        print("Top movies for reviewer {0}: {1}".format(
            critic_id, self.critics[critic_id]))
        for movie_id in recommended_movies:
            print("movie_id: {0}".format(movie_id))


def generate_key_words(synopsis):
    # instantiating Rake, by default it uses english stopwords from NLTK
    # and discards all puntuation characters as well
    r = Rake()

    # extracting the words by passing the text
    r.extract_keywords_from_text(synopsis)

    # getting the dictionary whith key words as keys and their scores as values
    key_words_dict_scores = r.get_word_degrees()

    # Removing stop words
    stop_words = set(stopwords.words('english'))
    words = ' '.join(list(key_words_dict_scores.keys()))

    word_tokens = word_tokenize(words)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]

    words = ' '.join(filtered_sentence)
    result = re.sub(r'[^[a-zA-Z]]', "", words)

    return result


def split_genre(genre):
    if "|" in genre:
        return genre.replace("|", " ").lower()
    return genre.lower()


def remove_space_director(director):
    director = director.replace(" ", "").lower()
    if "|" in director:
        return director.replace("|", " ")
    return director
