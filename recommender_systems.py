from sklearn.metrics.pairwise import cosine_similarity
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from scipy import sparse
import pandas as pd
import numpy as np
import surprise


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
        return self.reviews.groupby('id')['rating'].count().sort_values(
            ascending=False).head(top_n)

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
        return self.reviews.groupby('id')['rating'].mean().sort_values(
            ascending=False).head(top_n)

    def nb_collaborative_filtering(self, critic_id, top_n=5):
        lower_rating = self.reviews_rs['rating'].min()
        upper_rating = self.reviews_rs['rating'].max()

        print("Review range: {0} to {1}".format(lower_rating, upper_rating))

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
        print('Top movie for reviewer {0}: {1}'.format(critic_id,
                                                       self.critics[critic_id]))
        for i in i_max:
            movie_id = movies_ids_to_pred[i]
            print('movie_id: {0} with predicted rating: {1}'.format(movie_id,
                                                                    pred_ratings[
                                                                        i]))
