import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('dark')


def plot_pre_processing(reviews_bf, reviews):
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = 'Before Pre-Proccessing', 'After Pre-Processing '
    sizes = [len(reviews_bf), len(reviews_bf) - len(reviews)]

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=False,
            startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title("Before and after Pre-Processing ratings")
    plt.show()


def plot_nratings_per_movie(ratings_mean_count):
    mean = int(round(ratings_mean_count['rating_counts'].mean(), 0))
    plt.figure(figsize=(8, 6))
    plt.rcParams['patch.force_edgecolor'] = True
    plt.title("Number of ratings per movie")
    plt.xlabel('rating_counts')
    plt.ylabel('Nº movies')
    plt.axvline(ratings_mean_count['rating_counts'].mean(), color='r',
                linestyle='dashed', linewidth=2, label='mean = ' + str(mean))
    ratings_mean_count['rating_counts'].hist(bins=50)
    plt.legend()


def plot_average_ratings_per_movie(ratings_mean_count):
    mean = round(ratings_mean_count['rating'].mean(), 1)
    plt.figure(figsize=(8, 6))
    plt.rcParams['patch.force_edgecolor'] = True
    plt.title("Average Ratings per movie")
    plt.xlabel('Ratings (0-10)')
    plt.ylabel('Nº movies')
    plt.axvline(ratings_mean_count['rating'].mean(), color='r',
                linestyle='dashed', linewidth=2, label='mean = ' + str(mean))
    ratings_mean_count['rating'].hist(bins=50)
    plt.legend()


def plot_average_against_nratings_per_movie(ratings_mean_count):
    plt.figure(figsize=(8, 6))
    plt.rcParams['patch.force_edgecolor'] = True
    sns.jointplot(x='rating', y='rating_counts', data=ratings_mean_count,
                  alpha=0.4)
