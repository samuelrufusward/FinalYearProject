import matplotlib.pyplot as plt
import numpy as np
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler
import time
import itertools
import scipy
import scipy.cluster.hierarchy as sch
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from yellowbrick.cluster import KElbowVisualizer



def find_optimal_cluster_number_gmm(data):
    n_components = range(1, 30)
    covariance_type = ['spherical', 'tied', 'diag', 'full']
    score=[]
    for cov in covariance_type:
        for n_comp in n_components:
            gmm=GaussianMixture(n_components=n_comp,covariance_type=cov)
            gmm.fit(data)
            score.append((cov,n_comp,gmm.bic(data)))
    print(score)


def cluster_corr(corr_array, threshold_ratio=0.5, inplace=False):
    """
    Rearranges the correlation matrix, corr_array, so that groups of highly
    correlated variables are next to eachother

    Parameters
    ----------
    corr_array : pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix

    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix with the columns and rows rearranged
    """
    pairwise_distances = sch.distance.pdist(corr_array)
    linkage = sch.linkage(pairwise_distances, method='complete')


    max_pairwise_dist = pairwise_distances.max()
    print("max pairwise dist: ", max_pairwise_dist)
    cluster_distance_threshold = float(max_pairwise_dist) * threshold_ratio
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold,
                                        criterion='distance')
    print("cluster values: ", idx_to_cluster_array)
    print("n clusters: ", np.unique(idx_to_cluster_array))

    # Visualise dendrogram
    sch.dendrogram(linkage)
    plt.axhline(cluster_distance_threshold, color='k', ls='--')
    plt.show()

    idx = np.argsort(idx_to_cluster_array)

    if not inplace:
        corr_array = corr_array.copy()

    if isinstance(corr_array, pd.DataFrame):
        return corr_array.iloc[idx, :].T.iloc[idx, :], idx_to_cluster_array
    return corr_array[idx, :][:, idx], idx_to_cluster_array


def time_series_kmeans_clustering(X, n_clusters=4, verbose=False, plot=False):

    kmeans = TimeSeriesKMeans(n_clusters=n_clusters,
                              n_init=2,
                              metric="dtw",
                              verbose=verbose,
                              n_jobs=-1,
                              max_iter_barycenter=10,
                              random_state=0)

    y_pred = kmeans.fit_predict(X)
    if plot:
        sz = X.shape[1]

        for yi in range(n_clusters):
            plt.subplot(n_clusters, n_clusters, n_clusters + 1 + yi)
            for xx in X[y_pred == yi]:
                plt.plot(xx.ravel(), "k-", alpha=.2)
            plt.plot(kmeans.cluster_centers_[yi].ravel(), "r-")
            plt.xlim(0, sz)
            plt.ylim(-4, 4)
            plt.text(0.05, 0.85, 'Cluster %d' % (yi + 1),
                     transform=plt.gca().transAxes)
            if yi == 1:
                plt.title("DBA $k$-means")
        plt.show()

    return y_pred


def resample_column(column, sample_size=15, verbose=False):
    start = time.time()
    column_list = []
    for i, v in column.items():
        column_list.append(v)
    end = time.time()
    if verbose:
        print("Time taken to turn column into list: {}".format(end - start))

    start = time.time()
    # Make time series shorter
    X_resampled = TimeSeriesResampler(sz=sample_size).fit_transform(column_list)
    end = time.time()
    if verbose:
        print("Time taken to resample column: {}".format(end - start))

    return X_resampled


def normalise_and_resample_column(column, sample_size=15, verbose=False):

    start = time.time()
    column_list = []
    for i, v in column.items():
        column_list.append(v)
    end = time.time()
    if verbose:
        print("Time taken to turn column into list: {}".format(end - start))

    start = time.time()
    scaler = TimeSeriesScalerMeanVariance()
    normalised_column = scaler.fit_transform(column_list)
    end = time.time()
    if verbose:
        print("Time taken to normalize column: {}".format(end - start))

    start = time.time()
    # Make time series shorter
    X_resampled = TimeSeriesResampler(sz=sample_size).fit_transform(normalised_column)
    end = time.time()
    if verbose:
        print("Time taken to resample column: {}".format(end - start))

    return X_resampled


def kmeans_ts_cluster_column(column, sample_size=15, plot=False, verbose=False):
    if verbose:
        print("")
    start = time.time()
    max_len = column.map(lambda x: len(x)).max()
    end = time.time()
    if verbose:
        print("Time taken to get max time series length: {}".format(end - start))

    start = time.time()
    padded_column = column.apply(lambda x: list(x) + [np.NaN] * (max_len - len(x)))
    end = time.time()
    if verbose:
        print("Time taken to pad column: {}".format(end - start))

    start = time.time()
    column_list = []
    for i, v in padded_column.items():
        column_list.append(v)
    end = time.time()
    if verbose:
        print("Time taken to turn column into list: {}".format(end - start))

    start = time.time()
    scaler = TimeSeriesScalerMeanVariance()
    normalised_column = scaler.fit_transform(column_list)
    end = time.time()
    if verbose:
        print("Time taken to normalize column: {}".format(end - start))

    start = time.time()
    # Make time series shorter
    X_resampled = TimeSeriesResampler(sz=sample_size).fit_transform(normalised_column)
    end = time.time()
    if verbose:
        print("Time taken to resample column: {}".format(end - start))

    start = time.time()
    cluster_labels = time_series_kmeans_clustering(X_resampled, plot=plot)
    end = time.time()
    print("Time taken to cluster series: {}".format(end - start))

    return cluster_labels


def kmeans_ts_plot_inertia(column, sample_size=15):
    inertia_dict = {}

    n_clusters = [i for i in range(2, 22, 5)]

    for n in n_clusters:
        print("n clusters: ", n_clusters)
        print("\n")
        start = time.time()
        max_len = column.map(lambda x: len(x)).max()
        end = time.time()
        print("Time taken to get max time series length: {}".format(end - start))

        start = time.time()
        padded_column = column.apply(lambda x: list(x) + [np.NaN] * (max_len - len(x)))
        end = time.time()
        print("Time taken to pad column: {}".format(end - start))

        start = time.time()
        column_list = []
        for i, v in padded_column.items():
            column_list.append(v)
        end = time.time()
        print("Time taken to turn column into list: {}".format(end - start))

        start = time.time()
        scaler = TimeSeriesScalerMeanVariance()
        normalised_column = scaler.fit_transform(column_list)
        end = time.time()
        print("Time taken to normalize column: {}".format(end - start))

        start = time.time()
        # Make time series shorter
        X_resampled = TimeSeriesResampler(sz=sample_size).fit_transform(normalised_column)
        end = time.time()
        print("Time taken to resample column: {}".format(end - start))

        start = time.time()
        kmeans = TimeSeriesKMeans(n_clusters=n,
                                  n_init=2,
                                  metric="dtw",
                                  verbose=False,
                                  n_jobs=-1,
                                  max_iter_barycenter=10,
                                  random_state=0)

        y_pred = kmeans.fit_predict(X_resampled)

        inertia = kmeans.inertia_
        end = time.time()
        print("Time taken to cluster series: {}".format(end - start))

        inertia_dict[n] = inertia
        print(inertia_dict)
    plt.plot(n_clusters, [inertia_dict[n] for n in n_clusters])
    plt.xlabel("N Clusters")
    plt.ylabel("Inertia")
    plt.title("K-Means Optimal Cluster Number")
    plt.show()


# Gap Statistic for K means
def optimalK(data, nrefs=3, maxClusters=15):
    """
    Calculates KMeans optimal K using Gap Statistic
    Params:
        data: ndarry of shape (n_samples, n_features)
        nrefs: number of sample reference datasets to create
        maxClusters: Maximum number of clusters to test for
    Returns: (gaps, optimalK)
    """
    gaps = np.zeros((len(range(1, maxClusters)),))
    resultsdf = pd.DataFrame({'clusterCount':[], 'gap':[]})
    for gap_index, k in enumerate(range(1, maxClusters)):
        # Holder for reference dispersion results
        refDisps = np.zeros(nrefs)
        # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(nrefs):

            # Create new random reference set
            randomReference = np.random.random_sample(size=data.shape)

            # Fit to it
            km = KMeans(k)
            km.fit(randomReference)

            refDisp = km.inertia_
            refDisps[i] = refDisp
        # Fit cluster to original data and create dispersion
        km = KMeans(k)
        km.fit(data)

        origDisp = km.inertia_
        # Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)
        # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap

        resultsdf = resultsdf.append({'clusterCount':k, 'gap':gap}, ignore_index=True)
    return (gaps.argmax() + 1, resultsdf)


def evaluate_kmeans(cluster_df, max_clusters):

    score_g, df = optimalK(cluster_df, nrefs=5, maxClusters=max_clusters)
    plt.plot(df['clusterCount'], df['gap'], linestyle='--', marker='o', color='b')
    plt.xlabel('K')
    plt.ylabel('Gap Statistic')
    plt.title('Gap Statistic vs. K')

    # Elbow Method for K means
    model = KMeans()
    # k is range of number of clusters.
    visualizer = KElbowVisualizer(model, k=(2,max_clusters), timings= True)
    visualizer.fit(cluster_df)        # Fit data to visualizer
    visualizer.show()        # Finalize and render figure

    # Silhouette Score for K means

    model = KMeans()
    # k is range of number of clusters.
    visualizer = KElbowVisualizer(model, k=(2,max_clusters),metric='silhouette', timings= True)
    visualizer.fit(cluster_df)        # Fit the data to the visualizer
    visualizer.show()        # Finalize and render the figure

    # Calinski Harabasz Score for K means

    model = KMeans()
    # k is range of number of clusters.
    visualizer = KElbowVisualizer(model, k=(2,max_clusters),metric='calinski_harabasz', timings= True)
    visualizer.fit(cluster_df)        # Fit the data to the visualizer
    visualizer.show()        # Finalize and render the figure

    # Davies Bouldin score for K means
    scores = []
    centers = list(range(2,max_clusters))
    for center in centers:
        scores.append(get_kmeans_score(cluster_df, center))

    plt.plot(centers, scores, linestyle='--', marker='o', color='b')
    plt.xlabel('K')
    plt.ylabel('Davies Bouldin score')
    plt.title('Davies Bouldin score vs. K')


def get_kmeans_score(data, center):
    '''
    returns the kmeans score regarding Davies Bouldin for points to centers
    INPUT:
        data - the dataset you want to fit kmeans to
        center - the number of centers you want (the k value)
    OUTPUT:
        score - the Davies Bouldin score for the kmeans model fit to the data
    '''
    #instantiate kmeans
    kmeans = KMeans(n_clusters=center)
    # Then fit the model to your data using the fit method
    model = kmeans.fit_predict(data)

    # Calculate Davies Bouldin score
    score = davies_bouldin_score(data, model)

    return score


def fit_kmeans(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    # Then fit the model to your data using the fit method
    cluster_labels = kmeans.fit_predict(X)
    return cluster_labels


def check_consistency_two_arrays(arr1, arr2):

    arr1_labels = np.unique(arr1)
    arr2_labels = np.unique(arr2)

    arr2_permutations = list(itertools.permutations(arr2_labels))
    consistency_list = list()
    for permutation in arr2_permutations:
        # Dictionary for mapping arr1 labels to arr2 labels
        mapping_dict = {arr1_labels[i]: permutation[i] for i in range(len(arr1_labels))}
        consistency_count = 0
        for i in range(len(arr2)):
            if arr1[i] == mapping_dict[arr2[i]]:
                consistency_count += 1
        consistency = consistency_count / len(arr1)
        consistency_list.append(consistency)

    max_consistency = max(consistency_list)
    print("max consistency:", max_consistency)
    return max_consistency


#cluster radius
def kmeans_distance(data, cx, cy, i_centroid, cluster_labels):
    distances = [np.sqrt((x - cx) ** 2 + (y - cy) ** 2) for (x, y) in data[cluster_labels == i_centroid]]
    return np.mean(distances)
