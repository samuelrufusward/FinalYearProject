from sklearn.decomposition import PCA, KernelPCA, NMF
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from metrics.pyDRMetrics import *


def scale_df(df):
    # define scaler
    scaler = StandardScaler()

    # create copy of DataFrame
    scaled_df = df.copy()

    # created scaled version of DataFrame
    scaled_df = pd.DataFrame(scaler.fit_transform(scaled_df), columns=scaled_df.columns)

    return scaled_df


def pca_reduction(X, n_components=2, plot=True, get_metrics=True):

    # define PCA model to use
    pca = PCA(n_components=n_components)

    # fit PCA model to data
    reduced_data = pca.fit_transform(X)
    if plot:
        plt.scatter(reduced_data[:,0], reduced_data[:,1])
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("PCA Reduction")
        plt.show()

        PC_values = np.arange(pca.n_components_) + 1
        plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
        plt.title('Scree Plot')
        plt.xlabel('Principal Component')
        plt.ylabel('Variance Explained')
        plt.show()

        importance_df = create_importance_dataframe(pca=pca, original_num_df=X)
        print(importance_df)

    if get_metrics:
        Xr = pca.inverse_transform(reduced_data)

        drm = DRMetrics(X, reduced_data, Xr) # construct a DRMetrics object

        print("Qlocal = ", drm.Qlocal) # get Qlocal
        print(drm.report()) # print out the summary

    return reduced_data


def nmf_reduction(X, n_components=2, plot=True, get_metrics=True):

    # define PCA model to use
    nmf = NMF(n_components=n_components)

    # fit PCA model to data
    reduced_data = nmf.fit_transform(X)
    if plot:
        plt.scatter(reduced_data[:,0], reduced_data[:,1])
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("PCA Reduction")
        plt.show()

        # PC_values = np.arange(nmf.n_components_) + 1
        # plt.plot(PC_values, nmf.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
        # plt.title('Scree Plot')
        # plt.xlabel('Principal Component')
        # plt.ylabel('Variance Explained')
        # plt.show()
        #
        # importance_df = create_importance_dataframe(pca=nmf, original_num_df=X)
        # print(importance_df)

    if get_metrics:
        Xr = nmf.inverse_transform(reduced_data)

        drm = DRMetrics(X, reduced_data, Xr) # construct a DRMetrics object

        print("Qlocal = ", drm.Qlocal) # get Qlocal
        print(drm.report()) # print out the summary

    return reduced_data


def create_importance_dataframe(pca, original_num_df):

    # Change pcs components ndarray to a dataframe
    importance_df = pd.DataFrame(pca.components_)

    # Assign columns
    importance_df.columns = original_num_df.columns
    print(original_num_df.columns[0])
    print(len(original_num_df.columns))

    # Change to absolute values
    importance_df = importance_df.apply(np.abs)

    # Transpose
    importance_df = importance_df.transpose()

    # Change column names again

    # First get number of pcs
    num_pcs = importance_df.shape[1]

    # Generate the new column names
    new_columns = [f'PC{i}' for i in range(1, num_pcs + 1)]

    # Now rename
    importance_df.columns = new_columns
    # Return importance df
    return importance_df


def kernel_pca_reduction(X, n_components=4, kernel_method='linear', plot=True, get_metrics=True):
    #kernel_methods = ['linear', 'poly', 'rbf', 'sigmoid', 'cosine']
    # define PCA model to use
    pca = KernelPCA(n_components=n_components, kernel=kernel_method, fit_inverse_transform=get_metrics)
    # fit PCA model to data
    reduced_data = pca.fit_transform(X)
    if plot:
        plt.scatter(reduced_data[:,0], reduced_data[:,1])
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("PCA Reduction")
        plt.show()

    if get_metrics:
        Xr = pca.inverse_transform(reduced_data)

        drm = DRMetrics(X, reduced_data, Xr) # construct a DRMetrics object

        print("Qlocal = ", drm.Qlocal) # get Qlocal
        print(drm.report()) # print out the summary

    return reduced_data