import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig
from matplotlib.colors import ListedColormap

from sklearn.datasets import load_iris

# loads iris dataset 
# (150 rows, 4 columns, in cm)
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)

# colours for the 3 species
species_colours = ListedColormap(["#440154", "#21918c", "red"])

# bigger font 
plt.rcParams["font.size"] = 14


def make_label(name, standardised):
    # standardised data has no units
    if standardised:
        # so the "(cm)" is taken off the label
        return name.replace(" (cm)", "")
    return name


def plot_pairs(d, title, standardised=False):
    # plots each variable against the other three
    # the colour shows which species each point is (data.target is 0, 1 or 2)
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    for ax, (i, j) in zip(axes.flatten(), pairs):
        sc = ax.scatter(d.iloc[:, i], d.iloc[:, j], c=data.target,
                        cmap=species_colours, s=15)
        ax.set_xlabel(make_label(d.columns[i], standardised))
        ax.set_ylabel(make_label(d.columns[j], standardised))
    # legend
    fig.legend(handles=sc.legend_elements()[0], labels=list(data.target_names),
               loc="center right", title="Species")
    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.show()


def plot_hists(d, title, standardised=False):
    #  histogram for each of the 4 variables
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    for ax, column in zip(axes.flatten(), d.columns):
        ax.hist(d[column], bins=20)
        ax.set_xlabel(make_label(column, standardised))
        ax.set_ylabel("Count")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_pc1_vs_pc2(pc1, pc2, title):
    # pc1 vs pc2, coloured by species
    sc = plt.scatter(pc1, pc2, c=data.target, cmap=species_colours, s=15)
    plt.legend(handles=sc.legend_elements()[0], labels=list(data.target_names),
               title="Species", loc="upper left", bbox_to_anchor=(1.02, 1))
    
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# look at the raw data first
print(df)

# plots of the raw data
plot_pairs(df, "Iris data")
plot_hists(df, "Iris data histograms")


def standardise(x):
    # equations 
    mean = np.sum(x) / len(x)                        # mean
    variance = np.sum((x - mean) ** 2) / len(x)      # variance
    sd = np.sqrt(variance)                           # standard deviation
    return (x - mean) / sd                           # standardised values


# apply to every column, so each one ends up with mean 0 and std dev 1
df_std = df.apply(standardise)

plot_pairs(df_std, "Standardised Iris data", standardised=True)
plot_hists(df_std, "Standardised Iris histograms", standardised=True)


def covariance(x, y):
    return np.sum((x - np.mean(x)) * (y - np.mean(y))) / len(x)


def covariance_matrix(d):
    # works out the covariance of every pair of columns
    # stores the answers in a 4x4 numpy array
    n = len(d.columns)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = covariance(d.iloc[:, i], d.iloc[:, j])
    return K


K_std = covariance_matrix(df_std)   # standardised data
K_raw = covariance_matrix(df)       # unstandardised data

print("Covariance matrix (standardised):\n", K_std)
print("Covariance matrix (unstandardised):\n", K_raw)

# for the standardised one the diagonal should all be 1
# and the matrix should be symmetric
print("diagonal is all 1:", np.allclose(np.diag(K_std), 1))
print("matrix is symmetric:", np.allclose(K_std, K_std.T))


# eig gives the eigenvalues and eigenvectors of the covariance matrix
eigenvalues, eigenvectors = eig(K_std)

# eig can return complex numbers with zero imaginary part, only need real part
eigenvalues = eigenvalues.real
eigenvectors = eigenvectors.real

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:", eigenvectors)


def get_magnitudes(eigenvectors):
    # each eigenvector is a column, so sum down the rows (axis=0)
    # all the magnitudes should come out as 1
    return np.sqrt(np.sum(eigenvectors ** 2, axis=0))


print("List of magnitudes of eigenvectors:", get_magnitudes(eigenvectors))


# eigen values to find pca
def sort_eigens(eigenvalues, eigenvectors):
    # .T so each row is one eigenvector
    df_eigen = pd.DataFrame(eigenvectors.T)

    # adds a column for the eigenvalues
    df_eigen['Eigenvalues'] = eigenvalues

    # sorts by eigenvalue, biggest first (biggest = pc1)
    df_eigen.sort_values("Eigenvalues", inplace=True, ascending=False)

    sorted_eigenvalues = np.array(df_eigen['Eigenvalues'])
    # .T puts the eigenvectors back into columns
    sorted_eigenvectors = np.array(df_eigen.drop(columns="Eigenvalues")).T

    return sorted_eigenvalues, sorted_eigenvectors


def reorient_data(d, eigenvectors):
    # turn the dataframe into a numpy array so can use matrix multiplication
    numpy_data = np.array(d)

    # multiply the data by the eigenvectors to get the pca features
    pca_features = np.dot(numpy_data, eigenvectors)

    # back into a dataframe for plotting
    return pd.DataFrame(pca_features, columns=["PC1", "PC2", "PC3", "PC4"])


sorted_eigenvalues, sorted_eigenvectors = sort_eigens(eigenvalues, eigenvectors)

print("Sorted eigenvalues:", sorted_eigenvalues)


# standardised data not raw df
pca_df = reorient_data(df_std, sorted_eigenvectors)

# new data looking at it same as before
plot_pairs(pca_df, "Principal components of the Iris data")
plot_hists(pca_df, "Principal component histograms")
plot_pc1_vs_pc2(pca_df["PC1"], pca_df["PC2"], "Iris data in terms of PC1 and PC2")



# scikit comparison
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

x = df.values                          # the 4 features, same df as first part
x = StandardScaler().fit_transform(x)  # standardise

# pca keeping the first 2 components
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data=principalComponents,
                           columns=['principal component 1', 'principal component 2'])

# the two sets of numbers should match
print("sklearn variance ratio:", pca.explained_variance_ratio_)
print("variance ratio:", sorted_eigenvalues / np.sum(sorted_eigenvalues))

plot_pc1_vs_pc2(principalDf['principal component 1'],
                principalDf['principal component 2'],
                "scikit-learn PCA of Iris data")
