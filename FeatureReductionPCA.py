import numpy as np
from   sklearn.decomposition import PCA
from PandasInput import pandaUtils
import sklearn.decomposition as skdc
import sklearn

def getDimensinalityPCAReducer(components=None):

    drPCA = PCA(n_components=components)
    return drPCA

def reduceDimensionsWithPCA(features: np.ndarray, components=None, \
                    verbose=False)-> (np.ndarray, skdc.PCA):
    cmps = components
    if  components is None:
        cmps = features.shape[1]

    cmps = min(cmps, features.shape[1])

    pca = PCA(n_components=cmps)
    features_reduced = pca.fit_transform(features)

    if  verbose:
        print("e1228 explained variance ratios (%d components)\n" % cmps, pca.explained_variance_ratio_)
        print("e1228 singular values (%d components)\n" % cmps, pca.singular_values_)

    return features_reduced, pca

if  __name__ == '__main__':

    def doPCAOnFeatures(features, components=2):
        pca = PCA(n_components=components)
        pca.fit(features)

        print("e1228 explained variance ratios (%d components)\n" % components, pca.explained_variance_ratio_)
        print("e1228 singular values (%d components)\n" % components, pca.singular_values_)

        features_transformed = pca.transform(features)

        return features_transformed


    print("e1102 skilern=", sklearn.__version__)

    fileName = "/Volumes/DATA_1TB/Safary_Downloads/ml_data.csv"
    features, labels = pandaUtils.readAndPrepareData(fileName, ['Unnamed: 0', 'ID'], ['mpr', 'nux'], 'outputs_class')

    ff6,_ = reduceDimensionsWithPCA(features, components=6, verbose=True)

    tt6 = doPCAOnFeatures(features, components=6)
    tt2 = doPCAOnFeatures(features, components=2)
    tt1 = doPCAOnFeatures(features, components=1)

    pca = PCA(n_components=6)
    pca.fit(features)
    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)

    pca = PCA(n_components=2, svd_solver='full')
    pca.fit(features)
    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)

    pca = PCA(n_components=1, svd_solver='arpack')
    pca.fit(features)
    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)

    pass


