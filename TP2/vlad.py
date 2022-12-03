import numpy as np
import sklearn.cluster as sklc


def vlad(descriptors, vocabulary, use_l2_norm=True, use_sqrt_norm=True):
    ''' Compute the VLAD descriptors of an image.

    @param sifts SIFT descriptors extracted from an image
    @param vocabulary Visual vocabulary
    @param use_l2_norm True to use global L2 normalization, False otherwise (default: True)
    @param use_sqrt_norm True to use square root normlization, False otherwise (default: True)
    @type sifts Array of shape (N, 128) (N = number of descriptors in the image)
    @type vocabulary Numpy array of shape (K, 128)
    @type use_l2_norm Boolean
    @type use_sqrt_norm Boolean
    @return VLAD vector of the image
    @rtype Numpy array of shape (128*K,)
    '''
    vlad = np.zeros(vocabulary.shape, dtype=np.float64)
    quantizer = sklc.KMeans(n_clusters=len(vocabulary), max_iter=1).fit(vocabulary)
    quantizer.cluster_centers_ = vocabulary.copy()
    ws = quantizer.predict(descriptors)

    # compute residuals
    for i in range(len(vlad)):
        if (ws == i).any():
            vlad[i, :] = np.sum(descriptors[ws == i] - quantizer.cluster_centers_[i], axis=0)

    # square root normalization
    if use_sqrt_norm:
        vlad[:] = np.sign(vlad) * np.sqrt(np.abs(vlad))

    vlad = vlad.reshape((vlad.shape[0] * vlad.shape[1],))
    if use_l2_norm:
        vlad[:] = vlad / np.maximum(np.linalg.norm(vlad), 1e-12)

    return vlad
