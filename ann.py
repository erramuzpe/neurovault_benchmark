from django.core.management.base import BaseCommand, CommandError
from neurovault.apps.statmaps.tests.utils import clearDB
from neurovault.apps.statmaps.models import Comparison, Similarity, User, Collection, Image
from neurovault.apps.statmaps.tests.utils import save_statmap_form
from neurovault.apps.statmaps.tasks import save_resampled_transformation_single

import os
import numpy as np


def createFeatures(subjects=None):
    if os.path.isfile('/code/neurovault/apps/statmaps/tests/features.npy') and subjects == None:
        return np.load('/code/neurovault/apps/statmaps/tests/features.npy')
    else:
        u1 = User.objects.create(username='neurovault3')
        i = 1
        features = np.empty([28549, subjects])
        for file in os.listdir('/code/neurovault/apps/statmaps/tests/bench/unthres/'):
            randomCollection = Collection(name='random' + file, owner=u1, DOI='10.3389/fninf.2015.00008' + str(i))
            randomCollection.save()
            image = save_statmap_form(image_path=os.path.join('/code/neurovault/apps/statmaps/tests/bench/unthres/', file),
                                      collection=randomCollection, image_name=file, ignore_file_warning=True)
            if not image.reduced_representation or not os.path.exists(image.reduced_representation.path):
                image = save_resampled_transformation_single(image.pk)
            features[:, i] = np.load(image.reduced_representation.file)
            i += 1
            if i == subjects:
                features[np.isnan(features)] = 0
                np.save('/code/neurovault/apps/statmaps/tests/features.npy', features)
                return features


class Command(BaseCommand):
    args = '<times_to_run>'
    help = 'bench'

    def handle(self, *args, **options):
        clearDB()
        features = createFeatures(5).T #TODO: pass args to this function


        # TODO: build specific build, fit and query functions for each algo

        ## Nearpy
        n_bits = 10
        hash_counts = 5
        metric = "euclidean"
        name = 'NearPy(n_bits=%d, hash_counts=%d)' % (n_bits, hash_counts)
        # fit
        import nearpy, nearpy.hashes, nearpy.distances

        hashes = []

        # doesn't seem like the NearPy code is using the metric??
        for k in xrange(hash_counts):
            nearpy_rbp = nearpy.hashes.RandomBinaryProjections('rbp_%d' % k, n_bits)
            hashes.append(nearpy_rbp)

        nearpy_engine = nearpy.Engine(features.shape[1], lshashes=hashes)

        for i, x in enumerate(features):
            nearpy_engine.store_vector(x.tolist(), i)
        #query
        results = nearpy_engine.neighbours(features[3])
        print results
        #results are the N-nearest neighbours! [vector, data_idx, distance]. (for now, distance is NaN)


        ## LSHF
        metric = 'angular'
        n_estimators=10
        n_candidates=50
        name = 'LSHF(n_est=%d, n_cand=%d)' % (n_estimators, n_candidates)

        # fit
        import sklearn.neighbors
        lshf = sklearn.neighbors.LSHForest(n_estimators=n_estimators, n_candidates=n_candidates)
        features = sklearn.preprocessing.normalize(features, axis=1, norm='l2')
        a = lshf.fit(features)

        # query
        n = 3 # number of neighbours
        feature = sklearn.preprocessing.normalize(features[3], axis=1, norm='l2')[0]
        results = lshf.kneighbors(feature, return_distance=False, n_neighbors=n)[0]
        print results


        # BallTree
        metric ='angular'
        leaf_size=20
        name = 'BallTree(leaf_size=%d)' % leaf_size

        # fit
        import sklearn.neighbors
        features = sklearn.preprocessing.normalize(features, axis=1, norm='l2')
        tree = sklearn.neighbors.BallTree(features, leaf_size=leaf_size)

        # query
        n = 3  # number of neighbours
        feature = sklearn.preprocessing.normalize(features[3], axis=1, norm='l2')[0]
        results = lshf.kneighbors(feature, return_distance=False, n_neighbors=n)[0]
        print results

        feature = sklearn.preprocessing.normalize(features[3], axis=1, norm='l2')[0]
        dist, ind = tree.query(feature, k=n)
        print ind

#
# class KDTree(BaseANN):
#     def __init__(self, metric, leaf_size=20):
#         self.name = 'KDTree(leaf_size=%d)' % leaf_size
#         self._leaf_size = leaf_size
#         self._metric = metric
#
#     def fit(self, X):
#         import sklearn.neighbors
#         if self._metric == 'angular':
#             X = sklearn.preprocessing.normalize(X, axis=1, norm='l2')
#         self._tree = sklearn.neighbors.KDTree(X, leaf_size=self._leaf_size)
#
#     def query(self, v, n):
#         if self._metric == 'angular':
#             v = sklearn.preprocessing.normalize(v, axis=1, norm='l2')[0]
#         dist, ind = self._tree.query(v, k=n)
#         return ind[0]
#
#
# class FLANN(BaseANN):
#     def __init__(self, metric, target_precision):
#         self._target_precision = target_precision
#         self.name = 'FLANN(target_precision=%f)' % target_precision
#         self._metric = metric
#
#     def fit(self, X):
#         import pyflann
#         self._flann = pyflann.FLANN(target_precision=self._target_precision, algorithm='autotuned', log_level='info')
#         if self._metric == 'angular':
#             X = sklearn.preprocessing.normalize(X, axis=1, norm='l2')
#         self._flann.build_index(X)
#
#     def query(self, v, n):
#         if self._metric == 'angular':
#             v = sklearn.preprocessing.normalize(v, axis=1, norm='l2')[0]
#         return self._flann.nn_index(v, n)[0][0]