from django.core.management.base import BaseCommand, CommandError
from neurovault.apps.statmaps.tests.utils import clearDB
from neurovault.apps.statmaps.models import Comparison, Similarity, User, Collection, Image
from neurovault.apps.statmaps.tests.utils import save_statmap_form
from neurovault.apps.statmaps.tasks import save_resampled_transformation_single

import os, scipy, pickle
import numpy as np


def CosineDistance(x, y):
    """
    Computes distance measure between vectors x and y. Returns float.
    """
    if scipy.sparse.issparse(x):
        x = x.toarray().ravel()
        y = y.toarray().ravel()
    return 1.0 - np.dot(x, y)

def EuclideanDistance(x, y):
    """
    Computes distance measure between vectors x and y. Returns float.
    """
    if scipy.sparse.issparse(x):
        return np.linalg.norm((x - y).toarray().ravel())
    else:
        return np.linalg.norm(x - y)

def ManhattanDistance(x, y):
    """
    Computes the Manhattan distance between vectors x and y. Returns float.
    """
    if scipy.sparse.issparse(x):
        return np.sum(np.absolute((x-y).toarray().ravel()))
    else:
        return np.sum(np.absolute(x-y))

def createFeatures(subjects=None):
    clearDB()
    if os.path.isfile('/code/neurovault/apps/statmaps/tests/features.npy') and subjects == None:
        return np.load('/code/neurovault/apps/statmaps/tests/features.npy').T, \
               pickle.load(open('/code/neurovault/apps/statmaps/tests/dict_feat.p',"rb" ))
    else:
        u1 = User.objects.create(username='neurovault3')
        features = np.empty([28549, subjects])
        dict_feat = {}
        for i, file in enumerate(os.listdir('/code/neurovault/apps/statmaps/tests/bench/images/')):
            # print 'Adding subject ' + file
            print i
            randomCollection = Collection(name='random' + file, owner=u1, DOI='10.3389/fninf.2015.00008' + str(i))
            randomCollection.save()
            image = save_statmap_form(image_path=os.path.join('/code/neurovault/apps/statmaps/tests/bench/images/', file),
                                      collection=randomCollection, image_name=file, ignore_file_warning=True)
            if not image.reduced_representation or not os.path.exists(image.reduced_representation.path):
                image = save_resampled_transformation_single(image.pk)
            features[:, i] = np.load(image.reduced_representation.file)
            dict_feat[i] = int(file.split(".")[0])
            if i == subjects-1:
                features[np.isnan(features)] = 0
                np.save('/code/neurovault/apps/statmaps/tests/features.npy', features)
                pickle.dump(dict_feat,open('/code/neurovault/apps/statmaps/tests/dict_feat.p',"wb" ))
                return features.T, dict_feat


class Command(BaseCommand):
    args = '<times_to_run>'
    help = 'bench'

    def handle(self, *args, **options):
        features, dict_feat = createFeatures(940) #TODO: pass args to this function

        # TODO: build specific build, fit and query functions for each algo
        ## Nearpy
        n_bits = 20
        hash_counts = 10
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
            nearpy_engine.store_vector(x.tolist(), dict_feat[i])

        #query
        for i in range(features.shape[0]):
            results = nearpy_engine.neighbours(features[i])
            print 'queried', dict_feat[i], 'results', zip(*results)[1]


#results are the N-nearest neighbours! [vector, data_idx, distance]. (for now, distance is NaN)
#
#
# # testing
# for i, x in enumerate(features[0:5]):
#     for j, y in enumerate(features[0:5]):
#         print i, j
#         print 'Cosine', CosineDistance(y,x)
#         print 'Euclidean', EuclideanDistance(y,x)
#         print 'Manhattan', ManhattanDistance(y,x)
# # seems like euclidean distance is a good aproximator of similarity. statmaps must be in the same range.
# for i, file in enumerate(os.listdir('/code/neurovault/apps/statmaps/tests/bench/unthres/')):
#     print 'subject ' + file
#     if i == 5:
#         break


#
# ## NEARPY TEST
# n_bits = 20
# hash_counts = 10
# metric = "euclidean"
# name = 'NearPy(n_bits=%d, hash_counts=%d)' % (n_bits, hash_counts)
# # fit
# import nearpy, nearpy.hashes, nearpy.distances
#
# hashes = []
# # doesn't seem like the NearPy code is using the metric??
# for k in xrange(hash_counts):
#     nearpy_rbp = nearpy.hashes.RandomBinaryProjections('rbp_%d' % k, n_bits)
#     hashes.append(nearpy_rbp)
#
# nearpy_engine = nearpy.Engine(features.shape[1], lshashes=hashes)
#
#
# index_ann = np.zeros(features.shape[0])
# query_ann = np.zeros(features.shape[0])
# for i, x in enumerate(features):
#     t = Timer()
#     with t:
#         nearpy_engine.store_vector(x.tolist(), dict_feat[i])
#     index_ann[i] = t.interval
# np.save('/code/neurovault/apps/statmaps/tests/results_index_ann', index_ann)
#
# # query
# import io
# for i in range(features.shape[0]):
#     t = Timer()
#     with t:
#         results = nearpy_engine.neighbours(features[i])
#     query_ann[i] = t.interval
#     with io.FileIO("/code/neurovault/apps/statmaps/tests/query_ann.txt", "a") as file:
#         print >> file, 'queried', dict_feat[i], 'results', zip(*results)[1]
#         file.close()
#         #print 'queried', dict_feat[i], 'results', zip(*results)[1]
# np.save('/code/neurovault/apps/statmaps/tests/results_query_ann', query_ann)
#






            # ## LSHF
        # metric = 'angular'
        # n_estimators=10
        # n_candidates=50
        # name = 'LSHF(n_est=%d, n_cand=%d)' % (n_estimators, n_candidates)
        # # fit
        # import sklearn.neighbors
        # lshf = sklearn.neighbors.LSHForest(n_estimators=n_estimators, n_candidates=n_candidates)
        # features = sklearn.preprocessing.normalize(features, axis=1, norm='l2')
        # a = lshf.fit(features)
        # # query
        # n = 3  # number of neighbours
        # feature = sklearn.preprocessing.normalize(features[3], axis=1, norm='l2')[0]
        # results = lshf.kneighbors(feature, return_distance=False, n_neighbors=n)[0]
        # print results
        #
        #
        # ## BallTree
        # metric ='angular'
        # leaf_size=20
        # name = 'BallTree(leaf_size=%d)' % leaf_size
        # # fit
        # import sklearn.neighbors
        # features = sklearn.preprocessing.normalize(features, axis=1, norm='l2')
        # tree = sklearn.neighbors.BallTree(features, leaf_size=leaf_size)
        # # query
        # n = 3  # number of neighbours
        # feature = sklearn.preprocessing.normalize(features[3], axis=1, norm='l2')[0]
        # dist, ind = tree.query(feature, k=n) # gives an array with dicstances and another one with idx
        # print ind
        #
        #
        # ## KDTree(BaseANN):
        # metric = 'angular'
        # leaf_size = 20
        # name = 'KDTree(leaf_size=%d)' % leaf_size
        #
        # # fit
        # import sklearn.neighbors
        # features = sklearn.preprocessing.normalize(features, axis=1, norm='l2')
        # tree = sklearn.neighbors.KDTree(features, leaf_size=leaf_size)
        #
        # # query
        # n = 3  # number of neighbours
        # feature = sklearn.preprocessing.normalize(features[3], axis=1, norm='l2')[0]
        # dist, ind = tree.query(feature, k=n)
        # print ind
        #
        #
        # ## PANNS
        # metric = 'euclidean'
        # n_trees = 10
        # n_candidates = 50
        # name = 'PANNS(n_trees=%d, n_cand=%d)' % (n_trees, n_candidates)
        # # fit
        # import panns
        # panns = panns.PannsIndex(features.shape[1], metric=metric)
        # for feature in features:
        #     panns.add_vector(feature)
        # panns.build(n_trees)
        # # query
        # n = 3  # number of neighbours
        # results = panns.query(features[3], n)
        # print zip(*results)[0]  # returns list of duples (idx, distance)
        #
        # ## FLANN
        # metric = 'angular'
        # target_precision =  0.98
        # name = 'FLANN(target_precision=%f)' % target_precision
        # # fit
        # import pyflann
        # flann = pyflann.FLANN(target_precision=target_precision, algorithm='autotuned',
        #                             log_level='info')
        # features = sklearn.preprocessing.normalize(features, axis=1, norm='l2')
        # flann.build_index(features)
        # # query
        # n = 3  # number of neighbours
        # feature = sklearn.preprocessing.normalize(features[3], axis=1, norm='l2')[0]
        # print flann.nn_index(feature, n)[0][0]  # returns 2 arrays. [[idx]] and [[distance]]
