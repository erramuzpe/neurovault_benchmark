from django.core.management.base import BaseCommand, CommandError
from neurovault.apps.statmaps.tests.utils import clearDB
from neurovault.apps.statmaps.models import Comparison, Similarity, User, Collection, Image
from neurovault.apps.statmaps.tests.utils import save_statmap_form
from neurovault.apps.statmaps.tasks import save_resampled_transformation_single
from neurovault.apps.statmaps.utils import get_similar_images

import os, scipy, pickle
import numpy as np
from scipy import stats


def createFeatures(subjects=None, resample_dim=[4, 4, 4]):
    if os.path.isfile('/code/neurovault/apps/statmaps/tests/features'+str(subjects)+str(resample_dim)+'.npy'): #and subjects == None:
        return np.load('/code/neurovault/apps/statmaps/tests/features'+str(subjects)+str(resample_dim)+'.npy').T, \
               pickle.load(open('/code/neurovault/apps/statmaps/tests/dict_feat'+str(subjects)+str(resample_dim)+'.p',"rb" ))
    else:
        clearDB()
        feature_dimension = get_feature_dimension(resample_dim)
        features = np.empty([feature_dimension, subjects]) #4*4*4 = 28549 #16*16*16 = 450
        dict_feat = {}
        u1 = User.objects.create(username='neurovault3')
        for i, file in enumerate(os.listdir('/code/neurovault/apps/statmaps/tests/bench/images/')):
            # print 'Adding subject ' + file
            print i
            randomCollection = Collection(name='random' + file, owner=u1, DOI='10.3389/fninf.2015.00008' + str(i))
            randomCollection.save()
            image = save_statmap_form(image_path=os.path.join('/code/neurovault/apps/statmaps/tests/bench/images/', file),
                                      collection=randomCollection, image_name=file, ignore_file_warning=True)
            if not image.reduced_representation or not os.path.exists(image.reduced_representation.path):
                image = save_resampled_transformation_single(image.pk, resample_dim)
            features[:, i] = np.load(image.reduced_representation.file)
            dict_feat[i] = int(file.split(".")[0])
            if i == subjects-1:
                features[np.isnan(features)] = 0
                np.save('/code/neurovault/apps/statmaps/tests/features'+str(subjects)+str(resample_dim)+'.npy', features)
                pickle.dump(dict_feat,open('/code/neurovault/apps/statmaps/tests/dict_feat'+str(subjects)+str(resample_dim)+'.p',"wb" ))
                return features.T, dict_feat

def get_feature_dimension(resample_dim):
    u1 = User.objects.create(username='dummy'+str(resample_dim))
    for file in os.listdir('/code/neurovault/apps/statmaps/tests/bench/images/'):
        randomCollection = Collection(name='random' + file, owner=u1, DOI='10.3389/fninf.2015.00008' + file)
        randomCollection.save()
        image = save_statmap_form(image_path=os.path.join('/code/neurovault/apps/statmaps/tests/bench/images/', file),
                                  collection=randomCollection, image_name=file, ignore_file_warning=True)
        if not image.reduced_representation or not os.path.exists(image.reduced_representation.path):
            image = save_resampled_transformation_single(image.pk, resample_dim)
        feature = np.load(image.reduced_representation.file)
        dimension = feature.shape[0]
        clearDB()
        break
    return dimension


def get_neurovault_scores(subjects, dict_feat):
    import json, requests
    if os.path.isfile('/code/neurovault/apps/statmaps/tests/dict_scores' + str(subjects) +'.p'):  # and subjects == None:
        return pickle.load(
                   open('/code/neurovault/apps/statmaps/tests/dict_scores' + str(subjects) +'.p',"rb"))
    else:
        scores = {}
        for value in dict_feat.itervalues():
            print "calc scores; value_id", value
            #url = 'http://www.neurovault.com/images/'+ str(value) +'/find_similar/json'
            #url = 'http://127.0.0.1/images/'+ str(value) +'/find_similar/json'
            #resp = requests.get(url=url)
            #data = json.loads(resp.text)
            similar_images = get_similar_images(value, 1000)
            data = similar_images.to_dict("split")
            del data["index"]
            # create a dict of dicts
            image_ids = [p[1] for p in data["data"]]
            corr_values = [p[4] for p in data["data"]]
            # Take our dict_feat, check if it is in the list. If not, delete (we want to compare 1:1)
            for id in image_ids:
                try:
                    idx = dict_feat.values().index(id)
                except ValueError:
                    idx2 = image_ids.index(id)
                    image_ids.remove(id)
                    corr_values.pop(idx2)
            scores[value] = dict(zip(image_ids,corr_values)) # id : corr value
        pickle.dump(scores, open('/code/neurovault/apps/statmaps/tests/dict_scores' + str(subjects) + '.p', "wb"))
        return scores


#######
# Accuracy metric #
#######
def dcg(r):
    """Score is discounted cumulative gain (dcg)"""
    r = np.asfarray(r)[:]
    if r.size:
        return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
    return 0.


class Command(BaseCommand):
    args = '<times_to_run>'
    help = 'bench'

    def handle(self, *args, **options):

        import nearpy, nearpy.hashes, nearpy.distances

        resample_dim_pool = [[16,16,16]]
        subjects = 940
        n_bits_pool = [5,7,9,11,13,15]
        hash_counts_pool = [10,40,60,100]
        metric_pool = ["euclidean"]
        z_score_pool = ["no"]

        for resample_dim in resample_dim_pool:
            features, dict_feat = createFeatures(subjects, resample_dim)
            dict_feat = pickle.load(open('/code/neurovault/apps/statmaps/tests/dict_feat_localhost.p', "rb"))
            features = features[:940, :]
            scores = get_neurovault_scores(940, dict_feat)

            for n_bits in n_bits_pool:
                for hash_counts in hash_counts_pool:
                    for metric in metric_pool:
                        for z_score in z_score_pool:

                            if metric == "euclidean":
                                distance = nearpy.distances.EuclideanDistance()
                            else:
                                distance = nearpy.distances.CosineDistance()

                            hashes = []
                            for k in xrange(hash_counts):
                                nearpy_rbp = nearpy.hashes.RandomBinaryProjections('rbp_%d' % k, n_bits)
                                hashes.append(nearpy_rbp)

                            filter_N = nearpy.filters.NearestFilter(100)

                            nearpy_engine = nearpy.Engine(features.shape[1], lshashes=hashes, distance=distance, vector_filters=[filter_N])

                            if z_score == "yes":
                                for i, x in enumerate(features):
                                    nearpy_engine.store_vector(stats.zscore(x).tolist(), dict_feat[i])
                            else:
                                for i, x in enumerate(features):
                                    nearpy_engine.store_vector(x.tolist(), dict_feat[i])


                            # query
                            query_score = np.zeros(features.shape[0])
                            max_query_score = np.zeros(features.shape[0])
                            size_of_r = np.zeros(features.shape[0])
                            for i in range(features.shape[0]):
                                results = nearpy_engine.neighbours(features[i])
                                try:
                                    ann_idx = zip(*results)[1][1:]
                                except:
                                    print "exception", i

                                img_id = dict_feat[i]
                                real_scores = scores[img_id]

                                r = np.zeros(len(ann_idx))
                                for j, idx in enumerate(ann_idx):
                                    try:
                                        r[j] = (real_scores[idx])
                                    except KeyError:
                                        r[j] = 0

                                sorted_r = np.sort(r)[::-1]
                                query_score[i] = dcg(r)
                                max_query_score[i] = dcg(sorted_r)
                                size_of_r[i] = r.shape[0]

                            print "DCG error score for [r_dim:", resample_dim, ",n_bit:", n_bits, ",hsh_c:", hash_counts, \
                                ",met:", metric, ",z_sc:", z_score, "] =", np.mean(query_score) ,\
                                np.mean(max_query_score) - np.mean(query_score) , np.mean(size_of_r)

                            # text_file = open("/code/neurovault/apps/statmaps/tests/DCG_scores_error.txt", "a")
                            # print >> text_file, "DCG score/error/size for [r_dim:", resample_dim, ",n_bit:", n_bits, ",hsh_c:", hash_counts, \
                            #     ",met:", metric, ",z_sc:", z_score, "] =", np.mean(query_score) ,\
                            #     np.mean(max_query_score) - np.mean(query_score) , np.mean(size_of_r)
                            # text_file.close()

                            del nearpy_engine, hashes




#             for n_bits in n_bits_pool:
#                 for hash_counts in hash_counts_pool:
#                     for metric in metric_pool:
#                         #for z_score in z_score_pool:
#
#                         if metric == "euclidean":
#                             distance = nearpy.distances.EuclideanDistance()
#                         else:
#                             distance = nearpy.distances.CosineDistance()
#
#                         # fit
#                         hashes = []
#                         for k in xrange(hash_counts):
#                             nearpy_rbp = nearpy.hashes.RandomBinaryProjections('rbp_%d' % k, n_bits)
#                             hashes.append(nearpy_rbp)
#                         nearpy_engine = nearpy.Engine(features.shape[1], lshashes=hashes, distance=distance)
#                         for i, x in enumerate(features):
#                             nearpy_engine.store_vector(x.tolist(), dict_feat[i])
#
#                         #query
#                         for i in range(features.shape[0]):
#                             results = nearpy_engine.neighbours(features[i])
#                             #print 'queried', dict_feat[i], 'results', zip(*results)[1]
#
#                         # comparison of results with scores!!
#                         # use of DCG()
#                         # results are sorted, sort scores
#
#                             # to sort:
#                             # sorted(dict1, key=dict1.get)
#                             # or
#                             # for w in sorted(d, key=d.get, reverse=True):
#                             #       print w, d[w]
#
#                         # 2 ways of sorting (corr or abs(corr))
#                         # I will start with abs(corr) since it is best for dcg calc
#
#
#
#
#
#
#
# #results are the N-nearest neighbours! [vector, data_idx, distance]. (for now, distance is NaN)
#
#
# from django.core.management.base import BaseCommand, CommandError
# from neurovault.apps.statmaps.tests.utils import clearDB
# from neurovault.apps.statmaps.models import Comparison, Similarity, User, Collection, Image
# from neurovault.apps.statmaps.tests.utils import save_statmap_form
# from neurovault.apps.statmaps.tasks import save_resampled_transformation_single
#
# import os, scipy, pickle
# import numpy as np
#
# features, dict_feat = np.load('/code/neurovault/apps/statmaps/tests/features_940_16_16_16.npy').T, \
#                       pickle.load(open('/code/neurovault/apps/statmaps/tests/dict_feat.p', "rb"))
# import timeit
# ## NEARPY TEST
# n_bits = 5
# hash_counts = 20
# metric = "euclidean"
# name = 'NearPy(n_bits=%d, hash_counts=%d)' % (n_bits, hash_counts)
# # fiting
# import nearpy, nearpy.hashes, nearpy.distances
# hashes = []
# # doesn't seem like the NearPy code is using the metric??
# for k in xrange(hash_counts):
#     nearpy_rbp = nearpy.hashes.RandomBinaryProjections('rbp_%d' % k, n_bits)
#     hashes.append(nearpy_rbp)
#
# filter_N = nearpy.filters.NearestFilter(100)
#
# nearpy_engine = nearpy.Engine(features.shape[1], distance= nearpy.distances.EuclideanDistance(),
#                               lshashes=hashes,vector_filters=[filter_N])
# #indexing
# t = Timer()
# with t:
#     for i, x in enumerate(features):
#         nearpy_engine.store_vector(x.tolist(), dict_feat[i])
# # querying
# for i in range(features.shape[0]):
#     t = Timer()
#     with t:
#         results = nearpy_engine.neighbours(features[i])
#     print 'queried', dict_feat[i], 'results', zip(*results)[1]
#
#
#
#
#
#
# #######
# # PCA #
# #######
# from sklearn.decomposition import PCA
# features, dict_feat = np.load('/code/neurovault/apps/statmaps/tests/features_940_4_4_4.npy'), \
#                       pickle.load(open('/code/neurovault/apps/statmaps/tests/dict_feat.p', "rb"))
# #features must be n_samples*n_features =  940 * 28549
# number_of_samples = 500
# pca = PCA(n_components = 20) #PCA(n_components = 20) or whatever
# pca.fit(features[:number_of_samples, :])
# print(pca.explained_variance_ratio_[:4])
# #a = pca.transform(features[501,:]).T
# a=pca.transform(features[501,:])
# b=pca.inverse_transform(a)
# EuclideanDistance(b,features[501,:]) #and compare
# import matplotlib.pyplot as plt
# plt.scatter(pca.components_[0,:],pca.components_[1,:])
# plt.show()
#
#
#
# n_bits = 5
# hash_counts = 20
# metric = "euclidean"
# name = 'NearPy(n_bits=%d, hash_counts=%d)' % (n_bits, hash_counts)
# # fiting
# import nearpy, nearpy.hashes, nearpy.distances
# hashes = []
# # doesn't seem like the NearPy code is using the metric??
# for k in xrange(hash_counts):
#     nearpy_rbp = nearpy.hashes.RandomBinaryProjections('rbp_%d' % k, n_bits)
#     hashes.append(nearpy_rbp)
#
# filter_N = nearpy.filters.NearestFilter(100)
#
# nearpy_engine = nearpy.Engine(number_of_samples, distance= nearpy.distances.EuclideanDistance(),
#                               lshashes=hashes,vector_filters=[filter_N])
# #indexing
#
# for i, x in enumerate(features):
#     t = Timer()
#     with t:
#         projection = pca.transform(features[i, :]).T
#         nearpy_engine.store_vector(projection.tolist(), dict_feat[i])
# # querying
# for i in range(features.shape[0]):
#     t = Timer()
#     with t:
#         results = nearpy_engine.neighbours(pca.transform(features[i, :]).T)
#     print 'queried', dict_feat[i], 'results', zip(*results)[1]
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# # Create redis storage adapter
# import redis
# import nearpy, nearpy.hashes, nearpy.distances
#
# n_bits = 5
# hash_counts = 20
#
# redis_object = redis.Redis(host='redis', port=6379, db=0)
# redis_storage = nearpy.storage.RedisStorage(redis_object)
#
# # Get hash config from redis
# config = redis_storage.load_hash_configuration('rbp_0')
#
# if config is None:
#     # Config is not existing, create hash from scratch, with 10 projections
#     lshash = []
#     # doesn't seem like the NearPy code is using the metric??
#     for k in xrange(hash_counts):
#         nearpy_rbp = nearpy.hashes.RandomBinaryProjections('rbp_%d' % k, n_bits)
#         lshash.append(nearpy_rbp)
# else:
#     lshash = []
#     for k in xrange(hash_counts):
#         config = redis_storage.load_hash_configuration('rbp_%d' % k)
#         # Config is existing, create hash with None parameters
#         # Apply configuration loaded from redis
#         lshash_aux = nearpy.hashes.RandomBinaryProjections(None, None)
#         lshash_aux.apply_config(config)
#         lshash.append(lshash_aux)
#
# # Create engine for feature space of 100 dimensions and use our hash.
# # This will set the dimension of the lshash only the first time, not when
# # using the configuration loaded from redis. Use redis storage to store
# # buckets.
# engine = nearpy.Engine(features.shape[1], distance= nearpy.distances.EuclideanDistance(),lshashes=lshash, storage=redis_storage, vector_filters=[nearpy.filters.NearestFilter(100)])
#
# # Do some stuff like indexing or querying with the engine...
# for i, x in enumerate(features[:200,:]):
#     t = Timer()
#     with t:
#         engine.store_vector(x.tolist(), dict_feat[i])
#
# for i in range(10):
#     t = Timer()
#     with t:
#         results = engine.neighbours(features[i])
#     print 'queried', dict_feat[i], 'results', zip(*results)[1]
#
# # Finally store hash configuration in redis for later use
# for k in xrange(hash_counts):
#     redis_storage.store_hash_configuration(lshash[k])
#
#
#
#
#
#
#
#
# ## RPFOREST TEST
# from rpforest import RPForest
# leaf_size = 5
# n_trees = 20
# name = 'RPForest(leaf_size=%d, n_trees=%d)' % (leaf_size, n_trees)
# model = RPForest(leaf_size=leaf_size, no_trees=n_trees)
# #fitting
# features = features.copy(order='C') #something related to Cython error
# model.fit(features)
# model.clear()
# #indexing
# for i, x in enumerate(features):
#     t = Timer()
#     with t:
#         model.index(dict_feat[i], x.tolist())
# #querying
# for i in range(features.shape[0]):
#     t = Timer()
#     with t:
#         results = model.get_candidates(features[i])
#     print 'queried', dict_feat[i], 'results', results
#
#
#
#
# import timeit
# class Timer:
#     def __init__(self, timer=None, disable_gc=False, verbose=True):
#         if timer is None:
#             timer = timeit.default_timer
#         self.timer = timer
#         self.disable_gc = disable_gc
#         self.verbose = verbose
#         self.start = self.end = self.interval = None
#     def __enter__(self):
#         if self.disable_gc:
#             self.gc_state = gc.isenabled()
#             gc.disable()
#         self.start = self.timer()
#         return self
#     def __exit__(self, *args):
#         self.end = self.timer()
#         if self.disable_gc and self.gc_state:
#             gc.enable()
#         self.interval = self.end - self.start
#         if self.verbose:
#             print('time taken: %f seconds' % self.interval)
