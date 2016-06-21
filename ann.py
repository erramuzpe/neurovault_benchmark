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


def createFeatures(subjects=None, resample_dim=[4, 4, 4]):
    clearDB()
    if os.path.isfile('/code/neurovault/apps/statmaps/tests/features.npy') and subjects == None:
        return np.load('/code/neurovault/apps/statmaps/tests/features.npy').T, \
               pickle.load(open('/code/neurovault/apps/statmaps/tests/dict_feat.p',"rb" ))
    else:
        u1 = User.objects.create(username='neurovault3')
        features = np.empty([450, subjects]) #4*4*4 = 28549 #16*16*16 = 450
        dict_feat = {}
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
                np.save('/code/neurovault/apps/statmaps/tests/features.npy', features)
                pickle.dump(dict_feat,open('/code/neurovault/apps/statmaps/tests/dict_feat.p',"wb" ))
                return features.T, dict_feat


class Command(BaseCommand):
    args = '<times_to_run>'
    help = 'bench'

    def handle(self, *args, **options):
        features, dict_feat = createFeatures(940,resample_dim=[16,16,16]) #TODO: pass args to this function

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









from django.core.management.base import BaseCommand, CommandError
from neurovault.apps.statmaps.tests.utils import clearDB
from neurovault.apps.statmaps.models import Comparison, Similarity, User, Collection, Image
from neurovault.apps.statmaps.tests.utils import save_statmap_form
from neurovault.apps.statmaps.tasks import save_resampled_transformation_single

import os, scipy, pickle
import numpy as np

features, dict_feat = np.load('/code/neurovault/apps/statmaps/tests/features_940_16_16_16.npy').T, \
                      pickle.load(open('/code/neurovault/apps/statmaps/tests/dict_feat.p', "rb"))
import timeit
## NEARPY TEST
n_bits = 5
hash_counts = 20
metric = "euclidean"
name = 'NearPy(n_bits=%d, hash_counts=%d)' % (n_bits, hash_counts)
# fiting
import nearpy, nearpy.hashes, nearpy.distances
hashes = []
# doesn't seem like the NearPy code is using the metric??
for k in xrange(hash_counts):
    nearpy_rbp = nearpy.hashes.RandomBinaryProjections('rbp_%d' % k, n_bits)
    hashes.append(nearpy_rbp)

filter_N = nearpy.filters.NearestFilter(100)

nearpy_engine = nearpy.Engine(features.shape[1], distance= nearpy.distances.EuclideanDistance(),
                              lshashes=hashes,vector_filters=[filter_N])
#indexing
t = Timer()
with t:
    for i, x in enumerate(features):
        nearpy_engine.store_vector(x.tolist(), dict_feat[i])
# querying
for i in range(features.shape[0]):
    t = Timer()
    with t:
        results = nearpy_engine.neighbours(features[i])
    print 'queried', dict_feat[i], 'results', zip(*results)[1]












# Create redis storage adapter
import redis
import nearpy, nearpy.hashes, nearpy.distances

n_bits = 5
hash_counts = 20

redis_object = redis.Redis(host='redis', port=6379, db=0)
redis_storage = nearpy.storage.RedisStorage(redis_object)

# Get hash config from redis
config = redis_storage.load_hash_configuration('rbp_0')

if config is None:
    # Config is not existing, create hash from scratch, with 10 projections
    lshash = []
    # doesn't seem like the NearPy code is using the metric??
    for k in xrange(hash_counts):
        nearpy_rbp = nearpy.hashes.RandomBinaryProjections('rbp_%d' % k, n_bits)
        lshash.append(nearpy_rbp)
else:
    lshash = []
    for k in xrange(hash_counts):
        config = redis_storage.load_hash_configuration('rbp_%d' % k)
        # Config is existing, create hash with None parameters
        # Apply configuration loaded from redis
        lshash_aux = nearpy.hashes.RandomBinaryProjections(None, None)
        lshash_aux.apply_config(config)
        lshash.append(lshash_aux)

# Create engine for feature space of 100 dimensions and use our hash.
# This will set the dimension of the lshash only the first time, not when
# using the configuration loaded from redis. Use redis storage to store
# buckets.
engine = nearpy.Engine(features.shape[1], distance= nearpy.distances.EuclideanDistance(),lshashes=lshash, storage=redis_storage, vector_filters=[nearpy.filters.NearestFilter(100)])

# Do some stuff like indexing or querying with the engine...
for i, x in enumerate(features[:200,:]):
    t = Timer()
    with t:
        engine.store_vector(x.tolist(), dict_feat[i])

for i in range(10):
    t = Timer()
    with t:
        results = engine.neighbours(features[i])
    print 'queried', dict_feat[i], 'results', zip(*results)[1]

# Finally store hash configuration in redis for later use
for k in xrange(hash_counts):
    redis_storage.store_hash_configuration(lshash[k])








## RPFOREST TEST
from rpforest import RPForest
leaf_size = 5
n_trees = 20
name = 'RPForest(leaf_size=%d, n_trees=%d)' % (leaf_size, n_trees)
model = RPForest(leaf_size=leaf_size, no_trees=n_trees)
#fitting
features = features.copy(order='C') #something related to Cython error
model.fit(features)
model.clear()
#indexing
for i, x in enumerate(features):
    t = Timer()
    with t:
        model.index(dict_feat[i], x.tolist())
#querying
for i in range(features.shape[0]):
    t = Timer()
    with t:
        results = model.get_candidates(features[i])
    print 'queried', dict_feat[i], 'results', results




import timeit
class Timer:
    def __init__(self, timer=None, disable_gc=False, verbose=True):
        if timer is None:
            timer = timeit.default_timer
        self.timer = timer
        self.disable_gc = disable_gc
        self.verbose = verbose
        self.start = self.end = self.interval = None
    def __enter__(self):
        if self.disable_gc:
            self.gc_state = gc.isenabled()
            gc.disable()
        self.start = self.timer()
        return self
    def __exit__(self, *args):
        self.end = self.timer()
        if self.disable_gc and self.gc_state:
            gc.enable()
        self.interval = self.end - self.start
        if self.verbose:
            print('time taken: %f seconds' % self.interval)
