from django.test import TestCase

from neurovault.apps.statmaps.models import Comparison, Similarity, User, Collection, Image
from neurovault.apps.statmaps.tasks import save_voxelwise_pearson_similarity, save_resampled_transformation_single
from neurovault.apps.statmaps.tests.utils import clearDB, save_statmap_form
from neurovault.apps.statmaps.tests.utils import  save_statmap_form

import os
import gc
import timeit
import numpy as np


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


def run_voxelwise_pearson_similarity(pk1):
    from neurovault.apps.statmaps.models import Image
    from neurovault.apps.statmaps.utils import get_images_to_compare_with

    imgs_pks = get_images_to_compare_with(pk1, for_generation=True)
    if imgs_pks:
        image = Image.objects.get(pk=pk1)
        # added for improved performance
        if not image.reduced_representation or not os.path.exists(image.reduced_representation.path):
            image = save_resampled_transformation_single(pk1)

        # exclude single subject maps from analysis
        for pk in imgs_pks:
            save_voxelwise_pearson_similarity.apply([pk, pk1])


class BenchmarkTestCase(TestCase):

    def setUp(self):
        print "Preparing to test image comparison..."
        clearDB()
        self.app_path = os.path.abspath(os.path.dirname(__file__))

        self.u1 = User.objects.create(username='neurovault')
        comparisonCollection1 = Collection(name='comparisonCollection1', owner=u1,
                                           DOI='10.3389/fninf.2015.00008')
        comparisonCollection1.save()

        save_statmap_form(
            image_path=os.path.join(app_path, 'bench/unthres/0003.nii.gz'),
            collection=comparisonCollection1,
            image_name="image1",
            ignore_file_warning=True)

    def run_voxelwise_pearson_similarity(pk1):
        from neurovault.apps.statmaps.models import Image
        from neurovault.apps.statmaps.utils import get_images_to_compare_with

        imgs_pks = get_images_to_compare_with(pk1, for_generation=True)
        # print imgs_pks

        if imgs_pks:
            image = Image.objects.get(pk=pk1)
            # added for improved performance
            if not image.reduced_representation or not os.path.exists(image.reduced_representation.path):
                image = save_resampled_transformation_single(pk1)

            # exclude single subject maps from analysis
            for pk in imgs_pks:
                save_voxelwise_pearson_similarity.apply([pk, pk1])

    def test_bench_index(self):
        i = 0
        num_files = len(os.listdir(os.path.join(self.app_path, 'bench/unthres/')))
        bench_table = np.zeros(num_files)

        for file in os.listdir(os.path.join(self.app_path, 'bench/unthres/')):
            print 'Adding subject ' + file

            randomCollection = Collection(name='random' + file, owner=self.u1, DOI='10.3389/fninf.2015.00008' + str(i))
            randomCollection.save()

            image = save_statmap_form(
                image_path=os.path.join(self.app_path, 'bench/unthres/', file),
                collection=randomCollection,
                image_name=file,
                ignore_file_warning=True)

            # Similarity scores are generated when statmap_form is saved?
            # comparison = Comparison.objects.filter(image1=self.image1, image2=image)
            # self.assertAlmostEqual(comparison[0].similarity_score, 1.0)
            # print comparison[0].similarity_score

            t = Timer()
            with t:
                run_voxelwise_pearson_similarity(
                    image.pk)  # TODO: change this depending on the indexing function
            # print "Time taken to index", i, " images: ", t.interval

            bench_table[i] = t.interval
            np.save(os.path.join(self.app_path, 'bench/results_test'), bench_table)
            i += 1