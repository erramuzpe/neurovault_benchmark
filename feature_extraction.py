# From Tungaraza et al. 2013
#
# Feature Vector Creation. We transformed each spatially distinct region into a feature vector by extracting
# the following six properties: the region centroid, the region area, the average activation value for all the voxels
# within that region, the variance of those activation values, the average distance of each voxel within that region
# to the region’s centroid, and the variance of the voxels distance to the region’s centroid.
#
#
# Similarity Measure. At this point, each brain contains a set of spatially distinct regions (or feature vectors)
# that are defined by the properties listed above. The similarity measure between a query brain and the other
# brains in the database is calculated by the Summed Minimum Distance (SMD) between the query brain Q and
# the target brain T. For every feature vector s in Q we calculate the Euclidean distance between s and every
# feature vector r in T and retain the minimum distance. Then we sum the minimum distances and divide the sum
# by the total number N Q of feature vectors in the query brain to obtain a query-to-target score. We perform the
# same procedure in the opposite direction to obtain a target-to-query score. The average of the query-to-target
# score and the target-to-query score is the SMD between the query and the target.
import os
import nibabel as nib
import numpy
from nilearn.image import resample_img

from pybraincompare.mr.transformation import make_resampled_transformation_vector,make_resampled_transformation


# Download statmaps. thresholded/unthreslhoded -> download_statmaps.py


# Feature Vector Creation. We need to create feature vectors that represent the statmaps as better as possible without
# too much information lost. We can start with the method above.



# Calculate pearson correlation from pickle files with brain masked vectors of image values
# def save_voxelwise_pearson_similarity_reduced_representation(pk1, pk2):
#     from neurovault.apps.statmaps.models import Similarity, Comparison
#     import numpy as np
#
#     # We will always calculate Comparison 1 vs 2, never 2 vs 1
#     if pk1 != pk2:
#         try:
#             sorted_images = get_images_by_ordered_id(pk1, pk2)
#         except Http404:
#             # files have been deleted in the meantime
#             return
#         image1 = sorted_images[0]
#         image2 = sorted_images[1]
#         pearson_metric = Similarity.objects.get(similarity_metric="pearson product-moment correlation coefficient",
#                                                 transformation="voxelwise")
#
#         # Make sure we have a transforms for pks in question
#         if not image1.reduced_representation or not os.path.exists(image1.reduced_representation.path):
#             image1 = save_resampled_transformation_single(pk1)  # cannot run this async
#
#         if not image2.reduced_representation or not os.path.exists(image1.reduced_representation.path):
#             image2 = save_resampled_transformation_single(pk2)  # cannot run this async
#
#         # Load image pickles
#         image_vector1 = np.load(image1.reduced_representation.file)
#         image_vector2 = np.load(image2.reduced_representation.file)
#
#         # Calculate binary deletion vector mask (find 0s and nans)
#         mask = make_binary_deletion_vector([image_vector1, image_vector2])
#
#         # Calculate pearson
#         pearson_score = calculate_pairwise_correlation(image_vector1[mask == 1],
#                                                        image_vector2[mask == 1],
#                                                        corr_type="pearson")
#
#         # Only save comparison if is not nan
#         if not numpy.isnan(pearson_score):
#             Comparison.objects.update_or_create(image1=image1, image2=image2,
#                                                 defaults={'similarity_metric': pearson_metric,
#                                                           'similarity_score': pearson_score})
#             return image1.pk, image2.pk, pearson_score
#         else:
#             print "Comparison returned NaN."
#     else:
#         raise Exception("You are trying to compare an image with itself!")
#
# def save_voxelwise_pearson_similarity_resample(pk1, pk2, resample_dim=[4, 4, 4]):
#     from neurovault.apps.statmaps.models import Similarity, Comparison
#
#     # We will always calculate Comparison 1 vs 2, never 2 vs 1
#     if pk1 != pk2:
#         try:
#             sorted_images = get_images_by_ordered_id(pk1, pk2)
#         except Http404:
#             # files have been deleted in the meantime
#             return
#         image1 = sorted_images[0]
#         image2 = sorted_images[1]
#         pearson_metric = Similarity.objects.get(
#             similarity_metric="pearson product-moment correlation coefficient",
#             transformation="voxelwise")
#
#         # Get standard space brain
#         mr_directory = get_data_directory()
#         reference = "%s/MNI152_T1_2mm_brain_mask.nii.gz" % (mr_directory)
#         image_paths = [image.file.path for image in [image1, image2]]
#         images_resamp, _ = resample_images_ref(images=image_paths,
#                                                reference=reference,
#                                                interpolation="continuous",
#                                                resample_dim=resample_dim)
#         # resample_images_ref will "squeeze" images, but we should keep error here for now
#         for image_nii, image_obj in zip(images_resamp, [image1, image2]):
#             if len(numpy.squeeze(image_nii.get_data()).shape) != 3:
#                 raise Exception("Image %s (id=%d) has incorrect number of dimensions %s" % (image_obj.name,
#                                                                                             image_obj.id,
#                                                                                             str(
#                                                                                                 image_nii.get_data().shape)))
#
#         # Calculate correlation only on voxels that are in both maps (not zero, and not nan)
#         image1_res = images_resamp[0]
#         image2_res = images_resamp[1]
#         binary_mask = make_binary_deletion_mask(images_resamp)
#         binary_mask = nib.Nifti1Image(binary_mask, header=image1_res.get_header(), affine=image1_res.get_affine())
#
#         # Will return nan if comparison is not possible
#         pearson_score = calculate_correlation([image1_res, image2_res], mask=binary_mask, corr_type="pearson")
#
#         # Only save comparison if is not nan
#         if not numpy.isnan(pearson_score):
#             Comparison.objects.update_or_create(image1=image1, image2=image2,
#                                                 defaults={'similarity_metric': pearson_metric,
#                                                           'similarity_score': pearson_score})
#
#             return image1.pk, image2.pk, pearson_score
#         else:
#             raise Exception("You are trying to compare an image with itself!")
from pybraincompare.mr.datasets import get_standard_mask
from pybraincompare.compare.mrutils import get_nii_obj
from nilearn.image import resample_img
import nibabel as nib
import numpy
import os


def make_resampled_transformation(nii_obj, resample_dim=[4, 4, 4]):
    nii_obj = get_nii_obj(nii_obj)[0]
    true_zeros = numpy.zeros(nii_obj.shape)  # default data_type is float64
    true_zeros[:] = nii_obj.get_data()
    true_zeros = nib.nifti1.Nifti1Image(true_zeros, affine=nii_obj.get_affine())
    standard = get_standard_mask(voxdim=resample_dim[0])
    true_zeros = resample_img(true_zeros, target_affine=standard.get_affine(),target_shape=standard.shape)
    masked_true_zeros = numpy.zeros(true_zeros.shape)
    masked_true_zeros[standard.get_data() != 0] = true_zeros.get_data()[standard.get_data() != 0]
    true_zeros = nib.nifti1.Nifti1Image(masked_true_zeros, affine=true_zeros.get_affine())
    return true_zeros
def make_resampled_transformation_vector(nii_obj,resample_dim=[4,4,4]):
    resamp_nii = make_resampled_transformation(nii_obj,resample_dim)
    standard = get_standard_mask(voxdim=resample_dim[0])
    return resamp_nii.get_data()[standard.get_data()]



import os
import nibabel as nib
import numpy as np
from nilearn.image import resample_img
from pybraincompare.mr.transformation import make_resampled_transformation_vector, make_resampled_transformation

from sklearn.decomposition import PCA


def save_resampled_transformation_single(file, resample_dim=[4, 4, 4]):
    nii_obj = nib.load(file)   # standard_mask=True is default
    resamp_nii = make_resampled_transformation(nii_obj, resample_dim)
    image_data = resamp_nii.get_data()
    image_vector = make_resampled_transformation_vector(nii_obj, resample_dim)

    pca = PCA(n_components=2)
    pca.fit(image_vector)
    a = pca.score(image_vector)

    print image_data.shape
    # do I have to create a 116380*4 array to do the PCA?
    # 116380 = image_data.shape[0] * image_data.shape[1]*image_data.shape[2]
    # 4 = (x,y,z,v) where v is the value
    size = image_data.shape
    mesh = np.array(np.meshgrid(np.arange(size[0]),np.arange(size[1]),np.arange(size[2]))).T.reshape(-1,3)
    image_spatial_data = [mesh, image_vector]




    # np.save(f, image_vector)
    return


indir = '/ann-benchmarks/data/thres/'

for root, dirs, filenames in os.walk(indir):
    print root
    for file in filenames:
        print 'computing subject ' + file
        file = indir + file
        save_resampled_transformation_single(file)
        # break





# Similarity Measure (distance). This must be implemented within the searching engine. It is not complicated to add a distance
# to some of the ANN libraries proposed (nearpy...), others must be studied.