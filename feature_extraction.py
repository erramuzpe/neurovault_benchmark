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


# Download statmaps. thresholded/unthreslhoded -> download_statmaps.py


# Feature Vector Creation. We need to create feature vectors that represent the statmaps as better as possible without
# too much information lost. We can start with the method above.


# Similarity Measure (distance). This must be implemented within the searching engine. It is not complicated to add a distance
# to some of the ANN libraries proposed (nearpy...), others must be studied.