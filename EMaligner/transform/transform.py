import renderapi
from .utils import AlignerTransformException
from .affine_model import AlignerAffineModel
from .similarity_model import AlignerSimilarityModel
from .polynomial_model import AlignerPolynomial2DTransform

#__all__ = [
#        'determine_ptpair_indices',
#        'create_array_for_tilepair',
#        'AlignerTransform']

#def determine_ptpair_indices(npts_in, nmin, nmax, nnz, choose_random):
#    # some criteria for returning nothing
#    if (npts < nmin):
#        return None, None
#
#    npts = npts_in
#    # determine number of points
#    if npts > nmax:
#        npts = nmax
#
#    # random subset
#    if choose_random:
#        a = np.arange(npts_in)
#        np.random.shuffle(a)
#        match_index = a[0:npts]
#    else:
#        match_index = np.arange(npts)
#    stride = np.arange(npts) * nnz
#
#    return match_index, stride
#
#
#def create_arrays_for_tilepair(npts, rows_per_ptmatch, nnz_per_row):
#    nd = npts * rows_per_ptmatch * nnz_per_row
#    ni = npts * rows_per_ptmatch
#    data = np.zeros(nd).astype('float64')
#    indices = np.zeros(nd).astype('int64')
#    indptr = np.zeros(ni)
#    weights = np.zeros(ni)
#    return data, indices, indptr, weights


class AlignerTransform(object):
    
    def __init__(self, name=None, transform=None, fullsize=False, order=2):
        if (name is None) & (transform is None):
           raise AlignerTransformException(
                   'must specify transform name or provide a transform')

        if transform is not None:
            name = transform.__class__.__name__

        if (name == 'AffineModel'):
            self.__class__ = AlignerAffineModel
            AlignerAffineModel.__init__(
                    self, transform=transform, fullsize=fullsize)
        elif (name == 'SimilarityModel'):
            self.__class__ = AlignerSimilarityModel
            AlignerSimilarityModel.__init__(self, transform=transform)
        elif (name == 'Polynomial2DTransform'):
            self.__class__ = AlignerPolynomial2DTransform
            AlignerPolynomial2DTransform.__init__(
                    self, transform=transform,
                    fullsize=fullsize, order=order)
        else:
            raise AlignerTransformException(
                    'transform %s not in possible choices:' % name)


