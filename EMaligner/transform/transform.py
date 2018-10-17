from .utils import AlignerTransformException
from .affine_model import AlignerAffineModel
from .similarity_model import AlignerSimilarityModel
from .polynomial_model import AlignerPolynomial2DTransform


class AlignerTransform(object):

    def __init__(self, name=None, transform=None, fullsize=False, order=2):
        if (name is None) & (transform is None):
            raise AlignerTransformException(
                   'must specify transform name or provide a transform')

        if transform is not None:
            name = transform.__class__.__name__

        # backwards compatibility
        if name == 'affine':
            name = 'AffineModel'
            fullsize = False
        if name == 'affine_fullsize':
            name = 'AffineModel'
            fullsize = True
        if name == 'rigid':
            name = 'SimilarityModel'

        # renderapi-consistent names
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
                    order=order)
        else:
            raise AlignerTransformException(
                    'transform %s not in possible choices:' % name)
