from .utils import AlignerTransformException
from .affine_model import AlignerAffineModel
from .similarity_model import AlignerSimilarityModel
from .polynomial_model import AlignerPolynomial2DTransform
from .rotation_model import AlignerRotationModel
from .translation_model import AlignerTranslationModel
from .thinplatespline_model import AlignerThinPlateSplineModel
__all__ = ['AlignerTransform']


class AlignerTransform(object):

    def __init__(self, name=None, transform=None, fullsize=False, order=2):
        if (name is None):
            raise AlignerTransformException(
                   'must specify transform name')

        # backwards compatibility
        if name == 'affine':
            name = 'AffineModel'
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
        elif (name == 'RotationModel'):
            self.__class__ = AlignerRotationModel
            AlignerRotationModel.__init__(self, transform=transform)
        elif (name == 'TranslationModel'):
            self.__class__ = AlignerTranslationModel
            AlignerTranslationModel.__init__(self, transform=transform)
        elif (name == 'Polynomial2DTransform'):
            self.__class__ = AlignerPolynomial2DTransform
            AlignerPolynomial2DTransform.__init__(
                    self, transform=transform,
                    order=order)
        elif (name == 'ThinPlateSplineTransform'):
            self.__class__ = AlignerThinPlateSplineModel
            AlignerThinPlateSplineModel.__init__(
                    self, transform=transform)
        else:
            raise AlignerTransformException(
                    'transform %s not in possible choices:' % name)
