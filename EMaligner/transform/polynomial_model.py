import renderapi
from .utils import AlignerTransformException
import numpy as np

class AlignerPolynomial2DTransform(renderapi.transform.Polynomial2DTransform):
    
    def __init__(self, transform=None, order=2, fullsize=False):
        self.order = order
        self.fullsize = fullsize

        if transform is not None:
            if isinstance(transform, renderapi.transform.Polynomial2DTransform):
                self.from_dict(transform.to_dict())
            else:
                raise EMalignerException("can't initialize %s with %s" % (
                    self.__class__, transform.__class__))
        else:
            params = np.zeros((2, (order + 1) * (order + 2) / 2))
            params[0, 1] = params[1, 2] = 1.0
            renderapi.transform.Polynomial2DTransform.__init__(self, params=params)

        self.DOF_per_tile = (self.order + 1) * (self.order + 2)
        self.nnz_per_row = (self.order + 1) * (self.order + 2)
        self.rows_per_ptmatch = 1


