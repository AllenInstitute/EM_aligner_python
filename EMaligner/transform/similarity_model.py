import renderapi
from ..utils import EMalignerException

class AlignerSimilarityModel(renderapi.transform.AffineModel):
    
    def __init__(self, transform=None):

        if transform is not None:
            if isinstance(transform, renderapi.transform.AffineModel):
                self.from_dict(transform.to_dict())
            else:
                raise EMalignerException("can't initialize %s with %s" % (
                    self.__class__, transform.__class__))
        else:
            self.from_dict(renderapi.transform.AffineModel().to_dict())

