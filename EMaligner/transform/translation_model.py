import renderapi
from .utils import AlignerTransformException
import numpy as np
from scipy.sparse import csr_matrix
__all__ = ['AlignerTranslationModel']


class AlignerTranslationModel(renderapi.transform.AffineModel):

    def __init__(self, transform=None):

        if transform is not None:
            if isinstance(transform, renderapi.transform.AffineModel):
                super(AlignerTranslationModel, self).__init__(
                        json=transform.to_dict())
            else:
                raise AlignerTransformException(
                        "can't initialize %s with %s" % (
                            self.__class__, transform.__class__))
        else:
            super(AlignerTranslationModel, self).__init__()

        self.DOF_per_tile = 1
        self.rows_per_ptmatch = 1

    def to_solve_vec(self):
        """sets solve vector values from transform parameters

        Returns
        -------
        vec : numpy array
            transform parameters in solve form
        """
        vec = np.array([
            self.M[0, 2],
            self.M[1, 2]])
        vec = vec.reshape(1, 2)
        return vec

    def from_solve_vec(self, vec):
        """reads values from solution and sets transform parameters

        Parameters
        ----------
        vec : numpy array
            input to this function is sliced so that vec[0] is the
            first relevant value for this transform

        Returns
        -------
        n : int
            number of values read from vec. Used to increment vec slice
            for next transform
        """
        self.M[0, 2] = vec[0, 0]
        self.M[1, 2] = vec[0, 1]
        n = 1
        return n

    def regularization(self, regdict):
        """regularization vector

        Parameters
        ----------
        regdict : dict
           see regularization class in schemas. controls values

        Return
        ------
        reg : numpy array
            array of regularization values of length DOF_per_tile
        """
        reg = np.ones(self.DOF_per_tile).astype('float64') * \
            regdict['default_lambda'] * regdict['translation_factor']
        return reg

    def block_from_pts(self, pts, w, col_ind, col_max):
        """partial sparse block for a tilepair/match

        Parameters
        ----------
        pts :  numpy array
            N x 2, the x, y values of the match (either p or q)
        w : numpy array
            the weights associated with the pts
        col_ind : int
            the starting column index for this tile
        col_max : int
            number of columns in the matrix

        Returns
        -------
        block : scipy.sparse.csr_matrix
            the partial block for this transform
        w : numpy array
            the weights associated with the rows of this block
        """
        data = np.ones(pts.shape[0])
        indices = data * col_ind
        indptr = np.arange(0, pts.shape[0] + 1)

        block = csr_matrix((data, indices, indptr), shape=(pts.shape[0], col_max))
        rhs = pts
        return block, w, rhs
