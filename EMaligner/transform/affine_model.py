import renderapi
from .utils import AlignerTransformException
import numpy as np
from scipy.sparse import csr_matrix
__all__ = ['AlignerAffineModel']


class AlignerAffineModel(renderapi.transform.AffineModel):

    def __init__(self, transform=None, fullsize=False):
        self.fullsize = fullsize

        if transform is not None:
            if isinstance(transform, renderapi.transform.AffineModel):
                super(AlignerAffineModel, self).__init__(
                        json=transform.to_dict())
            else:
                raise AlignerTransformException(
                        "can't initialize %s with %s" % (
                            self.__class__, transform.__class__))
        else:
            super(AlignerAffineModel, self).__init__()

        self.DOF_per_tile = 3
        self.nnz_per_row = 6
        self.rows_per_ptmatch = 1
        if self.fullsize:
            self.DOF_per_tile = 6
            self.rows_per_ptmatch = 2

    def to_solve_vec(self):
        """sets solve vector values from transform parameters

        Returns
        -------
        vec : numpy array
            shape depends on fullsize
        """

        vec = np.array([
            self.M[0, 0],
            self.M[0, 1],
            self.M[0, 2],
            self.M[1, 0],
            self.M[1, 1],
            self.M[1, 2]])
        if not self.fullsize:
            # split in half for half-size solve
            # transpose into Nx2
            vec = np.transpose(vec.reshape((2, int(vec.size/2))))
        else:
            vec = vec.reshape((vec.size, 1))
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
        vsh = vec.shape
        if vsh[1] == 1:
            self.M[0, 0] = vec[0]
            self.M[0, 1] = vec[1]
            self.M[0, 2] = vec[2]
            self.M[1, 0] = vec[3]
            self.M[1, 1] = vec[4]
            self.M[1, 2] = vec[5]
            n = 6
        else:
            self.M[0, 0] = vec[0, 0]
            self.M[0, 1] = vec[1, 0]
            self.M[0, 2] = vec[2, 0]
            self.M[1, 0] = vec[0, 1]
            self.M[1, 1] = vec[1, 1]
            self.M[1, 2] = vec[2, 1]
            n = 3
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
            regdict['default_lambda']
        reg[2::3] *= regdict['translation_factor']

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
        if self.fullsize:
            return self.block_from_pts_fullsize(pts, w, col_ind, col_max)
        else:
            return self.block_from_pts_halfsize(pts, w, col_ind, col_max)

    def block_from_pts_fullsize(self, pts, w, col_ind, col_max):
        data = np.hstack((pts, np.ones((pts.shape[0], 1)))).flatten()
        data = np.concatenate((data, data))

        i0 = col_ind + np.arange(3)
        indices = np.hstack((
            np.tile(i0, pts.shape[0]), np.tile(i0 + 3, pts.shape[0])))

        nrow = pts.shape[0] * 2
        indptr = np.arange(0, nrow + 1) * 3

        block = csr_matrix((data, indices, indptr), shape=(nrow, col_max))
        return block, np.hstack((w, w))

    def block_from_pts_halfsize(self, pts, w, col_ind, col_max):
        nrow = pts.shape[0]
        data = np.hstack((pts, np.ones((nrow, 1)))).flatten()

        i0 = col_ind + np.arange(3)
        indices = np.tile(i0, nrow)

        indptr = np.arange(0, nrow + 1) * 3

        block = csr_matrix((data, indices, indptr), shape=(nrow, col_max))
        return block, w
