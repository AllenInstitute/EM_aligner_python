import renderapi
import numpy as np
from scipy.sparse import csr_matrix
from scipy.linalg import block_diag
from .utils import aff_matrix, AlignerTransformException
__all__ = ['AlignerRotationModel']


class AlignerRotationModel(renderapi.transform.AffineModel):

    def __init__(self, transform=None, order=2):

        if transform is not None:
            if isinstance(
                    transform, renderapi.transform.AffineModel):
                super(AlignerRotationModel, self).__init__(
                        json=transform.to_dict())
            else:
                raise AlignerTransformException(
                        "can't initialize %s with %s" % (
                            self.__class__, transform.__class__))
        else:
            super(AlignerRotationModel, self).__init__()

        self.DOF_per_tile = 1
        self.nnz_per_row = 2
        self.rows_per_ptmatch = 1

    def to_solve_vec(self):
        """sets solve vector values from transform parameters

        Returns
        -------
        vec : numpy array
            transform parameters in solve form
        """

        return np.array([self.rotation]).reshape(-1, 1)

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
        newr = aff_matrix(vec[0][0], offs=[0.0, 0.0])
        self.M = newr.dot(self.M)
        return 1

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
        return reg

    def block_from_pts(self, pts, w, col_ind, col_max):
        """partial sparse block for a tilepair/match

        Parameters
        ----------
        pts :  numpy array
            N x 1, preprocessed from preprocess()
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

        data = np.ones(pts.size)
        indices = np.ones(pts.size) * col_ind
        indptr = np.arange(pts.size + 1)
        rhs = pts.reshape(-1, 1)

        block = csr_matrix((data, indices, indptr), shape=(pts.size, col_max))
        return block, w, rhs

    @staticmethod
    def preprocess(ppts, qpts, w):
        # center of mass
        pcm = ppts - ppts.mean(axis=0)
        qcm = qpts - qpts.mean(axis=0)

        # points very close to center of mass are noisy
        rfilter = np.argwhere(
                (np.linalg.norm(pcm, axis=1) > 15) &
                (np.linalg.norm(qcm, axis=1) > 15)).flatten()
        pcm = pcm[rfilter]
        qcm = qcm[rfilter]
        w = w[rfilter]

        pangs = np.arctan2(pcm[:, 1], pcm[:, 0])

        # rotate all the q values relative to p
        ams = block_diag(*[aff_matrix(-i) for i in pangs])
        qrot = ams.dot(qcm.flatten()).reshape(-1, 2)

        delta_angs = np.arctan2(qrot[:, 1], qrot[:, 0])

        pa = (-0.5 * delta_angs).reshape(-1, 1)
        qa = (0.5 * delta_angs).reshape(-1, 1)
        return pa, qa, w
