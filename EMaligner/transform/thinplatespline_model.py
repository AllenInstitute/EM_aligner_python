import renderapi
from .utils import AlignerTransformException
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
__all__ = ['AlignerThinPlateSplineTransform']


class AlignerThinPlateSplineTransform(renderapi.transform.ThinPlateSplineTransform):

    def __init__(self, transform=None):

        if transform is not None:
            if isinstance(transform, renderapi.transform.ThinPlateSplineTransform):
                super(AlignerThinPlateSplineTransform, self).__init__(
                        json=transform.to_dict())
            else:
                raise AlignerTransformException(
                        "can't initialize %s with %s" % (
                            self.__class__, transform.__class__))
        else:
            raise AlignerTransformException(
                    "not sure how to iniztialize thin plate spline")

        self.DOF_per_tile = self.nLm
        if self.aMtx is not None:
            self.DOF_per_tile += 3
        self.rows_per_ptmatch = 1

    def to_solve_vec(self):
        """sets solve vector values from transform parameters

        Returns
        -------
        vec : numpy array
            transform parameters in solve form
        """
        vec = self.dMtxDat.transpose()
        if self.aMtx is not None:
            vec = np.vstack((
                self.bVec,
                self.aMtx.transpose(),
                vec))
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
        n0 = 0
        n1 = self.nLm
        if self.aMtx is not None:
            self.bVec = vec[0, :]
            self.aMtx = vec[1:3, :].transpose()
            n0 += 3
            n1 += 3
        self.dMtxDat = vec[n0:n1, :].transpose()
        return n1

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
        n0 = 0
        if self.aMtx is not None:
            reg[0] *= regdict['translation_factor']
            reg[1:3] *= 1.0
            n0 += 3
        reg[n0:] *= regdict['thinplate_factor']
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
        data = cdist(
                pts,
                self.srcPts.transpose(),
                metric='sqeuclidean')
        data *= np.ma.log(np.sqrt(data)).filled(0.0)
        ncol = data.shape[1]
        if self.aMtx is not None:
            ones = np.ones(pts.shape[0]).reshape(-1, 1)
            data = np.hstack((ones, pts, data))
            ncol += 3
        data = data.flatten()
        indices = np.tile(np.arange(ncol) + col_ind, pts.shape[0])
        indptr = np.arange(0, pts.shape[0] + 1) * ncol

        block = csr_matrix(
                (data, indices, indptr),
                shape=(pts.shape[0], col_max))
        rhs = pts
        return block, w, rhs

    @property
    def scale(self):
        src = self.srcPts.transpose()
        dst = self.tform(src)
        a = renderapi.transform.AffineModel()
        a.estimate(src, dst)
        return a.scale

