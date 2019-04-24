import renderapi
import numpy as np
from scipy.sparse import csr_matrix
__all__ = ['AlignerPolynomial2DTransform']


class AlignerPolynomial2DTransform(renderapi.transform.Polynomial2DTransform):

    def __init__(self, transform=None, order=2):

        if transform is not None:
            if isinstance(
                    transform, renderapi.transform.Polynomial2DTransform):
                super(AlignerPolynomial2DTransform, self).__init__(
                        json=transform.to_dict())
            elif isinstance(
                    transform, renderapi.transform.AffineModel):
                params = np.zeros(
                        (2, int((order + 1) * (order + 2) / 2)))
                params[0, 0] = transform.B0
                params[1, 0] = transform.B1
                if order >= 1:
                    params[0, 1] = transform.M00
                    params[0, 2] = transform.M01
                    params[1, 1] = transform.M10
                    params[1, 2] = transform.M11
                super(AlignerPolynomial2DTransform, self).__init__(
                        params=params)
        else:
            params = np.zeros(
                    (2, int((order + 1) * (order + 2) / 2)))
            if order > 0:
                # identity
                params[0, 1] = params[1, 2] = 1.0
            super(AlignerPolynomial2DTransform, self).__init__(
                    params=params)

        self.DOF_per_tile = int((self.order + 1) * (self.order + 2) / 2)
        self.nnz_per_row = (self.order + 1) * (self.order + 2)
        self.rows_per_ptmatch = 1

    def to_solve_vec(self):
        """sets solve vector values from transform parameters

        Returns
        -------
        vec : numpy array
            transform parameters in solve form
        """

        vec = np.transpose(self.params)
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

        n = int((self.order + 1) * (self.order + 2) / 2)
        self.params = np.transpose(vec[0:n, :])
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
        n = int((self.order + 1) * (self.order + 2) / 2)
        if regdict['poly_factors'] is None:
            reg[0] *= regdict['translation_factor']
        else:
            ni = 0
            for i in range(self.order + 1):
                for j in range(i + 1):
                    reg[ni::n] *= regdict['poly_factors'][i]
                    ni += 1
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

        px = pts[:, 0]
        py = pts[:, 1]
        npts = px.size
        cols = []
        for j in range(self.order + 1):
            for i in range(j + 1):
                cols.append(px ** (j - i) * py ** i)

        data = np.hstack(cols).flatten()
        indices = np.tile(np.arange(self.DOF_per_tile) + col_ind, npts)
        indptr = np.arange(0, npts + 1) * self.DOF_per_tile

        block = csr_matrix((data, indices, indptr), shape=(npts, col_max))

        return block, w
