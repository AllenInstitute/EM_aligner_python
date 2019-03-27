import renderapi
from .utils import (
        AlignerTransformException,
        ptpair_indices,
        arrays_for_tilepair)
import numpy as np
import scipy.sparse as sparse


class AlignerPolynomial2DTransform(renderapi.transform.Polynomial2DTransform):

    def __init__(self, transform=None, order=2):

        if transform is not None:
            if isinstance(
                    transform, renderapi.transform.Polynomial2DTransform):
                self.from_dict(transform.to_dict())
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
                renderapi.transform.Polynomial2DTransform.__init__(
                        self, params=params)
        else:
            params = np.zeros(
                    (2, int((order + 1) * (order + 2) / 2)))
            if order > 0:
                # identity
                params[0, 1] = params[1, 2] = 1.0
            renderapi.transform.Polynomial2DTransform.__init__(
                    self, params=params)

        self.DOF_per_tile = int((self.order + 1) * (self.order + 2) / 2)
        self.nnz_per_row = (self.order + 1) * (self.order + 2)
        self.rows_per_ptmatch = 1

    def to_solve_vec(self):
        vec = np.transpose(self.params)
        return vec

    def from_solve_vec(self, vec):
        n = int((self.order + 1) * (self.order + 2) / 2)
        self.params = np.transpose(vec[0:n, :])
        return n

    def regularization(self, regdict):
        reg = np.ones(self.DOF_per_tile).astype('float64') * regdict['default_lambda']
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

    def CSR_from_tilepair(
            self, match, tile_ind1, tile_ind2,
            nmin, nmax, choose_random):
        if np.all(np.array(match['matches']['w']) == 0):
            # zero weights
            return None, None, None, None, None

        match_index, stride = ptpair_indices(
                len(match['matches']['q'][0]),
                nmin,
                nmax,
                self.nnz_per_row,
                choose_random)
        if match_index is None:
            # did not meet nmin requirement
            return None, None, None, None, None

        npts = match_index.size

        # empty arrays
        data, indices, indptr, weights = (
                arrays_for_tilepair(
                   npts,
                   self.rows_per_ptmatch,
                   self.nnz_per_row))

        px = np.array(match['matches']['p'][0])[match_index]
        py = np.array(match['matches']['p'][1])[match_index]
        qx = np.array(match['matches']['q'][0])[match_index]
        qy = np.array(match['matches']['q'][1])[match_index]

        k = 0
        qoff = int(self.nnz_per_row / 2)
        for j in range(self.order + 1):
            for i in range(j + 1):
                data[k + stride] = px ** (j - i) * py ** i
                data[k + stride + qoff] = -qx ** (j - i) * qy ** i
                k += 1

        ir = np.arange(int(self.nnz_per_row / 2))
        uindices = np.hstack((
            tile_ind1 * self.DOF_per_tile + ir,
            tile_ind2 * self.DOF_per_tile + ir))
        indices[0: npts * self.nnz_per_row] = np.tile(uindices, npts)
        indptr[0: npts] = np.arange(1, npts + 1) * self.nnz_per_row
        weights[0: npts] = np.array(match['matches']['w'])[match_index]

        return data, indices, indptr, weights, npts
