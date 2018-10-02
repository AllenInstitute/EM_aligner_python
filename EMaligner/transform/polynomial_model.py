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
            else:
                raise AlignerTransformException(
                        "can't initialize %s with %s" % (
                            self.__class__, transform.__class__))
        else:
            params = np.zeros(
                    (2, int((order + 1) * (order + 2) / 2)))
            if order > 0:
                # identity
                params[0, 1] = params[1, 2] = 1.0
            renderapi.transform.Polynomial2DTransform.__init__(
                    self, params=params)

        self.DOF_per_tile = (self.order + 1) * (self.order + 2)
        self.nnz_per_row = (self.order + 1) * (self.order + 2)
        self.rows_per_ptmatch = 1

    def to_solve_vec(self, input_tform):
        n = int((self.order + 1) * (self.order + 2) / 2)
        # fullsize is not implemented, though it is possible, but why?
        vec = np.zeros((n, 2))

        if isinstance(input_tform, renderapi.transform.AffineModel):
            vec[0, 0] = input_tform.M[0, 2]
            vec[0, 1] = input_tform.M[1, 2]
            if self.order > 0:
                # order=0 (translation only) could not hold these
                vec[1, 0] = input_tform.M[0, 0]
                vec[1, 1] = input_tform.M[1, 0]
                vec[2, 0] = input_tform.M[0, 1]
                vec[2, 1] = input_tform.M[1, 1]
        elif isinstance(
                input_tform, renderapi.transform.Polynomial2DTransform):
            try:
                nin = input_tform.params.shape[1]
            except AttributeError:
                raise AlignerTransformException(
                        "input transform "
                        "renderapi.transform.Polynomial2DTransform"
                        "must be initialized to have params attribute")
            if nin < n:
                # leave zeros for higher-order solve
                vec[0:nin, :] = np.transpose(input_tform.params)
            else:
                # copy or truncate to lower-order solve
                vec = np.transpose(input_tform.params[:, 0:n])
        else:
            raise AlignerTransformException(
                    "no method to represent input tform %s in solve as %s" % (
                        input_tform.__class__, self.__class__))
        return vec

    def from_solve_vec(self, vec):
        tforms = []
        n = int((self.order + 1) * (self.order + 2) / 2)
        nt = int(vec.shape[0] / n)
        for i in range(nt):
            params = np.transpose(vec[i * n: (i + 1) * n, :])
            tforms.append(
                    renderapi.transform.Polynomial2DTransform(
                        params=params))
        return tforms

    def create_regularization(self, sz, default, transfac):
        reg = np.ones(sz).astype('float64') * default
        n = int((self.order + 1) * (self.order + 2) / 2)
        reg[0::n] *= transfac
        outr = sparse.eye(reg.size, format='csr')
        outr.data = reg
        return outr

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
            tile_ind1 * self.DOF_per_tile / 2 + ir,
            tile_ind2 * self.DOF_per_tile / 2 + ir))
        indices[0: npts * self.nnz_per_row] = np.tile(uindices, npts)
        indptr[0: npts] = np.arange(1, npts + 1) * self.nnz_per_row
        weights[0: npts] = np.array(match['matches']['w'])[match_index]

        return data, indices, indptr, weights, npts
