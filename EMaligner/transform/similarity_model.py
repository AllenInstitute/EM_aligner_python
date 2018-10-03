import renderapi
from .utils import (
        AlignerTransformException,
        ptpair_indices,
        arrays_for_tilepair)
import numpy as np
import scipy.sparse as sparse


class AlignerSimilarityModel(renderapi.transform.AffineModel):

    def __init__(self, transform=None):

        if transform is not None:
            if isinstance(transform, renderapi.transform.AffineModel):
                self.from_dict(transform.to_dict())
            else:
                raise AlignerTransformException(
                        "can't initialize %s with %s" % (
                            self.__class__, transform.__class__))
        else:
            self.from_dict(renderapi.transform.AffineModel().to_dict())

        self.DOF_per_tile = 4
        self.nnz_per_row = 6
        self.rows_per_ptmatch = 4

    def to_solve_vec(self, input_tform):
        if isinstance(input_tform, renderapi.transform.AffineModel):
            vec = np.array([
                input_tform.M[0, 0],
                input_tform.M[0, 1],
                input_tform.M[0, 2],
                input_tform.M[1, 2]])
        elif isinstance(
                input_tform, renderapi.transform.Polynomial2DTransform):
            vec = np.array([
                input_tform.params[0, 1],
                input_tform.params[0, 2],
                input_tform.params[0, 0],
                input_tform.params[1, 0]])
        else:
            raise AlignerTransformException(
                    "no method to represent input tform %s in solve as %s" % (
                        input_tform.__class__, self.__class__))

        vec = vec.reshape((vec.size, 1))
        return vec

    def from_solve_vec(self, vec):
        tforms = []
        n = int(vec.shape[0] / 4)
        for i in range(n):
            self.M[0, 0] = vec[i * 4 + 0]
            self.M[0, 1] = vec[i * 4 + 1]
            self.M[0, 2] = vec[i * 4 + 2]
            self.M[1, 0] = -vec[i * 4 + 1]
            self.M[1, 1] = vec[i * 4 + 0]
            self.M[1, 2] = vec[i * 4 + 3]
            tforms.append(
                    renderapi.transform.AffineModel(
                        json=self.to_dict()))
        return tforms

    def create_regularization(self, sz, default, transfac):
        reg = np.ones(sz).astype('float64') * default
        reg[2::4] *= transfac
        reg[3::4] *= transfac
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

        # u=ax+by+c
        data[0 + stride] = px
        data[1 + stride] = py
        data[2 + stride] = 1.0
        data[3 + stride] = -1.0 * qx
        data[4 + stride] = -1.0 * qy
        data[5 + stride] = -1.0
        uindices = np.hstack((
            tile_ind1 * self.DOF_per_tile + np.array([0, 1, 2]),
            tile_ind2 * self.DOF_per_tile + np.array([0, 1, 2])))
        indices[0: npts * self.nnz_per_row] = np.tile(uindices, npts)
        # v=-bx+ay+d
        data[0 + stride + npts * self.nnz_per_row] = -1.0 * px
        data[1 + stride + npts * self.nnz_per_row] = py
        data[2 + stride + npts * self.nnz_per_row] = 1.0
        data[3 + stride + npts * self.nnz_per_row] = 1.0 * qx
        data[4 + stride + npts * self.nnz_per_row] = -1.0 * qy
        data[5 + stride + npts * self.nnz_per_row] = -1.0
        vindices = np.hstack((
            tile_ind1 * self.DOF_per_tile + np.array([1, 0, 3]),
            tile_ind2 * self.DOF_per_tile + np.array([1, 0, 3])))
        indices[
                npts*self.nnz_per_row:
                2 * npts * self.nnz_per_row] = np.tile(vindices, npts)
        # du
        data[0 + stride + 2 * npts * self.nnz_per_row] = \
            px - px.mean()
        data[1 + stride + 2 * npts * self.nnz_per_row] = \
            py - py.mean()
        data[2 + stride + 2 * npts * self.nnz_per_row] = \
            0.0
        data[3 + stride + 2 * npts * self.nnz_per_row] = \
            -1.0 * (qx - qx.mean())
        data[4 + stride + 2 * npts * self.nnz_per_row] = \
            -1.0 * (qy - qy.mean())
        data[5 + stride + 2 * npts * self.nnz_per_row] = \
            -0.0
        indices[2 * npts * self.nnz_per_row:
                3 * npts * self.nnz_per_row] = np.tile(uindices, npts)
        # dv
        data[0 + stride + 3 * npts * self.nnz_per_row] = \
            -1.0 * (px - px.mean())
        data[1 + stride + 3 * npts * self.nnz_per_row] = \
            py - py.mean()
        data[2 + stride + 3 * npts * self.nnz_per_row] = \
            0.0
        data[3 + stride + 3 * npts * self.nnz_per_row] = \
            1.0 * (qx - qx.mean())
        data[4 + stride + 3 * npts * self.nnz_per_row] = \
            -1.0 * (qy - qy.mean())
        data[5 + stride + 3 * npts * self.nnz_per_row] = \
            -0.0
        indices[3 * npts * self.nnz_per_row:
                4 * npts * self.nnz_per_row] = np.tile(uindices, npts)

        indptr[0: self.rows_per_ptmatch * npts] = \
            np.arange(1, self.rows_per_ptmatch * npts + 1) * \
            self.nnz_per_row
        weights[0: self.rows_per_ptmatch * npts] = \
            np.tile(np.array(
                match['matches']['w'])[match_index],
                self.rows_per_ptmatch)

        return data, indices, indptr, weights, npts
