import renderapi
from .utils import (
        AlignerTransformException,
        ptpair_indices,
        arrays_for_tilepair)
import numpy as np

class AlignerAffineModel(renderapi.transform.AffineModel):
    
    def __init__(self, transform=None, fullsize=False):
        self.fullsize = fullsize

        if transform is not None:
            if isinstance(transform, renderapi.transform.AffineModel):
                self.from_dict(transform.to_dict())
            else:
                raise AlignerTransformException("can't initialize %s with %s" % (
                    self.__class__, transform.__class__))
        else:
            self.from_dict(renderapi.transform.AffineModel().to_dict())

        self.DOF_per_tile = 6
        self.nnz_per_row = 6
        self.rows_per_ptmatch = 1
        if self.fullsize:
            self.rows_per_ptmatch = 2

    def to_solve_vec(self, input_tform):
        if isinstance(input_tform, renderapi.transform.AffineModel):
            vec = np.array([
                input_tform.M[0, 0],
                input_tform.M[0, 1],
                input_tform.M[0, 2],
                input_tform.M[1, 0],
                input_tform.M[1, 1],
                input_tform.M[1, 2]])
        elif isinstance(input_tform, renderapi.transform.Polynomial2DTransform):
            vec = np.array([
                input_tform.params[0, 1],
                input_tform.params[0, 2],
                input_tform.params[0, 0],
                input_tform.params[1, 1],
                input_tform.params[1, 2],
                input_tform.params[1, 0]])
        else:
            raise AlignerTransformException(
                    "no method to represent input tform %s in solve as %s" % (
                        input_tform.__class__, self.__class__))
        if not self.fullsize:
            # split in half for half-size solve
            vec = np.transpose(vec.reshape((2, vec.size/2)))
        else:
            vec = vec.reshape((vec.size, 1))

        return vec


    def CSR_from_tilepair(self, match, tile_ind1, tile_ind2, nmin, nmax, choose_random):
        if self.fullsize:
            return self.CSR_fullsize(match, tile_ind1, tile_ind2, nmin, nmax, choose_random)
        else:
            return self.CSR_halfsize(match, tile_ind1, tile_ind2, nmin, nmax, choose_random)

    def CSR_fullsize(self, match, tile_ind1, tile_ind2, nmin, nmax, choose_random):
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
        data, indices, indptr, weights = \
                arrays_for_tilepair(
                        npts,
                        self.rows_per_ptmatch,
                        self.nnz_per_row)

        # u=ax+by+c
        data[0 + stride] = np.array(match['matches']['p'][0])[match_index]
        data[1 + stride] = np.array(match['matches']['p'][1])[match_index]
        data[2 + stride] = 1.0
        data[3 + stride] = -1.0 * np.array(match['matches']['q'][0])[match_index]
        data[4 + stride] = -1.0 * np.array(match['matches']['q'][1])[match_index]
        data[5 + stride] = -1.0
        uindices = np.hstack((
            tile_ind1 * self.DOF_per_tile+np.array([0, 1, 2]),
            tile_ind2 * self.DOF_per_tile+np.array([0, 1, 2])))
        indices[0:npts * self.nnz_per_row] = np.tile(uindices, npts)
        # v=dx+ey+f
        data[
                (npts * self.nnz_per_row):
                (2 * npts * self.nnz_per_row)] = \
            data[0: npts * self.nnz_per_row]
        indices[npts * self.nnz_per_row:
                2 * npts * self.nnz_per_row] = \
            np.tile(uindices + 3, npts)
        
        # indptr and weights
        indptr[0: 2 * npts] = \
            np.arange(1, 2 * npts + 1) * self.nnz_per_row
        weights[0: 2 * npts] = \
            np.tile(np.array(match['matches']['w'])[match_index], 2)
        
        return data, indices, indptr, weights, npts


    def CSR_halfsize(self, match, tile_ind1, tile_ind2, nmin, nmax, choose_random):
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
        data, indices, indptr, weights = \
                arrays_for_tilepair(
                        npts,
                        self.rows_per_ptmatch,
                        self.nnz_per_row)

        # u=ax+by+c
        data[0 + stride] = np.array(match['matches']['p'][0])[match_index]
        data[1 + stride] = np.array(match['matches']['p'][1])[match_index]
        data[2 + stride] = 1.0
        data[3 + stride] = -1.0 * np.array(match['matches']['q'][0])[match_index]
        data[4 + stride] = -1.0 * np.array(match['matches']['q'][1])[match_index]
        data[5 + stride] = -1.0
        uindices = np.hstack((
            tile_ind1 * self.DOF_per_tile / 2 + np.array([0, 1, 2]),
            tile_ind2 * self.DOF_per_tile / 2 + np.array([0, 1, 2])))
        indices[0: npts * self.nnz_per_row] = np.tile(uindices, npts)
        indptr[0: npts] = np.arange(1, npts + 1) * self.nnz_per_row
        weights[0: npts] = np.array(match['matches']['w'])[match_index]

        return data, indices, indptr, weights, npts
