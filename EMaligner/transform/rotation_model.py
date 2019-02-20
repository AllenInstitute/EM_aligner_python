import renderapi
from .utils import (
        AlignerTransformException,
        ptpair_indices,
        arrays_for_tilepair,
        aff_matrix,
        rotate)
import numpy as np
import scipy.sparse as sparse


class AlignerRotationModel(renderapi.transform.AffineModel):

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

        self.DOF_per_tile = 1
        self.nnz_per_row = 2
        self.rows_per_ptmatch = 1
        self.fullsize = True

    def to_solve_vec(self, input_tform):
        if isinstance(input_tform, renderapi.transform.AffineModel):
            vec = np.array([input_tform.rotation])
        elif isinstance(
                input_tform, renderapi.transform.Polynomial2DTransform):
            tmp = renderapi.transform.AffineModel()
            tmp.M = np.array([
                [input_tform.params[0, 1], input_tform.params[0, 2], 0.0],
                [input_tform.params[1, 1], input_tform.params[1, 2], 0.0],
                [0.0, 0.0, 1.0]])
            vec = np.array([tmp.rotation])
        else:
            raise AlignerTransformException(
                    "no method to represent input tform %s in solve as %s" % (
                        input_tform.__class__, self.__class__))

        vec = vec.reshape((vec.size, 1))
        return vec

    def from_solve_vec(self, vec):
        tforms = []
        n = vec.size
        for i in range(n):
            self.M = aff_matrix(vec[i], offs=[0.0, 0.0])
            tforms.append(
                    renderapi.transform.AffineModel(
                        json=self.to_dict()))
        return tforms

    def create_regularization(self, sz, regdict):
        reg = np.ones(sz).astype('float64') * regdict['default_lambda']
        outr = sparse.eye(reg.size, format='csr')
        outr.data = reg
        return outr

    def CSR_from_tilepair(
            self, match, tile_ind1, tile_ind2,
            nmin, nmax, choose_random):
        if np.all(np.array(match['matches']['w']) == 0):
            # zero weights
            return None, None, None, None, None, None

        match_index, stride = ptpair_indices(
                len(match['matches']['q'][0]),
                nmin,
                nmax,
                self.nnz_per_row,
                choose_random)
        if match_index is None:
            # did not meet nmin requirement
            return None, None, None, None, None, None

        p = np.array(match['matches']['p']).transpose()
        q = np.array(match['matches']['q']).transpose()
        pm = p.mean(axis=0)
        qm = q.mean(axis=0)
        pcm = p - pm
        qcm = q - qm
        prcm = np.linalg.norm(pcm, axis=1)
        qrcm = np.linalg.norm(qcm, axis=1)
        # for rotation, both p and q points should be a minimum distance from the
        # COM center, otherwise, the atan2 angles can be high variance
        r_too_small = np.argwhere((prcm < 15) | (qrcm < 15))
        rfilt = np.in1d(match_index, r_too_small)
        match_index = match_index[rfilt == False]
         
        npts = match_index.size
        stride = np.arange(npts) * self.nnz_per_row

        # empty arrays
        data, indices, indptr, weights, b = (
                arrays_for_tilepair(
                        npts,
                        self.rows_per_ptmatch,
                        self.nnz_per_row))

        pangs = np.arctan2(pcm[match_index, 1], pcm[match_index, 0])
        qangs = np.arctan2(qcm[match_index, 1], qcm[match_index, 0])
        newangs = []
        for i in match_index:
            rqp = np.arctan2(pcm[i, 1], pcm[i, 0])
            rxy = rotate(qcm[i], -rqp)
            newang = np.arctan2(rxy[1], rxy[0])
            newangs.append(newang)
        newangs = np.array(newangs)

        # u=x+dx
        data[0 + stride] = 1.0
        data[1 + stride] = -1.0
        b[0: npts, 0] = np.array(newangs)
        uindices = np.array([
            tile_ind1 * self.DOF_per_tile,
            tile_ind2 * self.DOF_per_tile])
        indices[0: npts * self.nnz_per_row] = np.tile(uindices, npts)
        indptr[0: self.rows_per_ptmatch * npts] = \
            np.arange(1, self.rows_per_ptmatch * npts + 1) * \
            self.nnz_per_row
        weights[0: self.rows_per_ptmatch * npts] = \
            np.tile(np.array(
                match['matches']['w'])[match_index],
                self.rows_per_ptmatch)

        return data, indices, indptr, weights, b, npts
