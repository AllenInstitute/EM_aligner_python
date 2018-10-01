import numpy as np


class AlignerTransformException(Exception):
    """Exception raised when there is a \
            problem creating a mesh lens correction"""
    pass


def ptpair_indices(npts_in, nmin, nmax, nnz, choose_random):
    npts = npts_in

    # some criteria for returning nothing
    if (npts < nmin):
        return None, None

    # determine number of points
    if npts > nmax:
        npts = nmax

    # random subset
    if choose_random:
        a = np.arange(npts_in)
        np.random.shuffle(a)
        match_index = a[0:npts]
    else:
        match_index = np.arange(npts)
    stride = np.arange(npts) * nnz

    return match_index, stride


def arrays_for_tilepair(npts, rows_per_ptmatch, nnz_per_row):
    nd = npts * rows_per_ptmatch * nnz_per_row
    ni = npts * rows_per_ptmatch
    data = np.zeros(nd).astype('float64')
    indices = np.zeros(nd).astype('int64')
    indptr = np.zeros(ni)
    weights = np.zeros(ni)
    return data, indices, indptr, weights
