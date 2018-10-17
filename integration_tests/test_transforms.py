import pytest
import renderapi
from EMaligner.transform.transform import AlignerTransform
from EMaligner.transform.affine_model import AlignerAffineModel
from EMaligner.transform.similarity_model import AlignerSimilarityModel
from EMaligner.transform.polynomial_model import AlignerPolynomial2DTransform
from EMaligner.transform.utils import (
        AlignerTransformException,
        ptpair_indices,
        arrays_for_tilepair)
from scipy.sparse import csr_matrix
import numpy as np


def test_aliases():
    t = AlignerTransform(name='affine')
    assert(t.__class__ == AlignerAffineModel)
    assert(not t.fullsize)

    t = AlignerTransform(name='affine_fullsize')
    assert(t.__class__ == AlignerAffineModel)
    assert(t.fullsize)

    t = AlignerTransform(name='rigid')
    assert(t.__class__ == AlignerSimilarityModel)


def test_transform():
    # must specify something
    with pytest.raises(AlignerTransformException):
        t = AlignerTransform()

    # two ways to load affine
    t = AlignerTransform(name='AffineModel')
    assert(t.__class__ == AlignerAffineModel)
    del t
    rt = renderapi.transform.AffineModel()
    t = AlignerTransform(transform=rt)
    assert(t.__class__ == AlignerAffineModel)

    # two ways to load similarity
    t = AlignerTransform(name='SimilarityModel')
    assert(t.__class__ == AlignerSimilarityModel)
    del t
    rt = renderapi.transform.SimilarityModel()
    t = AlignerTransform(transform=rt)
    assert(t.__class__ == AlignerSimilarityModel)

    # two ways to load polynomial
    t = AlignerTransform(name='Polynomial2DTransform')
    assert(t.__class__ == AlignerPolynomial2DTransform)
    del t
    rt = renderapi.transform.Polynomial2DTransform(identity=True)
    t = AlignerTransform(transform=rt)
    assert(t.__class__ == AlignerPolynomial2DTransform)

    # specifying something not real
    with pytest.raises(AlignerTransformException):
        t = AlignerTransform(name='LudicrousModel')


def test_ptpairs():
    # doesn't meet nmin
    m, s = ptpair_indices(10, 15, 200, 20, True)
    assert (m is None) & (s is None)

    # does meet nmin
    m, s = ptpair_indices(50, 15, 200, 20, True)
    assert m.size == 50
    assert s.size == 50
    assert s.max() == (50 - 1) * 20

    # does meet nmin
    m, s = ptpair_indices(50, 15, 200, 20, False)
    assert m.size == 50
    assert s.size == 50
    assert s.max() == (50 - 1) * 20

    # exceed nmax
    m, s = ptpair_indices(50, 15, 40, 20, False)
    assert m.size == 40
    assert s.size == 40
    assert s.max() == (40 - 1) * 20

    # exceed nmax
    m, s = ptpair_indices(50, 15, 40, 20, True)
    assert m.size == 40
    assert s.size == 40
    assert s.max() == (40 - 1) * 20


def test_array_gen():
    data, indices, indptr, weights = arrays_for_tilepair(
            100, 2, 14)
    c = csr_matrix((data, indices, indptr))
    assert c.check_format() is None
    assert weights.size == 200


def example_match(npts):
    match = {}
    match['matches'] = {
            "w": list(np.ones(npts)),
            "p": [list(np.random.randn(npts)), list(np.random.randn(npts))],
            "q": [list(np.random.randn(npts)), list(np.random.randn(npts))]
            }
    return match


def test_affine_model():
    # can't do this
    rt = renderapi.transform.Polynomial2DTransform()
    with pytest.raises(AlignerTransformException):
        t = AlignerAffineModel(transform=rt)

    # check args
    rt = renderapi.transform.AffineModel()
    t = AlignerTransform(transform=rt)
    assert(t.__class__ == AlignerAffineModel)
    assert(not t.fullsize)
    t = AlignerTransform(transform=rt, fullsize=True)
    assert(t.__class__ == AlignerAffineModel)
    assert(t.fullsize)

    # make CSR (fullsize)
    t = AlignerTransform(transform=rt, fullsize=True)
    match = example_match(100)
    data, indices, indptr, weights, npts = t.CSR_from_tilepair(
            match, 1, 2, 5, 500, True)
    indptr = np.insert(indptr, 0, 0)
    c = csr_matrix((data, indices, indptr))
    assert c.check_format() is None
    assert weights.size == 100*t.rows_per_ptmatch
    assert npts == 100

    # make CSR (halfsize)
    t = AlignerTransform(transform=rt, fullsize=False)
    match = example_match(100)
    data, indices, indptr, weights, npts = t.CSR_from_tilepair(
            match, 1, 2, 5, 500, True)
    indptr = np.insert(indptr, 0, 0)
    c = csr_matrix((data, indices, indptr))
    assert c.check_format() is None
    assert weights.size == 100*t.rows_per_ptmatch
    assert npts == 100

    # make CSR zero weights
    t = AlignerTransform(transform=rt, fullsize=False)
    match = example_match(100)
    match['matches']['w'] = list(np.zeros(100*t.rows_per_ptmatch))
    data, indices, indptr, weights, npts = t.CSR_from_tilepair(
            match, 1, 2, 5, 500, True)
    assert data is None
    t = AlignerTransform(transform=rt, fullsize=True)
    data, indices, indptr, weights, npts = t.CSR_from_tilepair(
            match, 1, 2, 5, 500, True)
    assert data is None

    # minimum size
    t = AlignerTransform(transform=rt, fullsize=False)
    match = example_match(100)
    data, indices, indptr, weights, npts = t.CSR_from_tilepair(
            match, 1, 2, 200, 500, True)
    assert data is None
    t = AlignerTransform(transform=rt, fullsize=True)
    data, indices, indptr, weights, npts = t.CSR_from_tilepair(
            match, 1, 2, 200, 500, True)
    assert data is None

    # to vec
    rt = renderapi.transform.AffineModel()
    t = AlignerTransform(transform=rt, fullsize=True)
    v = t.to_solve_vec(rt)
    assert np.all(v == np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).reshape(6, 1))
    t.fullsize = False
    rt = renderapi.transform.AffineModel()
    v = t.to_solve_vec(rt)
    assert np.all(v == np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]))
    rt = renderapi.transform.Polynomial2DTransform(identity=True)
    v = t.to_solve_vec(rt)
    assert np.all(v == np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]))
    rt = renderapi.transform.NonLinearCoordinateTransform()
    with pytest.raises(AlignerTransformException):
        v = t.to_solve_vec(rt)

    # from vec
    vec = np.tile([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 6)
    t.fullsize = True
    tforms = t.from_solve_vec(vec)
    assert len(tforms) == 6
    for rt in tforms:
        assert np.all(np.isclose(
            rt.M, renderapi.transform.AffineModel().M))

    t.fullsize = False
    vec = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    vec = np.concatenate((vec, vec))
    tforms = t.from_solve_vec(vec)
    assert len(tforms) == 2
    for rt in tforms:
        assert np.all(np.isclose(
            rt.M, renderapi.transform.AffineModel().M))

    # reg
    r = t.create_regularization(96, 1.0, 0.1)
    assert np.all(r.data[0::6] == 1.0)
    assert np.all(r.data[1::6] == 1.0)
    assert np.all(r.data[2::6] == 0.1)
    assert np.all(r.data[3::6] == 1.0)
    assert np.all(r.data[4::6] == 1.0)
    assert np.all(r.data[5::6] == 0.1)


def test_similarity_model():
    # can't do this
    rt = renderapi.transform.Polynomial2DTransform()
    with pytest.raises(AlignerTransformException):
        t = AlignerSimilarityModel(transform=rt)

    # check args
    rt = renderapi.transform.SimilarityModel()
    t = AlignerTransform(transform=rt)
    assert(t.__class__ == AlignerSimilarityModel)

    # make CSR
    t = AlignerTransform(transform=rt)
    match = example_match(100)
    data, indices, indptr, weights, npts = t.CSR_from_tilepair(
            match, 1, 2, 5, 500, True)
    indptr = np.insert(indptr, 0, 0)
    c = csr_matrix((data, indices, indptr))
    assert c.check_format() is None
    assert weights.size == 100*t.rows_per_ptmatch
    assert npts == 100

    # make CSR zero weights
    t = AlignerTransform(transform=rt, fullsize=False)
    match = example_match(100)
    match['matches']['w'] = list(np.zeros(100*t.rows_per_ptmatch))
    data, indices, indptr, weights, npts = t.CSR_from_tilepair(
            match, 1, 2, 5, 500, True)
    assert data is None
    t = AlignerTransform(transform=rt, fullsize=True)
    data, indices, indptr, weights, npts = t.CSR_from_tilepair(
            match, 1, 2, 5, 500, True)
    assert data is None

    # minimum size
    t = AlignerTransform(transform=rt, fullsize=False)
    match = example_match(100)
    data, indices, indptr, weights, npts = t.CSR_from_tilepair(
            match, 1, 2, 200, 500, True)
    assert data is None
    t = AlignerTransform(transform=rt, fullsize=True)
    data, indices, indptr, weights, npts = t.CSR_from_tilepair(
            match, 1, 2, 200, 500, True)
    assert data is None

    # to vec
    rt = renderapi.transform.SimilarityModel()
    t = AlignerTransform(transform=rt)
    v = t.to_solve_vec(rt)
    assert np.all(v == np.array([1, 0, 0, 0]).reshape(4, 1))
    rt = renderapi.transform.Polynomial2DTransform(identity=True)
    v = t.to_solve_vec(rt)
    assert np.all(v == np.array([1, 0, 0, 0]).reshape(4, 1))
    rt = renderapi.transform.NonLinearCoordinateTransform()
    with pytest.raises(AlignerTransformException):
        v = t.to_solve_vec(rt)

    # from vec
    vec = np.tile([1.0, 0.0, 0.0, 0.0], 6)
    tforms = t.from_solve_vec(vec)
    assert len(tforms) == 6
    for rt in tforms:
        assert np.all(np.isclose(
            rt.M, renderapi.transform.SimilarityModel().M))

    # reg
    r = t.create_regularization(96, 1.0, 0.1)
    assert np.all(r.data[0::4] == 1.0)
    assert np.all(r.data[1::4] == 1.0)
    assert np.all(r.data[2::4] == 0.1)
    assert np.all(r.data[3::4] == 0.1)


def test_polynomial_model():
    # can't do this
    rt = renderapi.transform.AffineModel()
    with pytest.raises(AlignerTransformException):
        t = AlignerPolynomial2DTransform(transform=rt)

    # check args
    for o in range(4):
        t = AlignerTransform(name="Polynomial2DTransform", order=o)
        assert(t.__class__ == AlignerPolynomial2DTransform)
        assert(t.order == o)

    # make CSR
    for order in range(4):
        n = int((order + 1) * (order + 2) / 2)
        params = np.zeros((2, n))
        rt = renderapi.transform.Polynomial2DTransform(params=params)
        t = AlignerTransform(transform=rt)
        match = example_match(100)
        data, indices, indptr, weights, npts = t.CSR_from_tilepair(
                match, 1, 2, 5, 500, True)
        indptr = np.insert(indptr, 0, 0)
        c = csr_matrix((data, indices, indptr))
        assert c.check_format() is None
        assert weights.size == 100*t.rows_per_ptmatch
        assert npts == 100

    # make CSR zero weights
    t = AlignerTransform(transform=rt)
    match = example_match(100)
    match['matches']['w'] = list(np.zeros(100*t.rows_per_ptmatch))
    data, indices, indptr, weights, npts = t.CSR_from_tilepair(
            match, 1, 2, 5, 500, True)
    assert data is None

    # minimum size
    t = AlignerTransform(transform=rt)
    match = example_match(100)
    data, indices, indptr, weights, npts = t.CSR_from_tilepair(
            match, 1, 2, 200, 500, True)
    assert data is None

    # to vec
    for order in range(4):
        n = int((order + 1) * (order + 2) / 2)
        params = np.zeros((2, n))
        rt = renderapi.transform.Polynomial2DTransform(params=params)
        t = AlignerTransform(transform=rt)
        v = t.to_solve_vec(rt)
        assert np.all(v == np.transpose(params))
        rt = renderapi.transform.AffineModel()
        v = t.to_solve_vec(rt)
        it = renderapi.transform.Polynomial2DTransform(identity=True)
        if order == 0:
            assert(np.all(v == 0))
        else:
            assert np.all(v[0:3, :] == np.transpose(it.params))
        if v.shape[0] > 3:
            assert np.all(v[3:, :] == 0)
        rt = renderapi.transform.NonLinearCoordinateTransform()
        with pytest.raises(AlignerTransformException):
            v = t.to_solve_vec(rt)

    # pass low-order input into a higher-order solve tform
    osolve = 3
    n = int((osolve + 1) * (osolve + 2) / 2)
    params = np.zeros((2, n))
    rt = renderapi.transform.Polynomial2DTransform(params=params)
    t = AlignerTransform(transform=rt)
    assert t.to_solve_vec(rt).shape == (n, 2)

    oin = 2
    n2 = int((oin + 1) * (oin + 2) / 2)
    params = np.zeros((2, n2))
    rt2 = renderapi.transform.Polynomial2DTransform(params=params)
    assert t.to_solve_vec(rt2).shape == (n, 2)

    # try to pass in an uninitialized transform
    n = int((order + 1) * (order + 2) / 2)
    params = np.zeros((2, n))
    rt = renderapi.transform.Polynomial2DTransform(params=params)
    t = AlignerTransform(transform=rt)
    uninit = renderapi.transform.Polynomial2DTransform()
    with pytest.raises(AlignerTransformException):
        v = t.to_solve_vec(uninit)

    # from vec
    for order in range(4):
        n = int((order + 1) * (order + 2) / 2)
        vec = np.zeros((n, 2))
        rt0 = renderapi.transform.Polynomial2DTransform(params=vec)
        t = AlignerTransform(transform=rt0)
        vec = np.concatenate((vec, vec, vec, vec))
        tforms = t.from_solve_vec(vec)
        assert len(tforms) == 4
        for rt in tforms:
            assert np.all(np.isclose(
                np.transpose(rt.params), rt0.params))

    # reg
    for order in range(4):
        n = int((order + 1) * (order + 2) / 2)
        vec = np.zeros((n, 2))
        rt0 = renderapi.transform.Polynomial2DTransform(params=vec)
        t = AlignerTransform(transform=rt0)

        r = t.create_regularization(n * 17, 1.0, 0.1)
        assert np.all(r.data[0::n] == 0.1)
        for j in range(1, n):
            assert np.all(r.data[j::n] == 1.0)
