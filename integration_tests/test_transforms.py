import pytest
import renderapi
from EMaligner.transform import *
from EMaligner.transform.utils import *
from scipy.sparse import csr_matrix

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

#    # two ways to load polynomial
#    t = AlignerTransform(name='Polynomial2DTransform')
#    assert(t.__class__ == AlignerPolynomial2DTransform)
#    del t
#    rt = renderapi.transform.Polynomial2DTransform(identity=True)
#    t = AlignerTransform(transform=rt)
#    assert(t.__class__ == AlignerPolynomial2DTransform)

    # specifying something not real 
    with pytest.raises(AlignerTransformException):
        t = AlignerTransform(name='LudicrousModel')

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
    assert(t.fullsize == False)
    t = AlignerTransform(transform=rt, fullsize=True)
    assert(t.__class__ == AlignerAffineModel)
    assert(t.fullsize == True)


    # make CSR (fullsize)
    t = AlignerTransform(transform=rt, fullsize=True)
    match = example_match(100)
    data, indices, indptr, weights, npts = t.CSR_from_tilepair(
            match, 1, 2, 5, 500, True)
    indptr = np.insert(indptr, 0, 0)
    c = csr_matrix((data, indices, indptr))
    assert c.check_format() is None
    assert weights.size==100*t.rows_per_ptmatch
    assert npts==100

    # make CSR (halfsize)
    t = AlignerTransform(transform=rt, fullsize=False)
    match = example_match(100)
    data, indices, indptr, weights, npts = t.CSR_from_tilepair(
            match, 1, 2, 5, 500, True)
    indptr = np.insert(indptr, 0, 0)
    c = csr_matrix((data, indices, indptr))
    assert c.check_format() is None
    assert weights.size==100*t.rows_per_ptmatch
    assert npts==100

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
    assert np.all(v == np.array([1, 0, 0, 0, 1, 0]).reshape(6, 1))
    t.fullsize = False
    rt = renderapi.transform.AffineModel()
    v = t.to_solve_vec(rt)
    assert np.all(v == np.array([[1, 0], [0, 1],[0, 0]]))
    rt = renderapi.transform.Polynomial2DTransform(identity=True)
    v = t.to_solve_vec(rt)
    assert np.all(v == np.array([[1, 0], [0, 1],[0, 0]]))
    rt = renderapi.transform.NonLinearCoordinateTransform()
    with pytest.raises(AlignerTransformException):
        v = t.to_solve_vec(rt)

    # from vec
    vec = np.tile([1, 0, 0, 0, 1, 0], 6)
    t.fullsize = True
    tforms = t.from_solve_vec(vec)
    assert len(tforms) == 6
    for rt in tforms:
        assert rt == renderapi.transform.AffineModel()

    t.fullsize = False
    vec = np.array([[1, 0], [0, 1],[0, 0]])
    vec = np.concatenate((vec, vec))
    tforms = t.from_solve_vec(vec)
    assert len(tforms) == 2
    for rt in tforms:
        assert rt == renderapi.transform.AffineModel()

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

#def test_polynomial_model():
#    # can't do this
#    rt = renderapi.transform.AffineModel()
#    with pytest.raises(AlignerTransformException):
#        t = AlignerPolynomial2DTransform(transform=rt)
#
#    # check args
#    for o in range(1, 4):
#        t = AlignerTransform(name="Polynomial2DTransform", order=o)
#        assert(t.__class__ == AlignerPolynomial2DTransform)
#        assert(t.order == o)
#
#
