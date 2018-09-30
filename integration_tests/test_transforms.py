import pytest
import renderapi
from EMaligner.transform import *

def test_transform():
    # must specify something
    with pytest.raises(EMalignerException):
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
    with pytest.raises(EMalignerException):
        t = AlignerTransform(name='LudicrousModel')

def test_affine_model():
    # can't do this
    rt = renderapi.transform.Polynomial2DTransform()
    with pytest.raises(EMalignerException):
        t = AlignerAffineModel(transform=rt)

    # check args
    rt = renderapi.transform.AffineModel()
    t = AlignerTransform(transform=rt)
    assert(t.__class__ == AlignerAffineModel)
    assert(t.fullsize == False)
    t = AlignerTransform(transform=rt, fullsize=True)
    assert(t.__class__ == AlignerAffineModel)
    assert(t.fullsize == True)


def test_similarity_model():
    # can't do this
    rt = renderapi.transform.Polynomial2DTransform()
    with pytest.raises(EMalignerException):
        t = AlignerSimilarityModel(transform=rt)

    # check args
    rt = renderapi.transform.SimilarityModel()
    t = AlignerTransform(transform=rt)
    assert(t.__class__ == AlignerSimilarityModel)


def test_polynomial_model():
    # can't do this
    rt = renderapi.transform.AffineModel()
    with pytest.raises(EMalignerException):
        t = AlignerPolynomial2DTransform(transform=rt)

    # check args
    for o in range(1, 4):
        t = AlignerTransform(name="Polynomial2DTransform", order=o)
        assert(t.__class__ == AlignerPolynomial2DTransform)
        assert(t.order == o)


