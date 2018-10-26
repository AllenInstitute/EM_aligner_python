import pytest
import renderapi
from test_data import (render_params,
                       montage_raw_tilespecs_json,
                       montage_parameters)
from EMaligner import EMaligner
import json
from marshmallow.exceptions import ValidationError
import copy

# https://www.peterbe.com/plog/be-careful-with-using-dict-to-create-a-copy

FILE_PMS = './integration_tests/test_files/montage_pointmatches.json'


@pytest.fixture(scope='module')
def render():
    render = renderapi.connect(**render_params)
    return render


@pytest.fixture(scope='module')
def raw_stack(render):
    test_raw_stack = 'input_raw_stack'
    tilespecs = [
            renderapi.tilespec.TileSpec(json=d)
            for d in montage_raw_tilespecs_json]
    renderapi.stack.create_stack(test_raw_stack, render=render)
    renderapi.client.import_tilespecs(test_raw_stack, tilespecs, render=render)
    renderapi.stack.set_stack_state(test_raw_stack, 'COMPLETE', render=render)
    yield test_raw_stack


@pytest.fixture(scope='function')
def loading_raw_stack(render):
    test_raw_stack = 'input_raw_stack_loading'
    tilespecs = [
            renderapi.tilespec.TileSpec(json=d)
            for d in montage_raw_tilespecs_json]
    renderapi.stack.create_stack(test_raw_stack, render=render)
    renderapi.client.import_tilespecs(test_raw_stack, tilespecs, render=render)
    yield test_raw_stack


@pytest.fixture(scope='module')
def montage_pointmatches(render):
    test_montage_collection = 'montage_collection'
    pms_from_json = []
    with open(FILE_PMS, 'r') as f:
        pms_from_json = json.load(f)

    renderapi.pointmatch.import_matches(
            test_montage_collection, pms_from_json, render=render)
    yield test_montage_collection


@pytest.fixture(scope='module')
def montage_pointmatches_weighted(render):
    test_montage_collection2 = 'montage_collection2'
    pms_from_json = []
    with open(FILE_PMS, 'r') as f:
        pms_from_json = json.load(f)
    n = len(pms_from_json[0]['matches']['w'])
    pms_from_json[0]['matches']['w'] = [0.0 for i in range(n)]

    renderapi.pointmatch.import_matches(
            test_montage_collection2, pms_from_json, render=render)
    yield test_montage_collection2


@pytest.mark.parametrize("stack_state", ["COMPLETE", "LOADING"])
def test_weighted(
        render, montage_pointmatches_weighted, loading_raw_stack,
        stack_state, tmpdir):
    renderapi.stack.set_stack_state(
            loading_raw_stack, stack_state, render=render)
    p = copy.deepcopy(montage_parameters)
    p['input_stack']['name'] = loading_raw_stack
    p['pointmatch']['name'] = montage_pointmatches_weighted
    mod = EMaligner.EMaligner(input_data=p, args=[])
    mod.run()
    assert mod.results['precision'] < 1e-7
    assert mod.results['error'] < 200


def one_solve(parameters, tf, fullsize=False, order=2,
              precision=1e-7, error=200):
    p = dict(parameters)
    p['output_stack']['name'] = p['input_stack']['name'] + 'solved_' + tf
    p['transformation'] = tf
    p['fullsize'] = fullsize
    p['poly_order'] = order
    mod = EMaligner.EMaligner(input_data=p, args=[])
    mod.run()
    assert mod.results['precision'] < precision
    assert mod.results['error'] < error


def test_different_transforms(
        render, montage_pointmatches, loading_raw_stack, tmpdir):
    p = copy.deepcopy(montage_parameters)
    p['input_stack']['name'] = loading_raw_stack
    p['pointmatch']['name'] = montage_pointmatches

    one_solve(p, 'AffineModel', fullsize=False)
    one_solve(p, 'AffineModel', fullsize=True)
    one_solve(p, 'SimilarityModel')
    one_solve(
            p, 'Polynomial2DTransform',
            order=3, precision=0.5)
    one_solve(
            p, 'Polynomial2DTransform',
            order=2, precision=1e-4)
    one_solve(p, 'Polynomial2DTransform', order=1)
    one_solve(p, 'Polynomial2DTransform', order=0)

    p = copy.deepcopy(montage_parameters)
    p['input_stack']['name'] = loading_raw_stack
    p['pointmatch']['name'] = montage_pointmatches
    p['regularization'] = {
            'default_lambda': 1000.0,
            'translation_factor': 1e-5,
            'poly_factors': [1e-5, 1000.0, 1e6]}
    one_solve(p, 'Polynomial2DTransform', order=2, precision=1e-4)


def test_poly_validation():
    p = copy.deepcopy(montage_parameters)
    p['regularization'] = {
            'default_lambda': 1000.0,
            'translation_factor': 1e-5,
            'poly_factors': [1e-5, 1000.0, 1e6, 1e3]}
    p['output_stack']['name'] = p['input_stack']['name'] + 'solved_' + 'poly_validate'
    p['transformation'] = 'Polynomial2DTransform'
    p['poly_order'] = 2
    with pytest.raises(ValidationError):
        # because poly_factors should be length 3
        mod = EMaligner.EMaligner(input_data=p, args=[])


@pytest.mark.parametrize("stack_state", ["COMPLETE", "LOADING"])
def test_first_test(
        render, montage_pointmatches,
        loading_raw_stack, stack_state, tmpdir):
    renderapi.stack.set_stack_state(
            loading_raw_stack, stack_state, render=render)
    p = copy.deepcopy(montage_parameters)
    p['input_stack']['name'] = loading_raw_stack
    p['pointmatch']['name'] = montage_pointmatches
    mod = EMaligner.EMaligner(input_data=p, args=[])
    mod.run()
    assert mod.results['precision'] < 1e-7
    assert mod.results['error'] < 200

    # try with affine_fullsize
    p['transformation'] = 'AffineModel'
    p['fullsze_transform'] = True
    mod = EMaligner.EMaligner(input_data=p, args=[])
    mod.run()
    assert mod.results['precision'] < 1e-7
    assert mod.results['error'] < 200

    # try with render interface
    p['input_stack']['db_interface'] = 'render'
    p['output_stack']['db_interface'] = 'render'
    p['pointmatch']['db_interface'] = 'render'
    mod = EMaligner.EMaligner(input_data=p, args=[])
    mod.run()
    assert mod.results['precision'] < 1e-7
    assert mod.results['error'] < 200

    # try different render output settings
    mod.args['render_output'] = 'stdout'
    mod.run()
    fout = tmpdir.join("myfile")
    mod.args['render_output'] = str(fout)
    mod.run()
