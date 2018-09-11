import pytest
import renderapi
from test_data import (render_params,
                       render_json_template,
                       example_env,
                       montage_raw_tilespecs_json,
                       montage_parameters)
from EMaligner import EMaligner
import json
import os

#FILE_RAW_TILES = './integration_tests/test_files/raw_tiles_for_montage.json'
FILE_PMS = './integration_tests/test_files/montage_pointmatches.json'

@pytest.fixture(scope='module')
def render():
    render = renderapi.connect(**render_params)
    return render

# raw stack tiles
@pytest.fixture(scope='module')
def raw_stack(render):
    test_raw_stack = 'input_raw_stack'
    tilespecs = [renderapi.tilespec.TileSpec(json=d) for d in montage_raw_tilespecs_json]
    renderapi.stack.create_stack(test_raw_stack,render=render)
    renderapi.client.import_tilespecs(test_raw_stack,tilespecs,render=render)
    renderapi.stack.set_stack_state(test_raw_stack,'COMPLETE',render=render)
    yield test_raw_stack
    renderapi.stack.delete_stack(test_raw_stack,render=render)


@pytest.fixture(scope='function')
def loading_raw_stack(render):
    test_raw_stack = 'input_raw_stack_loading'
    tilespecs = [renderapi.tilespec.TileSpec(json=d) for d in montage_raw_tilespecs_json]
    renderapi.stack.create_stack(test_raw_stack,render=render)
    renderapi.client.import_tilespecs(test_raw_stack,tilespecs,render=render)
    yield test_raw_stack
    renderapi.stack.delete_stack(test_raw_stack,render=render)


@pytest.fixture(scope='module')
def montage_pointmatches(render):
    test_montage_collection = 'montage_collection'
    pms_from_json = []
    with open(FILE_PMS, 'r') as f:
        pms_from_json = json.load(f)

    renderapi.pointmatch.import_matches(test_montage_collection,pms_from_json,render=render)
    yield test_montage_collection

@pytest.fixture(scope='module')
def montage_pointmatches_weighted(render):
    test_montage_collection2 = 'montage_collection2'
    pms_from_json = []
    with open(FILE_PMS, 'r') as f:
        pms_from_json = json.load(f)
    n = len(pms_from_json[0]['matches']['w'])
    pms_from_json[0]['matches']['w'] = [0.0 for i in range(n)]

    renderapi.pointmatch.import_matches(test_montage_collection2,pms_from_json,render=render)
    yield test_montage_collection2


@pytest.mark.parametrize("stack_state", ["COMPLETE", "LOADING"])
def test_weighted(render,montage_pointmatches_weighted,loading_raw_stack, stack_state, tmpdir):
    renderapi.stack.set_stack_state(loading_raw_stack, stack_state, render=render)
    montage_parameters['input_stack']['name'] = loading_raw_stack
    montage_parameters['pointmatch']['name'] = montage_pointmatches_weighted
    mod = EMaligner.EMaligner(input_data = montage_parameters,args=[])
    mod.run()
    assert mod.results['precision'] < 1e-7
    assert mod.results['error'] < 200


@pytest.mark.parametrize("stack_state", ["COMPLETE", "LOADING"])
def test_first_test(render,montage_pointmatches,loading_raw_stack, stack_state, tmpdir):
    renderapi.stack.set_stack_state(loading_raw_stack, stack_state, render=render)
    montage_parameters['input_stack']['name'] = loading_raw_stack
    montage_parameters['pointmatch']['name'] = montage_pointmatches
    mod = EMaligner.EMaligner(input_data = montage_parameters,args=[])
    mod.run()
    assert mod.results['precision'] < 1e-7
    assert mod.results['error'] < 200

    #try with affine_fullsize
    montage_parameters['transformation'] = 'affine_fullsize'
    mod = EMaligner.EMaligner(input_data = montage_parameters,args=[])
    mod.run()
    assert mod.results['precision'] < 1e-7
    assert mod.results['error'] < 200

    #try with render interface
    montage_parameters['input_stack']['db_interface'] = 'render'
    montage_parameters['output_stack']['db_interface'] = 'render'
    montage_parameters['pointmatch']['db_interface'] = 'render'
    mod = EMaligner.EMaligner(input_data=montage_parameters, args=[])
    mod.run()
    assert mod.results['precision'] < 1e-7
    assert mod.results['error'] < 200

    # try different render output settings
    mod.args['render_output'] = 'stdout'
    mod.run()
    fout = tmpdir.join("myfile")
    mod.args['render_output'] = str(fout)
    mod.run()
