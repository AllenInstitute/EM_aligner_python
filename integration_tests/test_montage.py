import pytest
import renderapi
from test_data import (render_params,render_json_template,example_env,montage_parameters)
from EMaligner import EMaligner
import json
import os

FILE_RAW_TILES = './integration_tests/test_files/raw_tiles_for_montage.json'
FILE_PMS = './integration_tests/test_files/montage_pointmatches.json'

@pytest.fixture(scope='module')
def render():
    render = renderapi.connect(**render_params)
    return render

# raw stack tiles
@pytest.fixture(scope='module')
def raw_stack(render):
    test_raw_stack = 'input_raw_stack'
    tilespecs = [renderapi.tilespec.TileSpec(json=d) for d in json.load(open(FILE_RAW_TILES,'r'))]
    renderapi.stack.create_stack(test_raw_stack,render=render)
    renderapi.client.import_tilespecs(test_raw_stack,tilespecs,render=render)
    renderapi.stack.set_stack_state(test_raw_stack,'COMPLETE',render=render)
    yield test_raw_stack
    renderapi.stack.delete_stack(test_raw_stack,render=render)

@pytest.fixture(scope='module')
def montage_pointmatches(render):
    test_montage_collection = 'montage_collection'
    pms_from_json = json.load(open(FILE_PMS,'r'))
    renderapi.pointmatch.import_matches(test_montage_collection,pms_from_json,render=render)
    yield test_montage_collection

def test_first_test(render,montage_pointmatches,raw_stack):
    montage_parameters['input_stack']['name']=raw_stack
#    montage_parameters['input_stack']['project']=render_params['project']
    montage_parameters['pointmatch']['name'] = montage_pointmatches
    mod = EMaligner.EMaligner(input_data = montage_parameters,args=[])
    mod.run()
    assert mod.results['precision'] < 1e-7
    assert mod.results['error'] < 200
