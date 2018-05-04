import pytest
import renderapi
from test_data import (render_params,
                       render_json_template,
                       example_env,
                       montage_raw_tilespecs_json,
                       montage_parameters)
from EMaligner.EM_aligner_python_schema import *
from EMaligner.qctools.CheckPointMatches import CheckPointMatches
from EMaligner.qctools.CheckResiduals import CheckResiduals
from EMaligner.qctools.CheckTransforms import CheckTransforms
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
    #tilespecs = [renderapi.tilespec.TileSpec(json=d) for d in json.load(open(FILE_RAW_TILES,'r'))]
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

def test_pmplot(render,montage_pointmatches,raw_stack,tmpdir):
    montage_parameters['input_stack']['name']=raw_stack
    montage_parameters['pointmatch']['name'] = montage_pointmatches
    mod = CheckPointMatches(input_data = montage_parameters,args=[])
    mod.args['z1'] = 1015
    mod.args['z2'] = 1015
    mod.args['plot_dir'] = str(tmpdir.mkdir('plotoutput'))
    mod.run()
    assert os.path.exists(mod.outputname)

def test_resplot(render,montage_pointmatches,raw_stack,tmpdir):
    montage_parameters['input_stack']['name']=raw_stack
    montage_parameters['pointmatch']['name'] = montage_pointmatches
    mod = CheckResiduals(input_data = montage_parameters,args=[])
    mod.args['z1'] = 1015
    mod.args['z2'] = 1015
    mod.args['plot_dir'] = str(tmpdir.mkdir('plotoutput'))
    mod.args['savefig'] = "True"
    mod.run()
    assert os.path.exists(mod.outputname)

def test_trplot(render,montage_pointmatches,raw_stack,tmpdir):
    montage_parameters['input_stack']['name']=raw_stack
    montage_parameters['pointmatch']['name'] = montage_pointmatches
    mod = CheckTransforms(input_data = montage_parameters,args=[])
    mod.args['z1'] = 1015
    mod.args['plot_dir'] = str(tmpdir.mkdir('plotoutput'))
    mod.args['plot'] = "False"
    mod.run()
    assert os.path.exists(mod.outputname)

