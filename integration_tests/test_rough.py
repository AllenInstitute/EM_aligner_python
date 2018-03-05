import pytest
import renderapi
from test_data import (render_params,render_json_template,example_env,rough_parameters)
from EMaligner import EMaligner
import json
import os
import numpy as np

FILE_ROUGH_TILES = './integration_tests/test_files/rough_input_tiles.json'
FILE_ROUGH_PMS = './integration_tests/test_files/rough_input_matches.json'

@pytest.fixture(scope='module')
def render():
    render = renderapi.connect(**render_params)
    return render

# raw stack tiles
@pytest.fixture(scope='module')
def rough_input_stack(render):
    test_rough_stack = 'rough_input_stack'
    tilespecs = [renderapi.tilespec.TileSpec(json=d) for d in json.load(open(FILE_ROUGH_TILES,'r'))]
    renderapi.stack.create_stack(test_rough_stack,render=render)
    renderapi.client.import_tilespecs(test_rough_stack,tilespecs,render=render)
    renderapi.stack.set_stack_state(test_rough_stack,'COMPLETE',render=render)
    yield test_rough_stack
    renderapi.stack.delete_stack(test_rough_stack,render=render)

@pytest.fixture(scope='module')
def rough_pointmatches(render):
    test_rough_collection = 'rough_collection'
    pms_from_json = json.load(open(FILE_ROUGH_PMS,'r'))
    renderapi.pointmatch.import_matches(test_rough_collection,pms_from_json,render=render)
    yield test_rough_collection

def test_rough_rigid(render,rough_pointmatches,rough_input_stack,tmpdir):
    #do a rough rigid alignment
    rough_parameters['input_stack']['name']=rough_input_stack
    #rough_parameters['input_stack']['project']=render_params['project']
    rough_parameters['pointmatch']['name'] = rough_pointmatches
    mod = EMaligner.EMaligner(input_data = rough_parameters,args=[])
    mod.run()
    rigid_out=rough_parameters['output_stack']['name']
    assert mod.results['precision'] < 1e-7
    assert mod.results['error'] < 1e6

    tin = renderapi.tilespec.get_tile_specs_from_stack(rough_parameters['input_stack']['name'],render=render)
    tout = renderapi.tilespec.get_tile_specs_from_stack(rough_parameters['output_stack']['name'],render=render)

    assert len(tin)==len(tout)

    #add an affine on top of that
    rough_parameters['input_stack']['name']=rigid_out
    rough_parameters['output_stack']['name']='rough_affine'
    rough_parameters['transformation'] = 'affine'
    #rough_parameters['input_stack']['project']=render_params['project']
    #rough_parameters['pointmatch']['name'] = rough_pointmatches
    mod = EMaligner.EMaligner(input_data = rough_parameters,args=[])
    mod.run()
    assert mod.results['precision'] < 1e-7
    assert mod.results['error'] < 1e6

    #check output mode none
    rough_parameters['output_mode'] = 'none'
    mod = EMaligner.EMaligner(input_data = rough_parameters,args=[])
    mod.run()
    prec = mod.results['precision']
    err = mod.results['error']
    assert prec < 1e-7
    assert err < 1e6

    #check output mode HDF5
    rough_parameters['output_mode'] = 'hdf5'
    rough_parameters['hdf5_options']['output_dir'] = str(tmpdir.mkdir('hdf5output'))
    mod = EMaligner.EMaligner(input_data = rough_parameters,args=[])
    mod.run()
    indexfile = rough_parameters['hdf5_options']['output_dir']+'/index.txt'
    assert os.path.exists(indexfile)

    #check assemble from file
    rough_parameters['output_mode'] = 'none'
    rough_parameters['start_from_file']=indexfile
    mod = EMaligner.EMaligner(input_data = rough_parameters,args=[])
    mod.run()
    assert np.abs(mod.results['precision']-prec) < 1e-10
    assert np.abs(mod.results['error']-err) < 1e-10

    #check again with multiple hdf5 files
    #check output mode HDF5
    rough_parameters['output_mode'] = 'hdf5'
    rough_parameters['hdf5_options']['chunks_per_file'] = 3
    #if not os.path.exists(rough_parameters['hdf5_options']['output_dir']):
    #    cmd = 'mkdir -p '+rough_parameters['hdf5_options']['output_dir']
    #    os.system(cmd)
    mod = EMaligner.EMaligner(input_data = rough_parameters,args=[])
    mod.run()
    indexfile = rough_parameters['hdf5_options']['output_dir']+'/index.txt'
    assert os.path.exists(indexfile)

    #check assemble from file
    rough_parameters['output_mode'] = 'none'
    rough_parameters['start_from_file']=indexfile
    mod = EMaligner.EMaligner(input_data = rough_parameters,args=[])
    mod.run()
    assert np.abs(mod.results['precision']-prec) < 1e-10
    assert np.abs(mod.results['error']-err) < 1e-10

