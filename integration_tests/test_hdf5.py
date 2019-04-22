import pytest
import renderapi
from test_data import (
        render_params,
        rough_parameters)
from EMaligner import EMaligner
import json
import os
import copy
import numpy as np

dname = os.path.dirname(os.path.abspath(__file__))
FILE_ROUGH_TILES = os.path.join(
        dname, 'test_files', 'rough_input_tiles.json')
FILE_ROUGH_PMS = os.path.join(
        dname, 'test_files', 'rough_input_matches.json')


@pytest.fixture(scope='module')
def render():
    render = renderapi.connect(**render_params)
    return render


# raw stack tiles
@pytest.fixture(scope='module')
def rough_input_stack(render):
    test_rough_stack = 'hdf5_rough_input_stack'
    with open(FILE_ROUGH_TILES, 'r') as f:
        j = json.load(f)
    tilespecs = [renderapi.tilespec.TileSpec(json=d) for d in j]
    renderapi.stack.create_stack(test_rough_stack, render=render)
    renderapi.client.import_tilespecs(
            test_rough_stack, tilespecs, render=render)
    renderapi.stack.set_stack_state(
            test_rough_stack, 'COMPLETE', render=render)
    yield test_rough_stack
    renderapi.stack.delete_stack(test_rough_stack, render=render)


@pytest.fixture(scope='module')
def rough_pointmatches(render):
    test_rough_collection = 'hdf5_rough_collection'
    with open(FILE_ROUGH_PMS, 'r') as f:
        pms_from_json = json.load(f)
    renderapi.pointmatch.import_matches(
            test_rough_collection, pms_from_json, render=render)
    yield test_rough_collection
    renderapi.pointmatch.delete_collection(
            test_rough_collection, render=render)


@pytest.mark.parametrize("chunks", [-1, 1, 2])
def test_hdf5_mode_similarity(
        render,
        rough_input_stack,
        rough_pointmatches,
        tmpdir,
        chunks):
    # general parameters
    parameters = copy.deepcopy(rough_parameters)
    parameters['hdf5_options']['output_dir'] = str(tmpdir.mkdir('hdf5output'))
    parameters['input_stack']['name'] = rough_input_stack
    parameters['pointmatch']['name'] = rough_pointmatches
    parameters['output_mode'] = 'hdf5'
    parameters['hdf5_options']['chunks_per_file'] = chunks

    # check output mode HDF5
    mod = EMaligner.EMaligner(
            input_data=copy.deepcopy(parameters), args=[])
    mod.run()
    indexfile = os.path.join(
            parameters['hdf5_options']['output_dir'],
            'solution_input.h5')
    assert os.path.exists(indexfile)
    del mod

    # check assemble from file
    parameters['output_mode'] = 'none'
    parameters['assemble_from_file'] = indexfile
    mod = EMaligner.EMaligner(
            input_data=copy.deepcopy(parameters), args=[])
    mod.run()
    assert np.all(np.array(mod.results['precision']) < 1e-7)
    assert np.all(np.array(mod.results['error']) < 1e6)
    del mod

    # delete the output stack, just in case
    renderapi.stack.delete_stack(
                parameters['output_stack']['name'],
                render=render)

    parameters['ingest_from_file'] = indexfile
    parameters['output_mode'] = 'stack'
    mod = EMaligner.EMaligner(
            input_data=copy.deepcopy(parameters), args=[])
    mod.run()
    tin = renderapi.tilespec.get_tile_specs_from_stack(
            parameters['input_stack']['name'],
            render=render)
    tout = renderapi.tilespec.get_tile_specs_from_stack(
            parameters['output_stack']['name'],
            render=render)
    assert len(tin) == len(tout)
    os.remove(indexfile)
    del mod

    renderapi.stack.delete_stack(
                parameters['output_stack']['name'],
                render=render)
