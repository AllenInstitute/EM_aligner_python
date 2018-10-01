import pytest
import renderapi
from test_data import (
        render_params,
        rough_parameters)
from EMaligner import EMaligner
from EMaligner.utils import EMalignerException
import json
import os

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
    tilespecs = [renderapi.tilespec.TileSpec(json=d)
                 for d in json.load(open(FILE_ROUGH_TILES, 'r'))]
    renderapi.stack.create_stack(test_rough_stack, render=render)
    renderapi.client.import_tilespecs(
            test_rough_stack, tilespecs, render=render)
    renderapi.stack.set_stack_state(
            test_rough_stack, 'COMPLETE', render=render)
    yield test_rough_stack
    renderapi.stack.delete_stack(test_rough_stack, render=render)


# raw stack tiles with one z removed
@pytest.fixture(scope='module')
def rough_input_stack_2(render):
    test_rough_stack = 'rough_input_stack_2'
    tilespecs = [renderapi.tilespec.TileSpec(json=d)
                 for d in json.load(open(FILE_ROUGH_TILES, 'r'))]
    renderapi.stack.create_stack(
            test_rough_stack, render=render)
    renderapi.client.import_tilespecs(
            test_rough_stack, tilespecs, render=render)
    z_values = renderapi.stack.get_z_values_for_stack(
            test_rough_stack, render=render)
    renderapi.stack.delete_section(
            test_rough_stack, z_values[3], render=render)
    renderapi.stack.set_stack_state(
            test_rough_stack, 'COMPLETE', render=render)
    yield test_rough_stack
    renderapi.stack.delete_stack(test_rough_stack, render=render)


@pytest.fixture(scope='module')
def rough_pointmatches(render):
    test_rough_collection = 'rough_collection'
    pms_from_json = json.load(open(FILE_ROUGH_PMS, 'r'))
    renderapi.pointmatch.import_matches(
            test_rough_collection, pms_from_json, render=render)
    yield test_rough_collection


def test_rough_similarity(render, rough_pointmatches, rough_input_stack):
    rough_parameters2 = dict(rough_parameters)
    rough_parameters2['input_stack']['name'] = rough_input_stack
    rough_parameters2['output_stack']['name'] = rough_input_stack + '_out'
    rough_parameters2['pointmatch']['name'] = rough_pointmatches
    rough_parameters2['transformation'] = 'SimilarityModel'
    mod = EMaligner.EMaligner(input_data=rough_parameters2, args=[])
    mod.run()
    tin = renderapi.tilespec.get_tile_specs_from_stack(
            rough_parameters2['input_stack']['name'], render=render)
    tout = renderapi.tilespec.get_tile_specs_from_stack(
            rough_parameters2['output_stack']['name'], render=render)
    assert mod.results['precision'] < 1e-7
    assert mod.results['error'] < 1e6
    assert len(tin) == len(tout)

    with pytest.raises(EMalignerException):
        mod.args['profile_data_load'] = True
        mod.run()


def test_rough_similarity_2(render, rough_pointmatches, rough_input_stack_2):
    rough_parameters2 = dict(rough_parameters)
    rough_parameters2['input_stack']['name'] = rough_input_stack_2
    rough_parameters2['output_stack']['name'] = rough_input_stack_2 + '_out'
    rough_parameters2['pointmatch']['name'] = rough_pointmatches
    rough_parameters2['transformation'] = 'SimilarityModel'
    mod = EMaligner.EMaligner(input_data=rough_parameters2, args=[])
    mod.run()
    tin = renderapi.tilespec.get_tile_specs_from_stack(
            rough_parameters2['input_stack']['name'], render=render)
    tout = renderapi.tilespec.get_tile_specs_from_stack(
            rough_parameters2['output_stack']['name'], render=render)
    assert mod.results['precision'] < 1e-7
    assert mod.results['error'] < 1e6
    assert len(tin) == len(tout)


def test_missing_section(render, rough_pointmatches, rough_input_stack_2):
    rough_parameters2 = dict(rough_parameters)
    rough_parameters2['input_stack']['name'] = rough_input_stack_2
    rough_parameters2['output_stack']['name'] = \
        rough_input_stack_2 + '_out_missing'
    rough_parameters2['pointmatch']['name'] = rough_pointmatches
    rough_parameters2['transformation'] = 'SimilarityModel'

    # delete a section
    groups = renderapi.stack.get_z_values_for_stack(
            rough_input_stack_2,
            render=render)
    n = int(len(groups)/2)
    renderapi.stack.set_stack_state(
            rough_input_stack_2,
            state='LOADING',
            render=render)
    renderapi.stack.delete_section(
            rough_input_stack_2,
            groups[n],
            render=render)
    renderapi.stack.set_stack_state(
            rough_input_stack_2,
            state='COMPLETE',
            render=render)

    rough_parameters2['input_stack']['db_interface'] = 'render'
    mod = EMaligner.EMaligner(input_data=rough_parameters2, args=[])
    mod.run()
    tin = renderapi.tilespec.get_tile_specs_from_stack(
            rough_parameters2['input_stack']['name'], render=render)
    tout = renderapi.tilespec.get_tile_specs_from_stack(
            rough_parameters2['output_stack']['name'], render=render)
    assert mod.results['precision'] < 1e-7
    assert mod.results['error'] < 1e6
    assert len(tin) == len(tout)

    rough_parameters2['input_stack']['db_interface'] = 'mongo'
    mod = EMaligner.EMaligner(input_data=rough_parameters2, args=[])
    mod.run()
    tin = renderapi.tilespec.get_tile_specs_from_stack(
            rough_parameters2['input_stack']['name'], render=render)
    tout = renderapi.tilespec.get_tile_specs_from_stack(
            rough_parameters2['output_stack']['name'], render=render)
    assert mod.results['precision'] < 1e-7
    assert mod.results['error'] < 1e6
    assert len(tin) == len(tout)


def test_affine_on_similarity(render, rough_pointmatches, rough_input_stack):
    rough_parameters2 = dict(rough_parameters)
    rough_parameters2['input_stack']['name'] = rough_input_stack
    rough_parameters2['output_stack']['name'] = 'sim_out'
    rough_parameters2['pointmatch']['name'] = rough_pointmatches
    rough_parameters2['transformation'] = 'SimilarityModel'
    mod = EMaligner.EMaligner(input_data=rough_parameters2, args=[])
    mod.run()

    rough_parameters2['input_stack']['name'] = 'sim_out'
    rough_parameters2['output_stack']['name'] = 'rough_affine'
    rough_parameters2['transformation'] = 'AffineModel'
    mod = EMaligner.EMaligner(input_data=rough_parameters2, args=[])
    mod.run()

    assert mod.results['precision'] < 1e-7
    assert mod.results['error'] < 1e6


def test_output_mode_none(render, rough_pointmatches, rough_input_stack):
    rough_parameters2 = dict(rough_parameters)
    rough_parameters2['input_stack']['name'] = rough_input_stack
    rough_parameters2['pointmatch']['name'] = rough_pointmatches
    rough_parameters2['transformation'] = 'AffineModel'
    rough_parameters2['output_mode'] = 'none'
    mod = EMaligner.EMaligner(input_data=rough_parameters2, args=[])
    mod.run()
    assert mod.results['precision'] < 1e-7
    assert mod.results['error'] < 1e6


def hdf5_fun(render, parameters, rough_pointmatches):
    rough_parameters2 = dict(parameters)

    # check output mode HDF5
    mod = EMaligner.EMaligner(input_data=rough_parameters2, args=[])
    mod.run()
    indexfile = os.path.join(
            rough_parameters2['hdf5_options']['output_dir'],
            'solution_input.h5')
    assert os.path.exists(indexfile)

    # check assemble from file
    rough_parameters2['output_mode'] = 'none'
    rough_parameters2['assemble_from_file'] = indexfile
    mod = EMaligner.EMaligner(input_data=rough_parameters2, args=[])
    mod.run()
    assert mod.results['precision'] < 1e-7
    assert mod.results['error'] < 1e6

    # check ingest from file
    try:
        renderapi.stack.delete_stack(
                rough_parameters2['output_stack']['name'],
                render=render)
    except renderapi.errors.RenderError:
        pass

    rough_parameters2['ingest_from_file'] = indexfile
    rough_parameters2['output_mode'] = 'stack'
    mod = EMaligner.EMaligner(input_data=rough_parameters2, args=[])
    mod.run()
    tin = renderapi.tilespec.get_tile_specs_from_stack(
            rough_parameters2['input_stack']['name'],
            render=render)
    tout = renderapi.tilespec.get_tile_specs_from_stack(
            rough_parameters2['output_stack']['name'],
            render=render)
    assert len(tin) == len(tout)
    os.remove(indexfile)


def test_hdf5_mode_similarity(
        render, rough_input_stack, rough_pointmatches, tmpdir):
    # general parameters
    parameters = dict(rough_parameters)
    parameters['hdf5_options']['output_dir'] = str(tmpdir.mkdir('hdf5output'))
    parameters['input_stack']['name'] = rough_input_stack
    parameters['pointmatch']['name'] = rough_pointmatches
    parameters['output_mode'] = 'hdf5'

    # specific tests
    parameters['transformation'] = 'SimilarityModel'
    parameters['fullsize_transform'] = False
    hdf5_fun(render, parameters, rough_pointmatches)

    parameters['transformation'] = 'AffineModel'
    parameters['fullsize_transform'] = False
    hdf5_fun(render, parameters, rough_pointmatches)

    parameters['transformation'] = 'AffineModel'
    parameters['fullsize_transform'] = True
    hdf5_fun(render, parameters, rough_pointmatches)

    parameters['transformation'] = 'AffineModel'
    parameters['fullsize_transform'] = True
    parameters['hdf5_options']['chunks_per_file'] = 2
    hdf5_fun(render, parameters, rough_pointmatches)

    parameters['transformation'] = 'AffineModel'
    parameters['fullsize_transform'] = False
    parameters['hdf5_options']['chunks_per_file'] = 2
    hdf5_fun(render, parameters, rough_pointmatches)
