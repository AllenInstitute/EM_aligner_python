import pytest
import renderapi
from test_data import (
        render_params,
        rough_parameters)
from EMaligner import EMaligner, utils
from marshmallow import ValidationError
import json
import os
import copy
import numpy as np
import requests

dname = os.path.dirname(os.path.abspath(__file__))
FILE_ROUGH_TILES = os.path.join(
        dname, 'test_files', 'rough_input_tiles.json')
FILE_ROUGH_PMS = os.path.join(
        dname, 'test_files', 'rough_input_matches.json')
FILE_ROUGH_PMS_S1 = os.path.join(
        dname, 'test_files', 'rough_input_matches_split1.json')
FILE_ROUGH_PMS_S2 = os.path.join(
        dname, 'test_files', 'rough_input_matches_split2.json')


def mysession():
    s = requests.Session()
    #retry = requests.packages.urllib3.util.retry.Retry(
    #        total=5,
    #        connect=5,
    #        read=5,
    #        backoff_factor=2)
    #s.mount('http://', requests.adapters.HTTPAdapter(max_retries=retry))
    return s


#@pytest.fixture()
#def render():
#    render = renderapi.connect(**render_params)
#    return render


# raw stack tiles
@pytest.fixture(scope='module')
def rough_input_stack():
    render = renderapi.connect(**render_params)
    test_rough_stack = 'rough_input_stack'
    with open(FILE_ROUGH_TILES, 'r') as f:
        j = json.load(f)
    tilespecs = [renderapi.tilespec.TileSpec(json=d) for d in j]
    renderapi.stack.create_stack(test_rough_stack, render=render, session=mysession())
    renderapi.client.import_tilespecs(
            test_rough_stack, tilespecs, render=render, session=mysession())
    renderapi.stack.set_stack_state(
            test_rough_stack, 'COMPLETE', render=render, session=mysession())
    yield test_rough_stack
    renderapi.stack.delete_stack(test_rough_stack, render=render)


# raw stack tiles with one z removed
@pytest.fixture()
def rough_input_stack_2():
    render = renderapi.connect(**render_params)
    test_rough_stack2 = 'rough_input_stack_2'
    with open(FILE_ROUGH_TILES, 'r') as f:
        j = json.load(f)
    tilespecs = [renderapi.tilespec.TileSpec(json=d) for d in j]
    renderapi.stack.create_stack(
            test_rough_stack2, render=render, session=mysession())
    renderapi.client.import_tilespecs(
            test_rough_stack2, tilespecs, render=render, session=mysession())
    z_values = renderapi.stack.get_z_values_for_stack(
            test_rough_stack2, render=render, session=mysession())
    renderapi.stack.delete_section(
            test_rough_stack2, z_values[3], render=render, session=mysession())
    renderapi.stack.set_stack_state(
            test_rough_stack2, 'COMPLETE', render=render, session=mysession())
    yield test_rough_stack2
    renderapi.stack.delete_stack(test_rough_stack2, render=render)


@pytest.fixture(scope='module')
def rough_pointmatches():
    render = renderapi.connect(**render_params)
    test_rough_collection = 'rough_collection'
    with open(FILE_ROUGH_PMS, 'r') as f:
        pms_from_json = json.load(f)
    renderapi.pointmatch.import_matches(
            test_rough_collection, pms_from_json, render=render, session=mysession())
    yield test_rough_collection
    renderapi.pointmatch.delete_collection(
            test_rough_collection, render=render)


@pytest.fixture()
def split_rough_pointmatches():
    render = renderapi.connect(**render_params)
    test_rough_collection1 = 'rough_collection_split1'
    test_rough_collection2 = 'rough_collection_split2'
    with open(FILE_ROUGH_PMS_S1, 'r') as f:
        pms_from_json = json.load(f)
    renderapi.pointmatch.import_matches(
            test_rough_collection1, pms_from_json, render=render, session=mysession())
    with open(FILE_ROUGH_PMS_S2, 'r') as f:
        pms_from_json = json.load(f)
    renderapi.pointmatch.import_matches(
            test_rough_collection2, pms_from_json, render=render, session=mysession())
    yield [test_rough_collection1, test_rough_collection2]
    renderapi.pointmatch.delete_collection(
            test_rough_collection1, render=render)
    renderapi.pointmatch.delete_collection(
            test_rough_collection2, render=render)


def test_rough_similarity_explicit_depth(
        rough_pointmatches, rough_input_stack):
    render = renderapi.connect(**render_params)
    rough_parameters2 = copy.deepcopy(rough_parameters)
    rough_parameters2['input_stack']['name'] = rough_input_stack
    rough_parameters2['output_stack']['name'] = rough_input_stack + '_out'
    rough_parameters2['pointmatch']['name'] = rough_pointmatches
    rough_parameters2['transformation'] = 'SimilarityModel'
    rough_parameters2['matrix_assembly']['depth'] = [0, 1, 2]
    rough_parameters2['matrix_assembly']['explicit_weight_by_depth'] = \
        [0, 0.5, 0.33]
    rough_parameters2['n_parallel_jobs'] = 1
    mod = EMaligner.EMaligner(
            input_data=copy.deepcopy(rough_parameters2), args=[])
    mod.run()
    tin = renderapi.tilespec.get_tile_specs_from_stack(
            rough_parameters2['input_stack']['name'], render=render, session=mysession())
    tout = renderapi.tilespec.get_tile_specs_from_stack(
            rough_parameters2['output_stack']['name'], render=render, session=mysession())
    assert np.all(np.array(mod.results['precision']) < 1e-7)
    assert np.all(np.array(mod.results['error']) < 1e6)
    assert len(tin) == len(tout)

    renderapi.stack.delete_stack(rough_parameters2['output_stack']['name'], render=render)

    with pytest.raises(ValidationError):
        rough_parameters2['matrix_assembly']['depth'] = [0, 1]
        # not the same length as weights
        EMaligner.EMaligner(
                input_data=copy.deepcopy(rough_parameters2), args=[])
    rough_parameters2['matrix_assembly']['depth'] = [0, 1, 2]


def test_multi_stack_name_exception(
        rough_pointmatches, rough_input_stack):
    render = renderapi.connect(**render_params)
    rough_parameters2 = copy.deepcopy(rough_parameters)
    rough_parameters2['input_stack']['name'] = rough_input_stack
    rough_parameters2['output_stack']['name'] = rough_input_stack + '_out'
    rough_parameters2['pointmatch']['name'] = rough_pointmatches
    rough_parameters2['transformation'] = 'SimilarityModel'
    rough_parameters2['n_parallel_jobs'] = 1
    with pytest.raises(ValidationError):
        rough_parameters2['input_stack']['name'] = [
                rough_parameters2['input_stack']['name']] * 2
        # stacks should only have 1 name (so far)
        EMaligner.EMaligner(
                input_data=copy.deepcopy(rough_parameters2), args=[])


def test_multi_profile_exception(
        rough_pointmatches, rough_input_stack):
    render = renderapi.connect(**render_params)
    rough_parameters2 = copy.deepcopy(rough_parameters)
    rough_parameters2['input_stack']['name'] = rough_input_stack
    rough_parameters2['output_stack']['name'] = rough_input_stack + '_out'
    rough_parameters2['pointmatch']['name'] = rough_pointmatches
    rough_parameters2['transformation'] = 'SimilarityModel'
    rough_parameters2['n_parallel_jobs'] = 1
    mod = EMaligner.EMaligner(
            input_data=copy.deepcopy(rough_parameters2), args=[])
    with pytest.raises(utils.EMalignerException):
        mod.args['profile_data_load'] = True
        mod.run()


def test_rough_similarity_2(rough_pointmatches, rough_input_stack_2):
    render = renderapi.connect(**render_params)
    rough_parameters2 = copy.deepcopy(rough_parameters)
    rough_parameters2['input_stack']['name'] = rough_input_stack_2
    rough_parameters2['output_stack']['name'] = rough_input_stack_2 + '_out'
    rough_parameters2['pointmatch']['name'] = rough_pointmatches
    rough_parameters2['transformation'] = 'SimilarityModel'
    mod = EMaligner.EMaligner(
            input_data=copy.deepcopy(rough_parameters2), args=[])
    mod.run()
    tin = renderapi.tilespec.get_tile_specs_from_stack(
            rough_parameters2['input_stack']['name'], render=render, session=mysession())
    tout = renderapi.tilespec.get_tile_specs_from_stack(
            rough_parameters2['output_stack']['name'], render=render, session=mysession())

    renderapi.stack.delete_stack(rough_parameters2['output_stack']['name'], render=render)

    assert np.all(np.array(mod.results['precision']) < 1e-7)
    assert np.all(np.array(mod.results['error']) < 1e6)
    assert len(tin) == len(tout)


def test_rough_similarity_split(
        split_rough_pointmatches, rough_input_stack_2):
    render = renderapi.connect(**render_params)
    rough_parameters2 = copy.deepcopy(rough_parameters)
    rough_parameters2['input_stack']['name'] = rough_input_stack_2
    rough_parameters2['output_stack']['name'] = rough_input_stack_2 + '_out'
    rough_parameters2['pointmatch']['name'] = split_rough_pointmatches
    rough_parameters2['transformation'] = 'SimilarityModel'
    rough_parameters2['n_parallel_jobs'] = 1
    mod = EMaligner.EMaligner(
            input_data=copy.deepcopy(rough_parameters2), args=[])
    mod.run()
    tin = renderapi.tilespec.get_tile_specs_from_stack(
            rough_parameters2['input_stack']['name'], render=render, session=mysession())
    tout = renderapi.tilespec.get_tile_specs_from_stack(
            rough_parameters2['output_stack']['name'], render=render, session=mysession())

    renderapi.stack.delete_stack(rough_parameters2['output_stack']['name'], render=render)

    assert np.all(np.array(mod.results['precision']) < 1e-7)
    assert np.all(np.array(mod.results['error']) < 1e6)
    assert len(tin) == len(tout)

    rough_parameters2['pointmatch']['db_interface'] = "render"
    mod = EMaligner.EMaligner(
            input_data=copy.deepcopy(rough_parameters2), args=[])
    mod.run()
    tin = renderapi.tilespec.get_tile_specs_from_stack(
            rough_parameters2['input_stack']['name'], render=render, session=mysession())
    tout = renderapi.tilespec.get_tile_specs_from_stack(
            rough_parameters2['output_stack']['name'], render=render, session=mysession())

    renderapi.stack.delete_stack(rough_parameters2['output_stack']['name'], render=render)

    assert np.all(np.array(mod.results['precision']) < 1e-7)
    assert np.all(np.array(mod.results['error']) < 1e6)
    assert len(tin) == len(tout)


def test_missing_section(rough_pointmatches, rough_input_stack_2):
    render = renderapi.connect(**render_params)
    rough_parameters2 = copy.deepcopy(rough_parameters)
    rough_parameters2['input_stack']['name'] = rough_input_stack_2
    rough_parameters2['output_stack']['name'] = \
        rough_input_stack_2 + '_out_missing'
    rough_parameters2['pointmatch']['name'] = rough_pointmatches
    rough_parameters2['transformation'] = 'SimilarityModel'
    rough_parameters2['n_parallel_jobs'] = 1

    # delete a section
    groups = renderapi.stack.get_z_values_for_stack(
            rough_input_stack_2,
            render=render, session=mysession())
    n = int(len(groups)/2)
    renderapi.stack.set_stack_state(
            rough_input_stack_2,
            state='LOADING',
            render=render, session=mysession())
    renderapi.stack.delete_section(
            rough_input_stack_2,
            groups[n],
            render=render, session=mysession())
    renderapi.stack.set_stack_state(
            rough_input_stack_2,
            state='COMPLETE',
            render=render, session=mysession())

    rough_parameters2['input_stack']['db_interface'] = 'render'
    mod = EMaligner.EMaligner(
            input_data=copy.deepcopy(rough_parameters2), args=[])
    mod.run()
    tin = renderapi.tilespec.get_tile_specs_from_stack(
            rough_parameters2['input_stack']['name'], render=render, session=mysession())
    tout = renderapi.tilespec.get_tile_specs_from_stack(
            rough_parameters2['output_stack']['name'], render=render, session=mysession())

    renderapi.stack.delete_stack(rough_parameters2['output_stack']['name'], render=render)

    assert np.all(np.array(mod.results['precision']) < 1e-7)
    assert np.all(np.array(mod.results['error']) < 1e6)
    assert len(tin) == len(tout)

    rough_parameters2['input_stack']['db_interface'] = 'mongo'
    mod = EMaligner.EMaligner(
            input_data=copy.deepcopy(rough_parameters2), args=[])
    mod.run()
    tin = renderapi.tilespec.get_tile_specs_from_stack(
            rough_parameters2['input_stack']['name'], render=render, session=mysession())
    tout = renderapi.tilespec.get_tile_specs_from_stack(
            rough_parameters2['output_stack']['name'], render=render, session=mysession())

    renderapi.stack.delete_stack(rough_parameters2['output_stack']['name'], render=render)

    assert np.all(np.array(mod.results['precision']) < 1e-7)
    assert np.all(np.array(mod.results['error']) < 1e6)
    assert len(tin) == len(tout)


def test_affine_on_similarity(
        rough_pointmatches, rough_input_stack):
    render = renderapi.connect(**render_params)
    rough_parameters2 = copy.deepcopy(rough_parameters)
    rough_parameters2['input_stack']['name'] = rough_input_stack
    rough_parameters2['output_stack']['name'] = 'sim_out'
    rough_parameters2['pointmatch']['name'] = rough_pointmatches
    rough_parameters2['transformation'] = 'SimilarityModel'
    mod = EMaligner.EMaligner(
            input_data=copy.deepcopy(rough_parameters2), args=[])
    mod.run()

    rough_parameters2['input_stack']['name'] = 'sim_out'
    rough_parameters2['output_stack']['name'] = 'rough_affine'
    rough_parameters2['transformation'] = 'AffineModel'
    mod = EMaligner.EMaligner(
            input_data=copy.deepcopy(rough_parameters2), args=[])
    mod.run()
    
    renderapi.stack.delete_stack(rough_parameters2['input_stack']['name'], render=render)
    renderapi.stack.delete_stack(rough_parameters2['output_stack']['name'], render=render)

    assert np.all(np.array(mod.results['precision']) < 1e-7)
    assert np.all(np.array(mod.results['error']) < 1e6)


#def test_output_mode_none(render, rough_pointmatches, rough_input_stack):
#    rough_parameters2 = copy.deepcopy(rough_parameters)
#    rough_parameters2['input_stack']['name'] = rough_input_stack
#    rough_parameters2['pointmatch']['name'] = rough_pointmatches
#    rough_parameters2['transformation'] = 'AffineModel'
#    rough_parameters2['output_mode'] = 'none'
#    mod = EMaligner.EMaligner(
#            input_data=copy.deepcopy(rough_parameters2), args=[])
#    mod.run()
#    assert np.all(np.array(mod.results['precision']) < 1e-7)
#    assert np.all(np.array(mod.results['error']) < 1e6)
