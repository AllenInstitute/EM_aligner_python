import pytest
import renderapi
from test_data import (render_params,
                       montage_raw_tilespecs_json,
                       montage_parameters)
from EMaligner import EMaligner
import json
from marshmallow.exceptions import ValidationError
import copy
import os
import numpy as np
import shutil

dname = os.path.dirname(os.path.abspath(__file__))
FILE_PMS = os.path.join(
        dname, 'test_files', 'montage_pointmatches.json')


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
    renderapi.stack.delete_stack(test_raw_stack, render=render)


@pytest.fixture(scope='module')
def montage_pointmatches(render):
    test_montage_collection = 'montage_collection'
    pms_from_json = []
    with open(FILE_PMS, 'r') as f:
        pms_from_json = json.load(f)
    renderapi.pointmatch.import_matches(
            test_montage_collection, pms_from_json, render=render)
    yield test_montage_collection
    renderapi.pointmatch.delete_collection(
            test_montage_collection, render=render)


@pytest.fixture(scope='module')
def solved_montage(render, raw_stack, montage_pointmatches):
    p = copy.deepcopy(montage_parameters)
    p['input_stack']['name'] = raw_stack
    p['output_stack']['name'] = 'solver_output_stack'
    p['pointmatch']['name'] = montage_pointmatches
    mod = EMaligner.EMaligner(input_data=p, args=[])
    mod.run()
    yield mod
    renderapi.stack.delete_stack('output_stack_name', render=render)


def test_input_stack_file(
        render, raw_stack, montage_pointmatches, tmpdir, solved_montage):
    p = copy.deepcopy(montage_parameters)
    resolved = renderapi.resolvedtiles.get_resolved_tiles_from_z(
            raw_stack,
            p['first_section'],
            render=render)
    tmp_file_dir = str(tmpdir.mkdir('file_test_dir'))
    input_stack_file = os.path.join(
            tmp_file_dir, "input_stack.json")
    with open(input_stack_file, 'w') as f:
        json.dump(resolved.to_dict(), f)

    p['input_stack']['db_interface'] = 'file'
    p['input_stack']['input_file'] = input_stack_file
    p['output_mode'] = 'none'
    p['pointmatch']['name'] = montage_pointmatches

    tmod = EMaligner.EMaligner(input_data=p, args=[])
    tmod.run()

    for k in ['precision', 'error', 'err']:
        assert np.all(
                np.isclose(
                    np.array(tmod.results[k]),
                    np.array(solved_montage.results[k])))

    assert np.all(
            np.isclose(
                np.linalg.norm(solved_montage.results['x'], axis=0),
                np.linalg.norm(tmod.results['x'], axis=0)))

    orig_ids = np.array([
        t.tileId for t in solved_montage.solved_resolved.tilespecs])
    for t in tmod.solved_resolved.tilespecs:
        i = np.argwhere(orig_ids == t.tileId).flatten()[0]
        assert np.all(
                np.isclose(
                    solved_montage.solved_resolved.tilespecs[i].tforms[-1].M,
                    t.tforms[-1].M))

    shutil.rmtree(tmp_file_dir)
