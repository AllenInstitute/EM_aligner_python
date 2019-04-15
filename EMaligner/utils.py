from pymongo import MongoClient
import numpy as np
import renderapi
from renderapi.external.processpools import pool_pathos
import logging
import time
import warnings
import os
import sys
import json
from functools import partial
from scipy.sparse.linalg import factorized
from .transform.transform import AlignerTransform
import copy
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import h5py


logger = logging.getLogger(__name__)


class EMalignerException(Exception):
    """Exception raised when there is a \
            problem creating a mesh lens correction"""
    pass


def make_dbconnection(collection, which='tile', interface=None):
    if interface is None:
        interface = collection['db_interface']

    if interface == 'mongo':
        if collection['mongo_userName'] != '':
            client = MongoClient(
                    host=collection['mongo_host'],
                    port=collection['mongo_port'],
                    username=collection['mongo_userName'],
                    authSource=collection['mongo_authenticationDatabase'],
                    password=collection['mongo_password'])
        else:
            client = MongoClient(
                    host=collection['mongo_host'],
                    port=collection['mongo_port'])

        if collection['collection_type'] == 'stack':
            # for getting shared transforms, which='transform'
            mongo_collection_name = (
                    collection['owner'] +
                    '__' + collection['project'] +
                    '__' + collection['name'][0] +
                    '__'+which)
            dbconnection = client.render[mongo_collection_name]
        elif collection['collection_type'] == 'pointmatch':
            mongo_collection_name = [(
                    collection['owner'] +
                    '__' + name) for name in collection['name']]
            dbconnection = [
                    client.match[name] for name in mongo_collection_name]
    elif interface == 'render':
        dbconnection = renderapi.connect(**collection)
    elif interface == 'file':
        return None
    else:
        raise EMalignerException(
                "invalid interface in make_dbconnection()")
    return dbconnection


def get_unused_tspecs(stack, tids):
    dbconnection = make_dbconnection(stack)
    tspecs = []
    if stack['db_interface'] == 'render':
        for t in tids:
            tspecs.append(
                    renderapi.tilespec.get_tile_spec(
                        stack['name'][0],
                        t,
                        render=dbconnection,
                        owner=stack['owner'],
                        project=stack['project']))
    if stack['db_interface'] == 'mongo':
        for t in tids:
            tmp = list(dbconnection.find({'tileId': t}))
            tspecs.append(
                    renderapi.tilespec.TileSpec(
                        json=tmp[0]))
    return np.array(tspecs)


def determine_zvalue_pairs(resolved, depths):
    # create all possible pairs, given zvals and depth
    zvals = [t.z for t in resolved.tilespecs]
    zvals, uind = np.unique(zvals, return_index=True)
    sections = [resolved.tilespecs[i].layout.sectionId for i in uind]
    pairs = []
    for i in range(len(zvals)):
        for j in depths:
            # need to get rid of duplicates
            z2 = zvals[i] + j
            if z2 in zvals:
                i2 = np.argwhere(zvals == z2)[0][0]
                pairs.append({
                    'z1': zvals[i],
                    'z2': zvals[i2],
                    'section1': sections[i],
                    'section2': sections[i2]})
    return pairs


def ready_transforms(tilespecs, tform_name, fullsize, order):
    for t in tilespecs:
        t.tforms[-1] = AlignerTransform(
            name=tform_name,
            transform=t.tforms[-1],
            fullsize=fullsize,
            order=order)


def get_resolved_from_z(stack, tform_name, fullsize, order, z):
    resolved = renderapi.resolvedtiles.ResolvedTiles()
    dbconnection = make_dbconnection(stack)
    if stack['db_interface'] == 'render':
        try:
            resolved = renderapi.resolvedtiles.get_resolved_tiles_from_z(
                    stack['name'][0],
                    float(z),
                    render=dbconnection,
                    owner=stack['owner'],
                    project=stack['project'])
        except renderapi.errors.RenderError:
            pass
    if stack['db_interface'] == 'mongo':
        filt = {'z': float(z)}
        if dbconnection.count_documents(filt) != 0:
            # this sort ordering is the same as render, I think
            cursor = dbconnection.find(filt).sort([
                    ('layout.imageRow', 1),
                    ('layout.imageCol', 1)])
            tspecs = [renderapi.tilespec.TileSpec(json=c) for c in cursor]
            refids = np.unique([
                [tf.refId for tf in t.tforms if
                    isinstance(tf, renderapi.transform.ReferenceTransform)]
                for t in tspecs])
            # don't perpetuate unused reference transforms
            dbconnection2 = make_dbconnection(stack, which='transform')

            def tfjson(refid):
                return list(dbconnection2.find({"id": refid}))[0]

            shared_tforms = [renderapi.transform.load_transform_json(
                tfjson(refid)) for refid in refids]
            resolved.tilespecs = tspecs
            resolved.transforms = shared_tforms

    # turn the last transform of every tilespec into an AlignerTransform
    ready_transforms(resolved.tilespecs, tform_name, fullsize, order)

    return resolved


def get_resolved_tilespecs(
        stack, tform_name, pool_size, zvals, fullsize=False, order=2):
    t0 = time.time()
    if stack['db_interface'] == 'file':
        with open(stack['input_file'], 'r') as f:
            resolved = renderapi.resolvedtiles.ResolvedTiles(
                    json=json.load(f))
        resolved.tilespecs = [t for t in resolved.tilespecs if t.z in zvals]
        ready_transforms(resolved.tilespecs, tform_name, fullsize, order)
    else:
        resolved = renderapi.resolvedtiles.ResolvedTiles()
        getz = partial(get_resolved_from_z, stack, tform_name, fullsize, order)
        with renderapi.client.WithPool(pool_size) as pool:
            for rz in pool.map(getz, zvals):
                resolved.tilespecs += rz.tilespecs
                resolved.transforms += rz.transforms

    logger.info(
        "\n loaded %d tile specs from %d zvalues in "
        "%0.1f sec using interface: %s" % (
            len(resolved.tilespecs),
            len(zvals),
            time.time() - t0,
            stack['db_interface']))

    return resolved


def get_matches(iId, jId, collection, dbconnection):
    matches = []
    if collection['db_interface'] == 'render':
        if iId == jId:
            for name in collection['name']:
                matches.extend(renderapi.pointmatch.get_matches_within_group(
                        name,
                        iId,
                        owner=collection['owner'],
                        render=dbconnection))
        else:
            for name in collection['name']:
                matches.extend(
                        renderapi.pointmatch.get_matches_from_group_to_group(
                            name,
                            iId,
                            jId,
                            owner=collection['owner'],
                            render=dbconnection))
    if collection['db_interface'] == 'mongo':
        for dbconn in dbconnection:
            cursor = dbconn.find(
                    {'pGroupId': iId, 'qGroupId': jId},
                    {'_id': False})
            matches.extend(list(cursor))
            if iId != jId:
                # in principle, this does nothing if zi < zj, but, just in case
                cursor = dbconn.find(
                        {
                            'pGroupId': jId,
                            'qGroupId': iId},
                        {'_id': False})
                matches.extend(list(cursor))
    message = ("\n %d matches for section1=%s section2=%s "
               "in pointmatch collection" % (len(matches), iId, jId))
    logger.debug(message)

    return matches


def write_chunk_to_file(fname, c, file_weights):
    fcsr = h5py.File(fname, "w")

    indptr_dset = fcsr.create_dataset(
            "indptr",
            (c.indptr.size, 1),
            dtype='int64')
    indptr_dset[:] = (c.indptr).reshape(c.indptr.size, 1)

    indices_dset = fcsr.create_dataset(
            "indices",
            (c.indices.size, 1),
            dtype='int64')
    indices_dset[:] = c.indices.reshape(c.indices.size, 1)
    nrows = indptr_dset.size-1

    data_dset = fcsr.create_dataset(
            "data",
            (c.data.size,),
            dtype='float64')
    data_dset[:] = c.data

    weights_dset = fcsr.create_dataset(
            "weights",
            (file_weights.size,),
            dtype='float64')
    weights_dset[:] = file_weights
    fcsr.close()

    logger.info(
        "wrote %s %0.2fGB on disk" % (
            fname,
            os.path.getsize(fname)/(2.**30)))
    return {
            "name": os.path.basename(fname),
            "nnz": c.indices.size,
            "mincol": c.indices.min(),
            "maxcol": c.indices.max(),
            "nrows": nrows
            }


def write_reg_and_tforms(
        args,
        metadata,
        tforms,
        reg,
        tids,
        unused_tids):

    fname = os.path.join(
            args['hdf5_options']['output_dir'],
            'solution_input.h5')
    with h5py.File(fname, "w") as f:
        for j in np.arange(tforms.shape[1]):
            dsetname = 'transforms_%d' % j
            dset = f.create_dataset(
                    dsetname,
                    (tforms[:, j].size,),
                    dtype='float64')
            dset[:] = tforms[:, j]

        # a list of transform indices (clunky, but works for PETSc to count)
        tlist = np.arange(tforms.shape[1]).astype('int32')
        dset = f.create_dataset(
                "transform_list",
                (tlist.size, 1),
                dtype='int32')
        dset[:] = tlist.reshape(tlist.size, 1)

        # create a regularization vector
        vec = reg.diagonal()
        dset = f.create_dataset(
                "lambda",
                (vec.size,),
                dtype='float64')
        dset[:] = vec

        # keep track here what tile_ids were used
        str_type = h5py.special_dtype(vlen=str)
        dset = f.create_dataset(
                "used_tile_ids",
                (tids.size,),
                dtype=str_type)
        dset[:] = tids

        # keep track here what tile_ids were not used
        dset = f.create_dataset(
                "unused_tile_ids",
                (unused_tids.size,),
                dtype=str_type)
        dset[:] = unused_tids

        # keep track of input args
        dset = f.create_dataset(
                "input_args",
                (1,),
                dtype=str_type)
        dset[:] = json.dumps(args, indent=2)

        # metadata
        names = [m['name'] for m in metadata]
        dset = f.create_dataset(
                "datafile_names",
                (len(names),),
                dtype=str_type)
        dset[:] = names

        for key in ['nrows', 'nnz', 'mincol', 'maxcol']:
            vals = np.array([m[key] for m in metadata])
            dset = f.create_dataset(
                    "datafile_" + key,
                    (vals.size, 1),
                    dtype='int64')
            dset[:] = vals.reshape(vals.size, 1)

        print('wrote %s' % fname)


def get_stderr_stdout(outarg):
    if outarg == 'null':
        stdeo = open(os.devnull, 'wb')
        logger.info('render output is going to /dev/null')
    elif outarg == 'stdout':
        stdeo = sys.stdout
        if sys.version_info[0] >= 3:
            stdeo = sys.stdout.buffer
        logger.info('render output is going to stdout')
    else:
        i = 0
        odir, oname = os.path.split(outarg)
        while os.path.exists(outarg):
            t = oname.split('.')
            outarg = odir + '/'
            for it in t[:-1]:
                outarg += it
            outarg += '%d.%s' % (i, t[-1])
            i += 1
        stdeo = open(outarg, 'a')
    return stdeo


def write_to_new_stack(
        resolved,
        output_stack,
        outarg,
        overwrite_zlayer):

    ingestconn = make_dbconnection(output_stack, interface='render')
    logger.info(
        "\ningesting results to %s:%d %s__%s__%s" % (
            ingestconn.DEFAULT_HOST,
            ingestconn.DEFAULT_PORT,
            ingestconn.DEFAULT_OWNER,
            ingestconn.DEFAULT_PROJECT,
            output_stack['name'][0]))
    stdeo = get_stderr_stdout(outarg)

    if overwrite_zlayer:
        zvalues = np.unique(np.array([t.z for t in resolved.tilespecs]))
        for zvalue in zvalues:
            renderapi.stack.delete_section(
                    output_stack['name'][0],
                    zvalue,
                    render=ingestconn)

    renderapi.client.import_tilespecs_parallel(
            output_stack['name'][0],
            resolved.tilespecs,
            sharedTransforms=resolved.transforms,
            render=ingestconn,
            close_stack=False,
            mpPool=pool_pathos.PathosWithPool,
            stderr=stdeo,
            stdout=stdeo,
            use_rest=output_stack['use_rest'])


def solve(A, weights, reg, x0):
    time0 = time.time()
    # regularized least squares
    # ensure symmetry of K
    weights.data = np.sqrt(weights.data)
    rtWA = weights.dot(A)
    K = rtWA.transpose().dot(rtWA) + reg

    del weights, rtWA

    # factorize, then solve, efficient for large affine
    x = np.zeros_like(x0)
    err = np.zeros((A.shape[0], x.shape[1]))
    precision = [0] * x.shape[1]

    solve = factorized(K)
    for i in range(x0.shape[1]):
        # can solve for same A, but multiple x0's
        Lm = reg.dot(x0[:, i])
        x[:, i] = solve(Lm)
        err[:, i] = A.dot(x[:, i])
        precision[i] = \
            np.linalg.norm(K.dot(x[:, i]) - Lm) / np.linalg.norm(Lm)
        del Lm
    del K

    results = {}
    results['precision'] = precision
    results['error'] = np.linalg.norm(err, axis=0).tolist()
    results['err'] = [
            [m, e] for m, e in
            zip(np.abs(err).mean(axis=0), np.abs(err).std(axis=0))]
    results['x'] = x
    results['time'] = time.time() - time0

    return results


def message_from_solve_results(results):
    message = ' solved in %0.1f sec\n' % results['time']
    message += " precision [norm(Kx-Lm)/norm(Lm)] = "
    message += ", ".join(["%0.1e" % ix for ix in results['precision']])
    message += "\n error     [norm(Ax-b)] = "
    message += ", ".join(["%0.3f" % ix for ix in results['error']])
    message += "\n [mean(|Ax|)+/-std(|Ax|)] : "
    message += ", ".join([
        "%0.1f +/- %0.1f" % (e[0], e[1]) for e in results['err']])
    return message


def create_or_set_loading(stack):
    dbconnection = make_dbconnection(
            stack,
            interface='render')
    renderapi.stack.create_stack(
        stack['name'][0],
        render=dbconnection)


def set_complete(stack):
    dbconnection = make_dbconnection(
            stack,
            interface='render')
    renderapi.stack.set_stack_state(
        stack['name'][0],
        state='COMPLETE',
        render=dbconnection)


def get_z_values_for_stack(stack, zvals):
    dbconnection = make_dbconnection(stack)
    if stack['db_interface'] == 'render':
        zstack = renderapi.stack.get_z_values_for_stack(
                stack['name'][0],
                render=dbconnection)
    if stack['db_interface'] == 'mongo':
        zstack = dbconnection.distinct('z')
    if stack['db_interface'] == 'file':
        with open(stack['input_file'], 'r') as f:
            resolved = renderapi.resolvedtiles.ResolvedTiles(
                    json=json.load(f))
        zstack = np.unique([t.z for t in resolved.tilespecs])

    ind = np.isin(zvals, zstack)
    return zvals[ind]


def update_tilespecs(resolved, x, used):
    up_resolved = copy.deepcopy(resolved)
    index = 0
    for i in range(len(up_resolved.tilespecs)):
        if used[i]:
            index += up_resolved.tilespecs[i].tforms[-1].from_solve_vec(
                    x[index:, :])
    return up_resolved
