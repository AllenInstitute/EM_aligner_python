from pymongo import MongoClient
import numpy as np
import renderapi
from renderapi.external.processpools import pool_pathos
import collections
import logging
import time
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import h5py
import os
import sys
import json
from .transform.transform import AlignerTransform

logger2 = logging.getLogger(__name__)


class EMalignerException(Exception):
    """Exception raised when there is a \
            problem creating a mesh lens correction"""
    pass


def make_dbconnection(collection, which='tile'):
    if collection['db_interface'] == 'mongo':
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
                    '__' + collection['name'] +
                    '__'+which)
            dbconnection = client.render[mongo_collection_name]
        elif collection['collection_type'] == 'pointmatch':
            mongo_collection_name = (
                    collection['owner'] +
                    '__' + collection['name'])
            dbconnection = client.match[mongo_collection_name]
    elif collection['db_interface'] == 'render':
        dbconnection = renderapi.connect(**collection)
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
                        stack['name'],
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


def get_tileids_and_tforms(stack, tform_name, zvals, fullsize=False, order=2):
    dbconnection = make_dbconnection(stack)

    tile_ids = []
    tile_tforms = []
    tile_tspecs = []
    shared_tforms = []
    sectionIds = []
    t0 = time.time()

    for z in zvals:
        # load tile specs from the database
        if stack['db_interface'] == 'render':
            try:
                tmp = renderapi.resolvedtiles.get_resolved_tiles_from_z(
                        stack['name'],
                        float(z),
                        render=dbconnection,
                        owner=stack['owner'],
                        project=stack['project'])
                tspecs = tmp.tilespecs
                for st in tmp.transforms:
                    shared_tforms.append(st)
                try:
                    sectionId = renderapi.stack.get_sectionId_for_z(
                        stack['name'],
                        float(z),
                        render=dbconnection,
                        owner=stack['owner'],
                        project=stack['project'])
                except renderapi.errors.RenderError:
                    sectionId = collections.Counter([
                        ts.layout.sectionId for ts in tspecs]
                        ).most_common()[0][0]
            except renderapi.errors.RenderError:
                # missing section
                sectionId = None
                pass

        if stack['db_interface'] == 'mongo':
            filt = {'z': float(z)}
            if dbconnection.count_documents(filt) == 0:
                sectionId = None
            else:
                cursor = dbconnection.find(filt).sort([
                        ('layout.imageRow', 1),
                        ('layout.imageCol', 1)])
                tspecs = list(cursor)
                refids = []
                for ts in tspecs:
                    for m in np.arange(len(ts['transforms']['specList'])):
                        if 'refId' in ts['transforms']['specList'][m]:
                            refids.append(
                                    ts['transforms']['specList'][m]['refId'])
                refids = np.unique(np.array(refids))
                # be selective of which transforms to pass on to the new stack
                dbconnection2 = make_dbconnection(stack, which='transform')
                for refid in refids:
                    shared_tforms.append(
                            renderapi.transform.load_transform_json(
                                list(dbconnection2.find({"id": refid}))[0]))
                sectionId = dbconnection.find(
                        {"z": float(z)}).distinct("layout.sectionId")[0]

        if sectionId is not None:
            sectionIds.append(sectionId)

            # make lists of IDs and transforms
            solve_tf = AlignerTransform(
                    name=tform_name, fullsize=fullsize, order=order)
            for k in np.arange(len(tspecs)):
                if stack['db_interface'] == 'mongo':
                    tspecs[k] = renderapi.tilespec.TileSpec(json=tspecs[k])
                tile_ids.append(tspecs[k].tileId)
                tile_tspecs.append(tspecs[k])
                # make space in the solve vector
                # for a solve-type transform
                # with input transform as values (constraints)
                tile_tforms.append(
                        solve_tf.to_solve_vec(tspecs[k].tforms[-1]))

    logger2.info(
            "\n loaded %d tile specs from %d zvalues in "
            "%0.1f sec using interface: %s" % (
                len(tile_ids),
                len(zvals),
                time.time() - t0,
                stack['db_interface']))

    tile_tforms = np.concatenate(tile_tforms, axis=0)

    return {
            'tids': np.array(tile_ids),
            'tforms': tile_tforms,
            'tspecs': np.array(tile_tspecs).flatten(),
            'shared_tforms': shared_tforms,
            'sectionIds': sectionIds
            }


def get_matches(iId, jId, collection, dbconnection):
    if collection['db_interface'] == 'render':
        if iId == jId:
            matches = renderapi.pointmatch.get_matches_within_group(
                    collection['name'],
                    iId,
                    owner=collection['owner'],
                    render=dbconnection)
        else:
            matches = renderapi.pointmatch.get_matches_from_group_to_group(
                    collection['name'],
                    iId,
                    jId,
                    owner=collection['owner'],
                    render=dbconnection)
        matches = np.array(matches)
    if collection['db_interface'] == 'mongo':
        cursor = dbconnection.find(
                {'pGroupId': iId, 'qGroupId': jId},
                {'_id': False})
        matches = np.array(list(cursor))
        if iId != jId:
            # in principle, this does nothing if zi < zj, but, just in case
            cursor = dbconnection.find(
                    {
                        'pGroupId': jId,
                        'qGroupId': iId},
                    {'_id': False})
            matches = np.append(matches, list(cursor))
    message = ("\n %d matches for section1=%s section2=%s "
               "in pointmatch collection" % (len(matches), iId, jId))
    if len(matches) == 0:
        logger2.debug(message)
    else:
        logger2.debug(message)
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

    logger2.info(
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
        logger2.info('render output is going to /dev/null')
    elif outarg == 'stdout':
        stdeo = sys.stdout
        if sys.version_info[0] >= 3:
            stdeo = sys.stdout.buffer
        logger2.info('render output is going to stdout')
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
        input_stack,
        outputname,
        tform_name,
        fullsize,
        order,
        tspecs,
        shared_tforms,
        x,
        ingestconn,
        unused_tids,
        outarg,
        use_rest,
        overwrite_zlayer):

    # replace the last transform in the tilespec with the new one
    solve_tf = AlignerTransform(
            name=tform_name, fullsize=fullsize, order=order)
    render_tforms = solve_tf.from_solve_vec(x)
    for m in np.arange(len(tspecs)):
        tspecs[m].tforms[-1] = render_tforms[m]

    tspecs = tspecs.tolist()
    unused_tspecs = get_unused_tspecs(input_stack, unused_tids)
    tspecs = tspecs + unused_tspecs.tolist()
    logger2.info(
        "\ningesting results to %s:%d %s__%s__%s" % (
            ingestconn.DEFAULT_HOST,
            ingestconn.DEFAULT_PORT,
            ingestconn.DEFAULT_OWNER,
            ingestconn.DEFAULT_PROJECT,
            outputname))

    stdeo = get_stderr_stdout(outarg)

    if overwrite_zlayer:
        zvalues = np.unique(np.array([t.z for t in tspecs]))
        for zvalue in zvalues:
            renderapi.stack.delete_section(
                    outputname,
                    zvalue,
                    render=ingestconn)

    renderapi.client.import_tilespecs_parallel(
            outputname,
            tspecs,
            sharedTransforms=shared_tforms,
            render=ingestconn,
            close_stack=False,
            mpPool=pool_pathos.PathosWithPool,
            stderr=stdeo,
            stdout=stdeo,
            use_rest=use_rest)
