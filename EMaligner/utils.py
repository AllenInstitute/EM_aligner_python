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


def get_tileids_and_tforms(stack, tform_name, zvals):
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
                sectionId = ""
                pass

        if stack['db_interface'] == 'mongo':
            cursor = dbconnection.find(
                {'z': float(z)}).sort([
                    ('layout.imageRow', 1),
                    ('layout.imageCol', 1)])
            if cursor.count() == 0:
                sectionId = None
            else:
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
            for k in np.arange(len(tspecs)):
                if stack['db_interface'] == 'render':
                    tile_ids.append(tspecs[k].tileId)
                    if 'affine' in tform_name:
                        tile_tforms.append([
                            tspecs[k].tforms[-1].M[0, 0],
                            tspecs[k].tforms[-1].M[0, 1],
                            tspecs[k].tforms[-1].M[0, 2],
                            tspecs[k].tforms[-1].M[1, 0],
                            tspecs[k].tforms[-1].M[1, 1],
                            tspecs[k].tforms[-1].M[1, 2]])
                    elif tform_name == 'rigid':
                        tile_tforms.append([
                            tspecs[k].tforms[-1].M[0, 0],
                            tspecs[k].tforms[-1].M[0, 1],
                            tspecs[k].tforms[-1].M[0, 2],
                            tspecs[k].tforms[-1].M[1, 2]])
                if stack['db_interface'] == 'mongo':
                    tile_ids.append(tspecs[k]['tileId'])
                    last_tf = tspecs[k]['transforms']['specList'][-1]
                    dstring = last_tf['dataString']
                    dfloat = np.array(dstring.split()).astype('float')
                    if 'affine' in tform_name:
                        tile_tforms.append(dfloat[[0, 2, 4, 1, 3, 5]])
                    elif tform_name == 'rigid':
                        tile_tforms.append(dfloat[[0, 2, 4, 5]])
                    tspecs[k] = renderapi.tilespec.TileSpec(json=tspecs[k])
                tile_tspecs.append(tspecs[k])

    logger2.info(
            "---\nloaded %d tile specs from %d zvalues in "
            "%0.1f sec using interface: %s" % (
                len(tile_ids),
                len(zvals),
                time.time() - t0,
                stack['db_interface']))
    return (
            np.array(tile_ids),
            np.array(tile_tforms).flatten(),
            np.array(tile_tspecs).flatten(),
            shared_tforms,
            sectionIds)


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

    tmp = fname.split('/')
    indtxt = 'file %s ' % tmp[-1]
    indtxt += 'nrow %ld mincol %ld maxcol %ld nnz %ld\n' % (
            indptr_dset.size-1,
            c.indices.min(),
            c.indices.max(),
            c.indices.size)
    fcsr.close()
    logger2.info(
        "wrote %s %0.2fGB on disk" % (
            fname,
            os.path.getsize(fname)/(2.**30)))
    return indtxt


def write_reg_and_tforms(
        output_mode,
        hdf5_options,
        filt_tforms,
        reg,
        filt_tids,
        unused_tids):

    if output_mode == 'hdf5':
        fname = hdf5_options['output_dir'] + '/regularization.h5'
        f = h5py.File(fname, "w")
        tlist = []
        for j in np.arange(len(filt_tforms)):
            dsetname = 'transforms_%d' % j
            dset = f.create_dataset(
                    dsetname,
                    (filt_tforms[j].size,),
                    dtype='float64')
            dset[:] = filt_tforms[j]
            tlist.append(j)

        # a list of transform indices (clunky, but works for PETSc to count)
        tlist = np.array(tlist).astype('int32')
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
        dt = h5py.special_dtype(vlen=str)
        dset = f.create_dataset(
                "tile_ids",
                (filt_tids.size,),
                dtype=dt)
        dset[:] = filt_tids
        # keep track here what tile_ids were not used
        dt = h5py.special_dtype(vlen=str)
        dset = f.create_dataset(
                "unused_tile_ids",
                (unused_tids.size,),
                dtype=dt)
        dset[:] = unused_tids
        f.close()
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
        tform_type,
        tspecs,
        shared_tforms,
        x,
        ingestconn,
        unused_tids,
        outarg,
        use_rest,
        overwrite_zlayer):

    # replace the last transform in the tilespec with the new one
    for m in np.arange(len(tspecs)):
        if 'affine' in tform_type:
            tspecs[m].tforms[-1].M[0, 0] = x[m * 6 + 0]
            tspecs[m].tforms[-1].M[0, 1] = x[m * 6 + 1]
            tspecs[m].tforms[-1].M[0, 2] = x[m * 6 + 2]
            tspecs[m].tforms[-1].M[1, 0] = x[m * 6 + 3]
            tspecs[m].tforms[-1].M[1, 1] = x[m * 6 + 4]
            tspecs[m].tforms[-1].M[1, 2] = x[m * 6 + 5]
        elif tform_type == 'rigid':
            tspecs[m].tforms[-1].M[0, 0] = x[m * 4 + 0]
            tspecs[m].tforms[-1].M[0, 1] = x[m * 4 + 1]
            tspecs[m].tforms[-1].M[0, 2] = x[m * 4 + 2]
            tspecs[m].tforms[-1].M[1, 0] = -x[m * 4 + 1]
            tspecs[m].tforms[-1].M[1, 1] = x[m * 4 + 0]
            tspecs[m].tforms[-1].M[1, 2] = x[m * 4 + 3]
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
