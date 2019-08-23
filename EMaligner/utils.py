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
import scipy.sparse as sparse
from scipy.sparse.linalg import factorized
from .transform.transform import AlignerTransform, AlignerRotationModel
from . import jsongz
import collections
import itertools
import subprocess
import requests
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import h5py

logger = logging.getLogger(__name__)


class EMalignerException(Exception):
    """EM_aligner exception"""
    pass


def make_dbconnection(collection, which='tile', interface=None):
    """creates a multi-interface object for stacks and collections

    Parameters
    ----------
    collection : :class:`EMaligner.schemas.db_params`
    which : str
        switch for having mongo retrieve reference transforms
    interface : str or None
        specification to override EMaligner.schemas.db_params.db_interface

    Returns
    -------
    dbconnection : obj
        a multi-interface object used by other functions
        in :mod:`EMaligner.utils`

    """
    if interface is None:
        interface = collection['db_interface']

    if interface == 'mongo':
        mongoconn = collections.namedtuple('mongoconn', 'client collection')
        client = MongoClient(
                host=collection['mongo_host'],
                port=collection['mongo_port'])
        if collection['mongo_userName'] != '':
            client = MongoClient(
                    host=collection['mongo_host'],
                    port=collection['mongo_port'],
                    username=collection['mongo_userName'],
                    authSource=collection['mongo_authenticationDatabase'],
                    password=collection['mongo_password'])

        if collection['collection_type'] == 'stack':
            # for getting shared transforms, which='transform'
            mongo_collection_name = (
                    collection['owner'] +
                    '__' + collection['project'] +
                    '__' + collection['name'][0] +
                    '__'+which)
            dbconnection = mongoconn(
                    client=client,
                    collection=client.render[mongo_collection_name])
        elif collection['collection_type'] == 'pointmatch':
            mongo_collection_name = [(
                    collection['owner'] +
                    '__' + name) for name in collection['name']]
            dbconnection = [mongoconn(
                                client=client,
                                collection=client.match[name])
                            for name in mongo_collection_name]
    elif interface == 'render':
        dbconnection = renderapi.connect(**collection)
    elif interface == 'file':
        return None
    else:
        raise EMalignerException(
                "invalid interface in make_dbconnection()")
    return dbconnection


def determine_zvalue_pairs(resolved, depths):
    """creates a lidt of pairs by z that will be included in the solve

    Parameters
    ----------
    resolved : :class:`renderapi.resolvedtiles.ResolvedTiles`
        input tilespecs
    depths : List
        depths (z-differences) that will be included in the matrix

    Returns
    -------
    pairs : List of dict
        keys are z values and sectionIds for each pair

    """
    # create all possible pairs, given zvals and depth
    zvals, uind, inv = np.unique(
            [t.z for t in resolved.tilespecs],
            return_index=True,
            return_inverse=True)
    sections = [resolved.tilespecs[i].layout.sectionId for i in uind]
    index = np.arange(len(resolved.tilespecs))
    pairs = []
    for i, z1 in enumerate(zvals):
        ind1 = index[inv == i]
        for j in depths:
            # need to get rid of duplicates
            z2 = z1 + j
            if z2 in zvals:
                i2 = np.argwhere(zvals == z2)[0][0]
                ind = np.unique(np.hstack((index[inv == i2], ind1)))
                pairs.append({
                    'z1': z1,
                    'z2': z2,
                    'section1': sections[i],
                    'section2': sections[i2],
                    'ind': ind})
    return pairs


def ready_transforms(tilespecs, tform_name, fullsize, order):
    """mutate last transform in each tilespec to be an AlignerTransform

    Parameters
    ----------
    tilespecs : List
        :class:`renderapi.tilespec.TileSpec` objects.
    tform_name : str
        intended destination type for the mutation
    fullsize : bool
        passed as kwarg to AlignerTransform
    order : int
        passed as kwarg to AlignerTransform

    """
    for t in tilespecs:
        # for first starts with thin plate spline
        if ((tform_name == 'ThinPlateSplineTransform') &
                (not isinstance(
                        t.tforms[-1],
                        renderapi.transform.ThinPlateSplineTransform))):
            xt, yt = np.meshgrid(
                    np.linspace(0, t.width, 3), np.linspace(0, t.height, 3))
            src = np.vstack((xt.flatten(), yt.flatten())).transpose()
            dst = t.tforms[-1].tform(src)
            t.tforms[-1] = renderapi.transform.ThinPlateSplineTransform()
            t.tforms[-1].estimate(src, dst)
        t.tforms[-1] = AlignerTransform(
            name=tform_name,
            transform=t.tforms[-1],
            fullsize=fullsize,
            order=order)


def get_resolved_from_z(stack, tform_name, fullsize, order, z):
    """retrieves a ResolvedTiles object from some source and mutates the
       final transform for each tilespec into an AlignerTransform object

    Parameters
    ----------
    stack :  :class:`EMaligner.schemas.input_stack`
    tform_name : str
        specifies which transform to mutate into (solve for)
    fullsize : bool
        passed as kwarg to the EMaligner.transform.AlignerTransform
    order : int
        passed as kwarg to the EMaligner.transform.AlignerTransform
    z : int or float
        z value for one section

    Returns
    -------
    resolved : :class:`renderapi.resolvedtiles.ResolvedTiles`

    """
    resolved = renderapi.resolvedtiles.ResolvedTiles()
    dbconnection = make_dbconnection(stack)
    if stack['db_interface'] == 'render':
        try:
            with requests.Session() as s:
                s.mount('http://', requests.adapters.HTTPAdapter(max_retries=5))
                resolved = renderapi.resolvedtiles.get_resolved_tiles_from_z(
                        stack['name'][0],
                        float(z),
                        render=dbconnection,
                        owner=stack['owner'],
                        project=stack['project'],
                        session=s)
        except renderapi.errors.RenderError as e:
            logger.warning(str(e))
            pass
    if stack['db_interface'] == 'mongo':
        filt = {'z': float(z)}
        if dbconnection.collection.count_documents(filt) != 0:
            cursor = dbconnection.collection.find(filt)
            tspecs = [renderapi.tilespec.TileSpec(json=c) for c in cursor]
            cursor.close()
            dbconnection.client.close()
            refids = np.unique([
                [tf.refId for tf in t.tforms if
                    isinstance(tf, renderapi.transform.ReferenceTransform)]
                for t in tspecs])
            # don't perpetuate unused reference transforms
            dbconnection2 = make_dbconnection(stack, which='transform')

            def tfjson(refid):
                c = dbconnection2.collection.find({"id": refid})
                x = list(c)[0]
                c.close()
                return x

            shared_tforms = [renderapi.transform.load_transform_json(
                tfjson(refid)) for refid in refids]
            dbconnection2.client.close()
            resolved = renderapi.resolvedtiles.ResolvedTiles(
                    tilespecs=tspecs, transformList=shared_tforms)
            del tspecs, shared_tforms

    # turn the last transform of every tilespec into an AlignerTransform
    ready_transforms(resolved.tilespecs, tform_name, fullsize, order)

    return resolved


def get_resolved_tilespecs(
        stack, tform_name, pool_size, zvals, fullsize=False, order=2):
    """retrieves ResolvedTiles objects from some source and mutates the
       final transform for each tilespec into an AlignerTransform object

    Parameters
    ----------
    stack :  :class:`EMaligner.schemas.input_stack`
    tform_name : str
        specifies which transform to mutate into (solve for)
    pool_size : int
        level of parallelization for parallel reads
    fullsize : bool
        passed as kwarg to the EMaligner.transform.AlignerTransform
    order : int
        passed as kwarg to the EMaligner.transform.AlignerTransform
    zvals : :class:`numpy.ndarray`
        z values for desired sections

    Returns
    -------
    resolved : :class:`renderapi.resolvedtiles.ResolvedTiles`

    """
    t0 = time.time()
    if stack['db_interface'] == 'file':
        resolved = renderapi.resolvedtiles.ResolvedTiles(
                json=jsongz.load(stack['input_file']))
        resolved.tilespecs = [t for t in resolved.tilespecs if t.z in zvals]
        ready_transforms(resolved.tilespecs, tform_name, fullsize, order)
    else:
        resolved = renderapi.resolvedtiles.ResolvedTiles()
        getz = partial(get_resolved_from_z, stack, tform_name, fullsize, order)
        with renderapi.client.WithPool(pool_size) as pool:
            results = pool.map(getz, zvals)
        resolved.tilespecs = list(itertools.chain.from_iterable(
            [r.__dict__.pop('tilespecs') for r in results]))
        resolved.transforms = list(itertools.chain.from_iterable(
            [r.__dict__.pop('transforms') for r in results]))

    logger.info(
        "\n loaded %d tile specs from %d zvalues in "
        "%0.1f sec using interface: %s" % (
            len(resolved.tilespecs),
            len(zvals),
            time.time() - t0,
            stack['db_interface']))

    return resolved


def get_matches(iId, jId, collection, dbconnection):
    """retrieve point correspondences

    Parameters
    ----------
    iId : str
        sectionId for 1st section
    jId : str
        sectionId for 2nd section
    collection : :class:`EMaligner.schemas.pointmatch`
    dbconnection : object returned by :meth:`EMaligner.utils.make_dbconnection`

    Returns
    -------
    matches : List of dict
        standard render/mongo representation of point matches

    """
    matches = []
    if collection['db_interface'] == 'file':
        matches = jsongz.load(collection['input_file'])
        matches = [m for m in matches
                   if set([m['pGroupId'], m['qGroupId']]) & set([iId, jId])]
    if collection['db_interface'] == 'render':
        with requests.Session() as s:
            s.mount('http://', requests.adapters.HTTPAdapter(max_retries=5))
            if iId == jId:
                for name in collection['name']:
                    matches.extend(renderapi.pointmatch.get_matches_within_group(
                            name,
                            iId,
                            owner=collection['owner'],
                            render=dbconnection,
                            session=s))
            else:
                for name in collection['name']:
                    matches.extend(
                            renderapi.pointmatch.get_matches_from_group_to_group(
                                name,
                                iId,
                                jId,
                                owner=collection['owner'],
                                render=dbconnection,
                                session=s))
    if collection['db_interface'] == 'mongo':
        for dbconn in dbconnection:
            cursor = dbconn.collection.find(
                    {'pGroupId': iId, 'qGroupId': jId},
                    {'_id': False})
            matches.extend(list(cursor))
            cursor.close()
            if iId != jId:
                # in principle, this does nothing if zi < zj, but, just in case
                cursor = dbconn.collection.find(
                        {
                            'pGroupId': jId,
                            'qGroupId': iId},
                        {'_id': False})
                matches.extend(list(cursor))
                cursor.close()
        dbconn.client.close()
    message = ("\n %d matches for section1=%s section2=%s "
               "in pointmatch collection" % (len(matches), iId, jId))
    logger.debug(message)

    return matches


def write_chunk_to_file(fname, c, file_weights, rhs):
    """write a sub-matrix to an hdf5 file for an external solve

    Parameters
    ----------
    fname : str
        path to output file
    c : :class:`scipy.sparse.csr_matrix`
        N x M matrix sub block
    file_weights : :class:`numpy.ndarray`
        length N array of weights
    rhs : :class:`numpy.ndarray`
        N x nsolve right hand sides

    """
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

    for j in np.arange(rhs.shape[1]):
        dsetname = 'rhs_%d' % j
        dset = fcsr.create_dataset(
                dsetname,
                (rhs[:, j].size,),
                dtype='float64')
        dset[:] = rhs[:, j]

    # a list of rhs indices (clunky, but works for PETSc to count)
    rhslist = np.arange(rhs.shape[1]).astype('int32')
    dset = fcsr.create_dataset(
            "rhs_list",
            (rhslist.size, 1),
            dtype='int32')
    dset[:] = rhslist.reshape(rhslist.size, 1)

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
        resolved,
        metadata,
        tforms,
        reg):
    """write regularization and transforms (x0) to hdf5

    Parameters
    ----------
    args : dict
        passed from EMaligner object
    resolved : :class:`renderapi.resolvedtiles.ResolvedTiles`
        resolved tilespec object to output
    metadata : dict
        helper values about matrix for external solver
    tforms : :class:`numpy.ndarray`
        M x nsolve starting values (x0)
    reg : :class:`scipy.sparse.csr_matrix`
        M x M diagonal regularization values

    """

    fname = os.path.join(
            args['hdf5_options']['output_dir'],
            'solution_input.h5')
    with h5py.File(fname, "w") as f:
        for j in np.arange(tforms.shape[1]):
            dsetname = 'x_%d' % j
            dset = f.create_dataset(
                    dsetname,
                    (tforms[:, j].size,),
                    dtype='float64')
            dset[:] = tforms[:, j]

        # a list of transform indices (clunky, but works for PETSc to count)
        tlist = np.arange(tforms.shape[1]).astype('int32')
        dset = f.create_dataset(
                "solve_list",
                (tlist.size, 1),
                dtype='int32')
        dset[:] = tlist.reshape(tlist.size, 1)

        # create a regularization vector
        vec = reg.diagonal()
        dset = f.create_dataset(
                "reg",
                (vec.size,),
                dtype='float64')
        dset[:] = vec

        str_type = h5py.special_dtype(vlen=bytes)

        rname = os.path.join(
                os.path.dirname(fname),
                "resolved.json.gz")

        dset = f.create_dataset(
                "resolved_tiles",
                (1,),
                data=os.path.basename(rname))

        jsongz.dump(resolved.to_dict(), rname)

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
    """helper function for suppressing render output

    Parameters
    ----------
    outarg : str
        from input schema "render_output"

    Returns
    -------
    stdeo : file handle or None
        destination for stderr and stdout
    """
    if outarg == 'null':
        if sys.version_info[0] >= 3:
            stdeo = subprocess.DEVNULL
        else:
            stdeo = open(os.devnull, 'wb')
        logger.info('render output is going to /dev/null')
    else:
        stdeo = None
        logger.info('render output is going to stdout')
    return stdeo


def write_to_new_stack(
        resolved,
        output_stack,
        outarg,
        overwrite_zlayer,
        args,
        results):
    """write results to render or file output

    Parameters
    ----------
    resolved : :class:`renderapi.resolvedtiles.ResolvedTiles`
        resolved tilespecs containing tilespecs to write
    output_stack : dict
        from :class:`EMaligner.schemas.output_stack`
    outarg : str
        render_output argument
    overwrite_zlayer : bool
        delete section first before overwriting?
    args : dict
        from :class:`EMaligner.schemas.EMA_Schema`
    results : dict
        results from :meth:`EMaligner.utils.solve()`

    Returns
    -------
    output_stack : dict
        representation of :class:`EMaligner.schemas.output_stack`
    """

    if output_stack['db_interface'] == 'file':
        r = resolved.to_dict()
        r['solver_args'] = dict(args)
        r['results'] = dict(results)
        output_stack['output_file'] = jsongz.dump(
                r,
                output_stack['output_file'],
                compress=output_stack['compress_output'])
        logger.info('wrote {}'.format(output_stack['output_file']))
        return output_stack

    ingestconn = make_dbconnection(output_stack, interface='render')

    logger.info(
        "\ningesting results to %s:%d %s__%s__%s" % (
            ingestconn.DEFAULT_HOST,
            ingestconn.DEFAULT_PORT,
            ingestconn.DEFAULT_OWNER,
            ingestconn.DEFAULT_PROJECT,
            output_stack['name'][0]))

    if overwrite_zlayer:
        zvalues = np.unique(np.array([t.z for t in resolved.tilespecs]))
        for zvalue in zvalues:
            renderapi.stack.delete_section(
                    output_stack['name'][0],
                    zvalue,
                    render=ingestconn)

    stdeo = get_stderr_stdout(outarg)
    pool = renderapi.client.WithPool
    if (sys.version_info[0] < 3) & (outarg == 'null'):
        pool = pool_pathos.PathosWithPool
    renderapi.client.import_tilespecs_parallel(
            output_stack['name'][0],
            resolved.tilespecs,
            sharedTransforms=resolved.transforms,
            render=ingestconn,
            close_stack=False,
            mpPool=pool,
            poolsize=args['n_parallel_jobs'],
            stderr=stdeo,
            stdout=stdeo,
            use_rest=output_stack['use_rest'])

    return output_stack


def solve(A, weights, reg, x0, rhs):
    """regularized, weighted linear least squares solve

    Parameters
    ----------
    A : :class:`scipy.sparse.csr_matrix`
        the matrix, N (equations) x M (degrees of freedom)
    weights : :class:`scipy.sparse.csr_matrix`
        N x N diagonal matrix containing weights
    reg : :class:`scipy.sparse.csr_matrix`
        M x M diagonal matrix containing regularizations
    x0 : :class:`numpy.ndarray`
        M x nsolve float constraint values for the DOFs
    rhs : :class:`numpy.ndarray`
        rhs vector(s)
        N x nsolve float right-hand-side(s)

    Returns
    -------
    results : dict
       includes solution "x" and summary metrics
    """
    time0 = time.time()
    # regularized least squares
    # ensure symmetry of K
    weights.data = np.sqrt(weights.data)
    atwt = A.transpose().dot(weights.transpose())
    wa = weights.dot(A)
    K = atwt.dot(wa) + reg

    # save this for calculating error
    w = weights.diagonal() != 0

    del wa

    # factorize, then solve, efficient for large affine
    x = np.zeros_like(x0)
    err = np.zeros((A.shape[0], x.shape[1]))
    precision = [0] * x.shape[1]

    solve = factorized(K)
    for i in range(x0.shape[1]):
        # can solve for same A, but multiple x0's
        Lm = reg.dot(x0[:, i]) + atwt.dot(weights.dot(rhs[:, i]))
        x[:, i] = solve(Lm)
        err[:, i] = A.dot(x[:, i]) - rhs[:, i]
        precision[i] = \
            np.linalg.norm(K.dot(x[:, i]) - Lm) / np.linalg.norm(Lm)
        del Lm
    del K, A, atwt

    # only report errors where weights != 0
    err = err[w, :]

    results = {}
    results['precision'] = precision
    results['error'] = np.linalg.norm(err, axis=0).tolist()
    mag = np.linalg.norm(err, axis=1)
    results['mag'] = [mag.mean(), mag.std()]
    results['err'] = [
            [m, e] for m, e in
            zip(err.mean(axis=0), err.std(axis=0))]
    results['x'] = x
    results['time'] = time.time() - time0
    return results


def message_from_solve_results(results):
    """create summarizing string message about solve for
       logging

    Parameters
    ----------
    results : dict
       returned from :meth:`EMaligner.utils.solve` or read from
       external solver results

    Returns
    -------
    message : str
       human-readable summary message
    """
    message = ' solved in %0.1f sec\n' % results['time']
    message += " precision [norm(Kx-Lm)/norm(Lm)] = "
    message += ", ".join(["%0.1e" % ix for ix in results['precision']])
    message += "\n error     [norm(Ax-b)] = "
    message += ", ".join(["%0.3f" % ix for ix in results['error']])
    message += "\n [mean(Ax) +/- std(Ax)] : "
    message += ", ".join([
        "%0.1f +/- %0.1f" % (e[0], e[1]) for e in results['err']])
    message += "\n [mean(error mag) +/- std(error mag)] : "
    message += "%0.1f +/- %0.1f" % (results['mag'][0], results['mag'][1])
    return message


def create_or_set_loading(stack):
    """creates a new stack or sets existing stack to state LOADING

    Parameters
    ----------
    stack : :class:`EMaligner.schemas.output_stack`
    """

    if stack['db_interface'] == 'file':
        return
    dbconnection = make_dbconnection(
            stack,
            interface='render')
    renderapi.stack.create_stack(
        stack['name'][0],
        render=dbconnection)


def set_complete(stack):
    """set stack state to COMPLETE

    Parameters
    ----------
    stack : :class:`EMaligner.schemas.output_stack`
    """
    if stack['db_interface'] == 'file':
        return
    dbconnection = make_dbconnection(
            stack,
            interface='render')
    renderapi.stack.set_stack_state(
        stack['name'][0],
        state='COMPLETE',
        render=dbconnection)


def get_z_values_for_stack(stack, zvals):
    """multi-interface wrapper to find overlapping z values
       between a stack and the requested range.

    Parameters
    ----------
    stack : :class:`EMaligner.schema.input_stack`
    zvals : :class:`numpy.ndarray`
        int or float. input z values

    Returns
    -------
    zvals : :class:`numpy.ndarray`
        int or float. overlapping z values

    """
    dbconnection = make_dbconnection(stack)
    if stack['db_interface'] == 'render':
        zstack = renderapi.stack.get_z_values_for_stack(
                stack['name'][0],
                render=dbconnection)
    if stack['db_interface'] == 'mongo':
        zstack = dbconnection.collection.distinct('z')
        dbconnection.client.close()
    if stack['db_interface'] == 'file':
        resolved = renderapi.resolvedtiles.ResolvedTiles(
                json=jsongz.load(stack['input_file']))
        zstack = np.unique([t.z for t in resolved.tilespecs])

    ind = np.isin(zvals, zstack)
    return zvals[ind]


def update_tilespecs(resolved, x):
    """update tilespecs with new solution

    Parameters
    ----------
    resolved : :class:`renderapi.resolvedtiles.ResolvedTiles`
        resolved tilespecs to update
    x : :class:`numpy.ndarray`
        results of solve
    """
    index = 0
    for i in range(len(resolved.tilespecs)):
        index += resolved.tilespecs[i].tforms[-1].from_solve_vec(
                x[index:, :])
    return


def blocks_from_tilespec_pair(
        ptspec, qtspec, match, pcol, qcol, ncol, matrix_assembly):
    """create sparse matrix block from tilespecs and pointmatch

    Parameters
    ----------
    ptspec : :class:`renderapi.tilespec.TileSpec`
        ptspec.tforms[-1] is an AlignerTransform object
    qtspec : :class:`renderapi.tilespec.TileSpec`
        qtspec.tforms[-1] is an AlignerTransform object
    match : dict
        pointmatch between tilepairs
    pcol : int
        index for start of column entries for p
    qcol : int
        index for start of column entries for q
    ncol : int
        total number of columns in sparse matrix
    matrix_assembly : dict
        see class matrix_assembly in schemas, sets npts

    Returns
    -------
    pblock : :class:`scipy.sparse.csr_matrix`
        block for the p tilespec/match entry. The full block can be had
        from pblock - qblock, but, it is a little faster to do
        vstack and then subtract, so p and q remain separate
    qblock : :class:`scipy.sparse.csr_matrix`
        block for the q tilespec/match entry
    w : :class:`numpy.ndarray`
        weights for the rows in pblock and qblock
    """

    if np.all(np.array(match['matches']['w']) == 0):
        return None, None, None, None

    if len(match['matches']['w']) < matrix_assembly['npts_min']:
        return None, None, None, None

    ppts = np.array(match['matches']['p']).transpose()
    qpts = np.array(match['matches']['q']).transpose()
    w = np.array(match['matches']['w'])

    if isinstance(ptspec.tforms[-1], AlignerRotationModel):
        ppts, qpts, w = AlignerRotationModel.preprocess(ppts, qpts, w)

    if ppts.shape[0] > matrix_assembly['npts_max']:
        if matrix_assembly['choose_random']:
            ind = np.arange(ppts.shape[0])
            np.random.shuffle(ind)
            ind = ind[0: matrix_assembly['npts_max']]
        else:
            ind = np.arange(matrix_assembly['npts_max'])
        ppts = ppts[ind, :]
        qpts = qpts[ind, :]
        w = w[ind]

    pblock, weights, prhs = ptspec.tforms[-1].block_from_pts(
            ppts, w, pcol, ncol)
    qblock, _, qrhs = qtspec.tforms[-1].block_from_pts(
            qpts, w, qcol, ncol)

    return pblock, qblock, weights, qrhs - prhs


def concatenate_results(results):
    """row concatenates sparse matrix blocks and associated vectors

    Parameters
    ----------
    results : list
        dict with keys "block", "weights", "rhs", "zlist"

    Returns
    -------
    A : :class:`scipy.sparse.csr_matrix`
        the concatenated matrix, N x M
    weights : :class:`scipy.sparse.csr_matrix`
        diagonal matrix containing concatenated
        weights N x N
    rhs : :class:`numpy.ndarray`
        concatenated rhs vector(s)
        float. N x nsolve
    zlist : :class:`numpy.ndarray`
        float
        concatenated z list
    """
    ind = np.flatnonzero(results)
    if ind.size == 0:
        return None, None, None, None

    A = sparse.vstack([r['block'] for r in results[ind]])
    weights = sparse.diags(
                [np.concatenate([r['weights'] for r in results[ind]])],
                [0],
                format='csr')
    rhs = np.concatenate([r.pop('rhs') for r in results[ind]])
    zlist = np.concatenate([r.pop('zlist') for r in results[ind]])

    return A, weights, rhs, zlist


def transform_match(match, ptspec, qtspec, apply_list, tforms):
    """transform the match coordinates through a subset of the
       tilespec transform list

    Parameters
    ----------
    match : dict
        one match object
    ptspec : :class:`renderapi.tilespec.TileSpec`
        the tilespec for the p coordinates
    qtspec : :class:`renderapi.tilespec.TileSpec`
        the tilespec for the q coordinates
    apply_list : list
        list of indices for the transforms
    tforms : list
        list of reference transforms

    Returns
    -------
    match : dict
        one match object, with p and q transformed

    """
    if apply_list:
        for tspec, pq in zip([ptspec, qtspec], ['p', 'q']):
            try:
                dst = renderapi.transform.estimate_dstpts(
                        [tspec.tforms[i] for i in apply_list],
                        src=np.array(match['matches'][pq]).transpose(),
                        reference_tforms=tforms)
            except IndexError:
                logger.error("argument apply_list is {} but the tilespec "
                             " for {} has tforms of length {}.".format(
                                 apply_list,
                                 tspec.tileId,
                                 len(tspec.tforms)))
                raise
            match['matches'][pq] = dst.transpose().tolist()
    return match
