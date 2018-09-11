import numpy as np
import renderapi
import argschema
from .schemas import *
from .utils import (
        make_dbconnection,
        get_tileids_and_tforms,
        get_matches,
        write_chunk_to_file,
        write_reg_and_tforms,
        write_to_new_stack,
        EMalignerException,
        logger2)
import time
import scipy.sparse as sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import factorized
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import h5py
warnings.resetwarnings()
import os
import sys
import multiprocessing
import logging
import json

logger = logging.getLogger(__name__)


def CSR_from_tile_pair(args, match, tile_ind1, tile_ind2, transform):
    # determine number of points
    npts = len(match['matches']['q'][0])
    if npts > args['matrix_assembly']['npts_max']:
        npts = args['matrix_assembly']['npts_max']
    if npts < args['matrix_assembly']['npts_min']:
        return None, None, None, None, None

    if np.all(np.array(match['matches']['w']) == 0):
        # ignore zero-weighted tile pairs
        return None, None, None, None, None

    # create arrays
    nd = npts * transform['rows_per_ptmatch'] * transform['nnz_per_row']
    ni = npts * transform['rows_per_ptmatch']
    data = np.zeros(nd).astype('float64')
    indices = np.zeros(nd).astype('int64')
    indptr = np.zeros(ni)
    weights = np.zeros(ni)

    if args['matrix_assembly']['choose_random']:
        a = np.arange(len(match['matches']['q'][0]))
        np.random.shuffle(a)
        m = a[0:npts]
    else:
        m = np.arange(npts)

    mstep = np.arange(npts) * transform['nnz_per_row']

    if args['transformation'] == 'affine_fullsize':
        # u=ax+by+c
        data[0 + mstep] = np.array(match['matches']['p'][0])[m]
        data[1 + mstep] = np.array(match['matches']['p'][1])[m]
        data[2 + mstep] = 1.0
        data[3 + mstep] = -1.0 * np.array(match['matches']['q'][0])[m]
        data[4 + mstep] = -1.0 * np.array(match['matches']['q'][1])[m]
        data[5 + mstep] = -1.0
        uindices = np.hstack((
            tile_ind1 * transform['DOF_per_tile']+np.array([0, 1, 2]),
            tile_ind2 * transform['DOF_per_tile']+np.array([0, 1, 2])))
        indices[0:npts * transform['nnz_per_row']] = np.tile(uindices, npts)
        # v=dx+ey+f
        data[
                (npts * transform['nnz_per_row']):
                (2 * npts * transform['nnz_per_row'])] = \
            data[0: npts * transform['nnz_per_row']]
        indices[npts * transform['nnz_per_row']:
                2 * npts * transform['nnz_per_row']] = \
            np.tile(uindices + 3, npts)
        indptr[0: 2 * npts] = \
            np.arange(1, 2 * npts + 1) * transform['nnz_per_row']
        weights[0: 2 * npts] = \
            np.tile(np.array(match['matches']['w'])[m], 2)
    elif args['transformation'] == 'affine':
        # u=ax+by+c
        data[0 + mstep] = np.array(match['matches']['p'][0])[m]
        data[1 + mstep] = np.array(match['matches']['p'][1])[m]
        data[2 + mstep] = 1.0
        data[3 + mstep] = -1.0 * np.array(match['matches']['q'][0])[m]
        data[4 + mstep] = -1.0 * np.array(match['matches']['q'][1])[m]
        data[5 + mstep] = -1.0
        uindices = np.hstack((
            tile_ind1 * transform['DOF_per_tile'] / 2 + np.array([0, 1, 2]),
            tile_ind2 * transform['DOF_per_tile'] / 2 + np.array([0, 1, 2])))
        indices[0: npts * transform['nnz_per_row']] = np.tile(uindices, npts)
        indptr[0: npts] = np.arange(1, npts + 1) * transform['nnz_per_row']
        weights[0: npts] = np.array(match['matches']['w'])[m]
        # don't do anything for v

    elif args['transformation'] == 'rigid':
        px = np.array(match['matches']['p'][0])[m]
        py = np.array(match['matches']['p'][1])[m]
        qx = np.array(match['matches']['q'][0])[m]
        qy = np.array(match['matches']['q'][1])[m]
        # u=ax+by+c
        data[0 + mstep] = px
        data[1 + mstep] = py
        data[2 + mstep] = 1.0
        data[3 + mstep] = -1.0 * qx
        data[4 + mstep] = -1.0 * qy
        data[5 + mstep] = -1.0
        uindices = np.hstack((
            tile_ind1 * transform['DOF_per_tile'] + np.array([0, 1, 2]),
            tile_ind2 * transform['DOF_per_tile'] + np.array([0, 1, 2])))
        indices[0: npts * transform['nnz_per_row']] = np.tile(uindices, npts)
        # v=-bx+ay+d
        data[0 + mstep + npts * transform['nnz_per_row']] = -1.0 * px
        data[1 + mstep + npts * transform['nnz_per_row']] = py
        data[2 + mstep + npts * transform['nnz_per_row']] = 1.0
        data[3 + mstep + npts * transform['nnz_per_row']] = 1.0 * qx
        data[4 + mstep + npts * transform['nnz_per_row']] = -1.0 * qy
        data[5 + mstep + npts * transform['nnz_per_row']] = -1.0
        vindices = np.hstack((
            tile_ind1 * transform['DOF_per_tile'] + np.array([1, 0, 3]),
            tile_ind2 * transform['DOF_per_tile'] + np.array([1, 0, 3])))
        indices[
                npts*transform['nnz_per_row']:
                2 * npts * transform['nnz_per_row']] = np.tile(vindices, npts)
        # du
        data[0 + mstep + 2 * npts * transform['nnz_per_row']] = \
            px - px.mean()
        data[1 + mstep + 2 * npts * transform['nnz_per_row']] = \
            py - py.mean()
        data[2 + mstep + 2 * npts * transform['nnz_per_row']] = \
            0.0
        data[3 + mstep + 2 * npts * transform['nnz_per_row']] = \
            -1.0 * (qx - qx.mean())
        data[4 + mstep + 2 * npts * transform['nnz_per_row']] = \
            -1.0 * (qy - qy.mean())
        data[5 + mstep + 2 * npts * transform['nnz_per_row']] = \
            -0.0
        indices[2 * npts * transform['nnz_per_row']:
                3 * npts * transform['nnz_per_row']] = np.tile(uindices, npts)
        # dv
        data[0 + mstep + 3 * npts * transform['nnz_per_row']] = \
            -1.0 * (px - px.mean())
        data[1 + mstep + 3 * npts * transform['nnz_per_row']] = \
            py - py.mean()
        data[2 + mstep + 3 * npts * transform['nnz_per_row']] = \
            0.0
        data[3 + mstep + 3 * npts * transform['nnz_per_row']] = \
            1.0 * (qx - qx.mean())
        data[4 + mstep + 3 * npts * transform['nnz_per_row']] = \
            -1.0 * (qy - qy.mean())
        data[5 + mstep + 3 * npts * transform['nnz_per_row']] = \
            -0.0
        indices[3 * npts * transform['nnz_per_row']:
                4 * npts * transform['nnz_per_row']] = np.tile(uindices, npts)

        indptr[0: transform['rows_per_ptmatch'] * npts] = \
            np.arange(1, transform['rows_per_ptmatch'] * npts + 1) * \
            transform['nnz_per_row']
        weights[0: transform['rows_per_ptmatch'] * npts] = \
            np.tile(np.array(
                match['matches']['w'])[m],
                transform['rows_per_ptmatch'])

    return data, indices, indptr, weights, npts


def calculate_processing_chunk(fargs):
    # set up for calling using multiprocessing pool
    [zvals, sectionIds, zloc, args, tile_ids, transform] = fargs

    dbconnection = make_dbconnection(args['pointmatch'])
    sorter = np.argsort(tile_ids)

    # this dict will get returned
    chunk = {}
    chunk['tiles_used'] = []
    chunk['data'] = None
    chunk['indices'] = None
    chunk['indptr'] = None
    chunk['weights'] = None
    chunk['nchunks'] = 0
    chunk['zlist'] = []

    pstr = '  proc%d: ' % zloc

    # get point matches
    t0 = time.time()
    matches = get_matches(
            sectionIds[0],
            sectionIds[1],
            args['pointmatch'],
            dbconnection)
    if len(matches) == 0:
        return chunk

    # extract IDs for fast checking
    pids = []
    qids = []
    for m in matches:
        pids.append(m['pId'])
        qids.append(m['qId'])
    pids = np.array(pids)
    qids = np.array(qids)

    # remove matches that don't have both IDs in tile_ids
    instack = np.in1d(pids, tile_ids) & np.in1d(qids, tile_ids)
    matches = matches[instack]
    pids = pids[instack]
    qids = qids[instack]

    if len(matches) == 0:
        logger.debug(
                "%sno tile pairs in "
                "stack for pointmatch groupIds %s and %s" % (
                    pstr, sectionIds[0], sectionIds[1]))
        return chunk

    logger.debug(
            "%sloaded %d matches, using %d, "
            "for groupIds %s and %s in %0.1f sec "
            "using interface: %s" % (
                pstr,
                instack.size,
                len(matches),
                sectionIds[0],
                sectionIds[1],
                time.time() - t0,
                args['pointmatch']['db_interface']))

    t0 = time.time()
    # for the given point matches, these are the indices in tile_ids
    # these determine the column locations in A for each tile pair
    # this is a fast version of np.argwhere() loop
    pinds = sorter[np.searchsorted(tile_ids, pids, sorter=sorter)]
    qinds = sorter[np.searchsorted(tile_ids, qids, sorter=sorter)]

    # conservative pre-allocation of the arrays we need to populate
    # will truncate at the end
    nmatches = len(matches)
    nd = (
            transform['nnz_per_row'] *
            transform['rows_per_ptmatch'] *
            args['matrix_assembly']['npts_max'] *
            nmatches)
    ni = (
            transform['rows_per_ptmatch'] *
            args['matrix_assembly']['npts_max'] *
            nmatches)
    data = np.zeros(nd).astype('float64')
    indices = np.zeros(nd).astype('int64')
    indptr = np.zeros(ni + 1).astype('int64')
    weights = np.zeros(ni).astype('float64')

    # see definition of CSR format, wikipedia for example
    indptr[0] = 0

    # track how many rows
    nrows = 0

    tilepair_weightfac = tilepair_weight(
            float(sectionIds[0]),
            float(sectionIds[1]),
            args['matrix_assembly'])

    for k in np.arange(nmatches):
        # create the CSR sub-matrix for this tile pair
        d, ind, iptr, wts, npts = CSR_from_tile_pair(
                args,
                matches[k],
                pinds[k],
                qinds[k],
                transform)
        if d is None:
            continue  # if npts<nmin, for example

        # add both tile ids to the list
        chunk['tiles_used'].append(matches[k]['pId'])
        chunk['tiles_used'].append(matches[k]['qId'])

        # add sub-matrix to global matrix
        global_dind = np.arange(
                npts *
                transform['rows_per_ptmatch'] *
                transform['nnz_per_row']) + \
            nrows*transform['nnz_per_row']
        data[global_dind] = d
        indices[global_dind] = ind

        global_rowind = \
            np.arange(npts * transform['rows_per_ptmatch']) + nrows
        weights[global_rowind] = wts * tilepair_weightfac
        indptr[global_rowind + 1] = iptr + indptr[nrows]

        nrows += wts.size

    del matches
    # truncate, because we allocated conservatively
    data = data[0: nrows * transform['nnz_per_row']]
    indices = indices[0: nrows * transform['nnz_per_row']]
    indptr = indptr[0: nrows + 1]
    weights = weights[0: nrows]

    chunk['data'] = np.copy(data)
    chunk['weights'] = np.copy(weights)
    chunk['indices'] = np.copy(indices)
    chunk['indptr'] = np.copy(indptr)
    chunk['zlist'].append(float(sectionIds[0]))
    chunk['zlist'].append(float(sectionIds[1]))
    chunk['zlist'] = np.array(chunk['zlist'])
    del data, indices, indptr, weights

    return chunk


def tilepair_weight(z1, z2, matrix_assembly):
    if z1 == z2:
        tp_weight = matrix_assembly['montage_pt_weight']
    else:
        tp_weight = matrix_assembly['cross_pt_weight']
        if matrix_assembly['inverse_dz']:
            tp_weight = tp_weight/np.abs(z2-z1+1)
    return tp_weight


def mat_stats(m, name):
    logger.debug(
            ' matrix %s: ' % name +
            ' format: ', m.getformat(),
            ', shape: ', m.shape,
            ' nnz: ', m.nnz)
    if m.shape[0] == m.shape[1]:
        asymm = np.any(m.transpose().data != m.data)
        print(' symm: ', not asymm)


class EMaligner(argschema.ArgSchemaParser):
    default_schema = EMA_Schema

    def run(self):
        logger.setLevel(self.args['log_level'])
        logger2.setLevel(self.args['log_level'])
        t0 = time.time()
        zvals = np.arange(
                self.args['first_section'],
                self.args['last_section'] + 1)

        ingestconn = None
        # make a connection to the new stack
        if self.args['output_mode'] == 'stack':
            ingestconn = make_dbconnection(self.args['output_stack'])
            renderapi.stack.create_stack(
                    self.args['output_stack']['name'],
                    render=ingestconn)

        # montage
        if self.args['solve_type'] == 'montage':
            # check for zvalues in stack
            tmp = self.args['input_stack']['db_interface']
            self.args['input_stack']['db_interface'] = 'render'
            conn = make_dbconnection(self.args['input_stack'])
            self.args['input_stack']['db_interface'] = tmp
            z_in_stack = renderapi.stack.get_z_values_for_stack(
                    self.args['input_stack']['name'],
                    render=conn)
            newzvals = []
            for z in zvals:
                if z in z_in_stack:
                    newzvals.append(z)
            zvals = np.array(newzvals)
            for z in zvals:
                self.results = self.assemble_and_solve(
                        np.array([z]),
                        ingestconn)
        # 3D
        elif self.args['solve_type'] == '3D':
            self.results = self.assemble_and_solve(zvals, ingestconn)

        if ingestconn is not None:
            if self.args['close_stack']:
                renderapi.stack.set_stack_state(
                        self.args['output_stack']['name'],
                        state='COMPLETE',
                        render=ingestconn)
        logger.info(' total time: %0.1f' % (time.time() - t0))

    def assemble_and_solve(self, zvals, ingestconn):
        t0 = time.time()

        self.set_transform()

        if self.args['ingest_from_file'] != '':
            assemble_result = self.assemble_from_hdf5(
                    self.args['ingest_from_file'],
                    zvals,
                    read_data=False)
            if self.args['transformation'] == 'affine':
                x = self.combine_x_affine(
                        assemble_result['tforms'][0],
                        assemble_result['tforms'][1])
            else:
                x = assemble_result['tforms'][0]
            results = {}

        else:
            # assembly
            if self.args['assemble_from_file'] != '':
                assemble_result = self.assemble_from_hdf5(
                        self.args['assemble_from_file'],
                        zvals)
            else:
                assemble_result = self.assemble_from_db(zvals)

            if assemble_result['A'] is not None:
                mat_stats(assemble_result['A'], 'A')

            self.ntiles_used = assemble_result['tids'].size
            logger.info(' A created in %0.1f seconds' % (time.time() - t0))

            if self.args['profile_data_load']:
                raise EMalignerException(
                        "exiting after timing profile")

            # solve
            message, x, results = \
                self.solve_or_not(
                       assemble_result['A'],
                       assemble_result['weights'],
                       assemble_result['reg'],
                       assemble_result['tforms'])
            logger.info('\n' + message)
            if assemble_result['A'] is not None:
                results['Ashape'] = assemble_result['A'].shape
            del assemble_result['A']

        if self.args['output_mode'] == 'stack':
            write_to_new_stack(
                    self.args['input_stack'],
                    self.args['output_stack']['name'],
                    self.args['transformation'],
                    assemble_result['tspecs'],
                    assemble_result['shared_tforms'],
                    x,
                    ingestconn,
                    assemble_result['unused_tids'],
                    self.args['render_output'],
                    self.args['output_stack']['use_rest'],
                    self.args['overwrite_zlayer'])
            if self.args['render_output'] == 'stdout':
                logger.info(message)
        del assemble_result['shared_tforms'], assemble_result['tspecs'], x

        return results

    assemble_struct = {
                'A': None,
                'weights': None,
                'reg': None,
                'tspecs': None,
                'tforms': None,
                'tids': None,
                'shared_tforms': None,
                'unused_tids': None}

    def assemble_from_hdf5(self, filename, zvals, read_data=True):
        assemble_result = dict(self.assemble_struct)

        from_stack = get_tileids_and_tforms(
                        self.args['input_stack'],
                        self.args['transformation'],
                        zvals)

        assemble_result['shared_tforms'] = from_stack.pop('shared_tforms')

        with h5py.File(filename, 'r') as f:
            assemble_result['tids'] = np.array(
                    f.get('used_tile_ids')[()]).astype('U')
            assemble_result['unused_tids'] = np.array(
                    f.get('unused_tile_ids')[()]).astype('U')
            k = 0
            assemble_result['tforms'] = []
            while True:
                name = 'transforms_%d' % k
                if name in f.keys():
                    assemble_result['tforms'].append(f.get(name)[()])
                    k += 1
                else:
                    break

            reg = f.get('lambda')[()]
            datafile_names = f.get('datafile_names')[()]
            file_args = json.loads(f.get('input_args')[()][0])

        # get the tile IDs and transforms
        tile_ind = np.in1d(from_stack['tids'], assemble_result['tids'])
        assemble_result['tspecs'] = from_stack['tspecs'][tile_ind]

        outr = sparse.eye(reg.size, format='csr')
        outr.data = reg
        assemble_result['reg'] = outr

        if read_data:
            data = np.array([]).astype('float64')
            weights = np.array([]).astype('float64')
            indices = np.array([]).astype('int64')
            indptr = np.array([]).astype('int64')

            fdir = os.path.dirname(filename)
            i = 0
            for fname in datafile_names:
                with h5py.File(os.path.join(fdir, fname), 'r') as f:
                    data = np.append(data, f.get('data')[()])
                    indices = np.append(indices, f.get('indices')[()])
                    if i == 0:
                        indptr = np.append(indptr, f.get('indptr')[()])
                        i += 1
                    else:
                        indptr = np.append(
                                indptr,
                                f.get('indptr')[()][1:] + indptr[-1])
                    weights = np.append(weights, f.get('weights')[()])
                    logger.info('  %s read' % fname)

            assemble_result['A'] = csr_matrix((data, indices, indptr))

            outw = sparse.eye(weights.size, format='csr')
            outw.data = weights
            assemble_result['weights'] = outw

        # alert about differences between this call and the original
        for k in file_args.keys():
            if k in self.args.keys():
                if file_args[k] != self.args[k]:
                    logger.warning("for key \"%s\" " % k)
                    logger.warning("  from file: " + str(file_args[k]))
                    logger.warning("  this call: " + str(self.args[k]))
            else:
                logger.warning("for key \"%s\" " % k)
                logger.warning("  file     : " + str(file_args[k]))
                logger.warning("  this call: not specified")

        logger.info("csr inputs read from files listed in : "
                    "%s" % self.args['assemble_from_file'])

        return assemble_result

    def assemble_from_db(self, zvals):
        assemble_result = dict(self.assemble_struct)

        from_stack = get_tileids_and_tforms(
                        self.args['input_stack'],
                        self.args['transformation'],
                        zvals)
        assemble_result['shared_tforms'] = from_stack.pop('shared_tforms')

        # create A matrix in compressed sparse row (CSR) format
        CSR_A = self.create_CSR_A(
                from_stack['tids'],
                zvals,
                from_stack['sectionIds'])
        assemble_result['A'] = CSR_A.pop('A')
        assemble_result['weights'] = CSR_A.pop('weights')

        # some book-keeping if there were some unused tiles
        tile_ind = np.in1d(from_stack['tids'], CSR_A['tiles_used'])
        assemble_result['tspecs'] = from_stack['tspecs'][tile_ind]
        assemble_result['tids'] = \
            from_stack['tids'][tile_ind]
        assemble_result['unused_tids'] = \
            from_stack['tids'][np.invert(tile_ind)]

        # remove columns in A for unused tiles
        slice_ind = np.repeat(
                tile_ind,
                self.transform['DOF_per_tile'] / len(from_stack['tforms']))
        if self.args['output_mode'] != 'hdf5':
            # for large matrices,
            # this might be expensive to perform on CSR format
            assemble_result['A'] = assemble_result['A'][:, slice_ind]

        assemble_result['tforms'] = []
        for j in np.arange(len(from_stack['tforms'])):
            assemble_result['tforms'].append(
                    from_stack['tforms'][j][slice_ind])
        del from_stack, CSR_A['tiles_used'], tile_ind

        # create the regularization vectors
        assemble_result['reg'] = self.create_regularization(
                assemble_result['tforms'])

        # output the regularization vectors to hdf5 file
        if self.args['output_mode'] == 'hdf5':
            write_reg_and_tforms(
                    dict(self.args),
                    CSR_A['metadata'],
                    assemble_result['tforms'],
                    assemble_result['reg'],
                    assemble_result['tids'],
                    assemble_result['unused_tids'])

        return assemble_result

    def set_transform(self):
        self.transform = {}
        self.transform['name'] = self.args['transformation']
        if self.args['transformation'] == 'affine':
            self.transform['DOF_per_tile'] = 6
            self.transform['nnz_per_row'] = 6
            self.transform['rows_per_ptmatch'] = 1
        if self.args['transformation'] == 'affine_fullsize':
            self.transform['DOF_per_tile'] = 6
            self.transform['nnz_per_row'] = 6
            self.transform['rows_per_ptmatch'] = 2
        if self.args['transformation'] == 'rigid':
            self.transform['DOF_per_tile'] = 4
            self.transform['nnz_per_row'] = 6
            self.transform['rows_per_ptmatch'] = 4

    def determine_zvalue_pairs(self, zvals, sectionIds):
        # create all possible pairs, given zvals and depth
        zs = np.array(sectionIds).astype(float)
        pairs = []
        for z in zs:
            i = 0
            while i <= self.args['matrix_assembly']['depth']:
                if z+i in zvals:
                    pairs.append([str(z), str(z + i)])
                i += 1
        return np.array(pairs)

    def concatenate_chunks(self, chunks):
        i = 0
        while chunks[i]['data'] is None:
            if i == len(chunks) - 1:
                break
            i += 1
        c0 = chunks[i]
        for c in chunks[(i + 1):]:
            if c['data'] is not None:
                for ckey in ['data', 'weights', 'indices', 'zlist']:
                    c0[ckey] = np.append(c0[ckey], c[ckey])
                ckey = 'indptr'
                lastptr = c0[ckey][-1]
                c0[ckey] = np.append(c0[ckey], c[ckey][1:] + lastptr)
        return c0

    def create_CSR_A(self, tile_ids, zvals, sectionIds):
        func_result = {
                'A': None,
                'weights': None,
                'tiles_used': None,
                'metadata': None}

        pool = multiprocessing.Pool(self.args['n_parallel_jobs'])

        pairs = self.determine_zvalue_pairs(zvals, sectionIds)
        npairs = pairs.shape[0]

        # split up the work
        if self.args['hdf5_options']['chunks_per_file'] == -1:
            proc_chunks = [np.arange(npairs)]
        else:
            proc_chunks = np.array_split(
                    np.arange(npairs),
                    np.ceil(
                        float(npairs) /
                        self.args['hdf5_options']['chunks_per_file']))

        fargs = []
        for i in np.arange(npairs):
            fargs.append([
                zvals,
                pairs[i, :],
                i,
                self.args,
                tile_ids,
                self.transform])
        results = pool.map(calculate_processing_chunk, fargs)
        pool.close()
        pool.join()

        tiles_used = []
        for i in np.arange(len(results)):
            tiles_used += results[i]['tiles_used']
        func_result['tiles_used'] = np.array(tiles_used)

        func_result['metadata'] = []
        if self.args['output_mode'] == 'hdf5':
            results = np.array(results)
            for pchunk in proc_chunks:
                cat_chunk = self.concatenate_chunks(results[pchunk])
                if cat_chunk['data'] is not None:
                    c = csr_matrix((
                        cat_chunk['data'],
                        cat_chunk['indices'],
                        cat_chunk['indptr']))
                    fname = self.args['hdf5_options']['output_dir'] + \
                        '/%d_%d.h5' % (
                                cat_chunk['zlist'].min(),
                                cat_chunk['zlist'].max())
                    func_result['metadata'].append(
                            write_chunk_to_file(
                                fname,
                                c,
                                cat_chunk['weights']))

        else:
            data = np.array([]).astype('float64')
            weights = np.array([]).astype('float64')
            indices = np.array([]).astype('int64')
            indptr = np.array([]).astype('int64')
            for i in np.arange(len(results)):
                if results[i]['data'] is not None:
                    data = np.append(data, results[i]['data'])
                    indices = np.append(indices, results[i]['indices'])
                    weights = np.append(weights, results[i]['weights'])
                    if indptr.size == 0:
                        indptr = np.append(indptr, results[i]['indptr'])
                    else:
                        indptr = np.append(
                                indptr,
                                results[i]['indptr'][1:] + indptr[-1])
                results[i] = None
            A = csr_matrix((data, indices, indptr))
            outw = sparse.eye(weights.size, format='csr')
            outw.data = weights
            func_result['A'] = A
            func_result['weights'] = outw

        return func_result

    def create_regularization(self, tile_tforms):
        # affine (half-size) or any
        # other transform, we only need the first one:
        tile_tforms = tile_tforms[0]

        # create a regularization vector
        reg = np.ones_like(tile_tforms).astype('float64') * \
            self.args['regularization']['default_lambda']
        if 'affine' in self.args['transformation']:
            reg[2::3] = reg[2::3] * \
                self.args['regularization']['translation_factor']
        elif self.args['transformation'] == 'rigid':
            reg[2::4] = reg[2::4] * \
                self.args['regularization']['translation_factor']
            reg[3::4] = reg[3::4] * \
                self.args['regularization']['translation_factor']
        if self.args['regularization']['freeze_first_tile']:
            reg[0:self.transform['DOF_per_tile']] = 1e15

        outr = sparse.eye(reg.size, format='csr')
        outr.data = reg
        return outr

    def combine_x_affine(self, xu, xv):
        x = np.zeros(xu.size * 2).astype('float64')
        for i in np.arange(3):
            x[i::6] = xu[i::3]
            x[i + 3::6] = xv[i::3]
        return x

    def solve_or_not(self, A, weights, reg, filt_tforms):
        t0 = time.time()
        # not
        if self.args['output_mode'] in ['hdf5']:
            message = '*****\nno solve for file output\n'
            message += 'solve from the files you just wrote:\n\n'
            message += 'python '
            for arg in sys.argv:
                message += arg+' '
            message = message + '--assemble_from_file ' + \
                self.args['hdf5_options']['output_dir']
            message = message + ' --output_mode none'
            message += '\n\nor, run it again to solve with no output:\n\n'
            message += 'python '
            for arg in sys.argv:
                message += arg + ' '
            message = message.replace(' hdf5 ', ' none ')
            x = None
            results = None
        else:
            # regularized least squares
            # ensure symmetry of K
            weights.data = np.sqrt(weights.data)
            rtWA = weights.dot(A)
            K = rtWA.transpose().dot(rtWA) + reg

            logger.info(' K created in %0.1f seconds' % (time.time() - t0))
            t0 = time.time()
            del weights, rtWA

            # factorize, then solve, efficient for large affine
            solve = factorized(K)
            if self.args['transformation'] == 'affine':
                # affine assembles only half the matrix
                # then applies the LU decomposition to
                # the u and v transforms separately
                Lm = reg.dot(filt_tforms[0])
                xu = solve(Lm)
                erru = A.dot(xu)
                precisionu = \
                    np.linalg.norm(K.dot(xu) - Lm) / np.linalg.norm(Lm)

                Lm = reg.dot(filt_tforms[1])
                xv = solve(Lm)
                errv = A.dot(xv)
                precisionv = \
                    np.linalg.norm(K.dot(xv) - Lm) / np.linalg.norm(Lm)
                precision = np.sqrt(precisionu ** 2 + precisionv ** 2)

                # recombine
                x = self.combine_x_affine(xu, xv)
                err = np.hstack((erru, errv))
                del xu, xv, erru, errv, precisionu, precisionv
            else:
                # simpler case for rigid, or
                # affine_fullsize, but 2x larger than affine
                Lm = reg.dot(filt_tforms[0])
                x = solve(Lm)
                err = A.dot(x)
                precision = \
                    np.linalg.norm(K.dot(x) - Lm) / np.linalg.norm(Lm)
            del K, Lm

            error = np.linalg.norm(err)

            results = {}
            results['time'] = time.time()-t0
            results['precision'] = precision
            results['error'] = error
            results['err'] = [np.abs(err).mean(),np.abs(err).std()]

            message = ' solved in %0.1f sec\n' % (time.time() - t0)
            message += (
                    " precision [norm(Kx-Lm)/norm(Lm)] "
                    "= %0.1e\n" % precision)
            message += (
                    " error     [norm(Ax-b)] "
                    "= %0.3f\n" % error)
            message += (
                    " [mean(|Ax|)+/-std(|Ax|)] : "
                    "%0.1f +/- %0.1f pixels" % (
                        np.abs(err).mean(),
                        np.abs(err).std()))

            if self.args['transformation'] == 'rigid':
                scale = np.sqrt(
                        np.power(x[0::self.transform['DOF_per_tile']], 2.0) +
                        np.power(x[1::self.transform['DOF_per_tile']], 2.0))
            if 'affine' in self.args['transformation']:
                scale = np.sqrt(
                        np.power(x[0::self.transform['DOF_per_tile']], 2.0) +
                        np.power(x[1::self.transform['DOF_per_tile']], 2.0))
                scale += np.sqrt(
                        np.power(x[3::self.transform['DOF_per_tile']], 2.0) +
                        np.power(x[4::self.transform['DOF_per_tile']], 2.0))
                scale /= 2
            scale = scale.sum() / self.ntiles_used
            results['scale'] = scale
            message += '\n avg scale = %0.2f' % scale

        return message, x, results


if __name__ == '__main__':
    mod = EMaligner(schema_type=EMA_Schema)
    mod.run()
