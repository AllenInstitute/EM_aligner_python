import numpy as np
import renderapi
import argschema
from .schemas import EMA_Schema
from . import utils
from .transform.transform import AlignerTransform
import time
import scipy.sparse as sparse
from scipy.sparse import csr_matrix
import warnings
import os
import sys
import multiprocessing
import logging
import json
import h5py
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.resetwarnings()

logger = logging.getLogger(__name__)


def calculate_processing_chunk(fargs):
    # set up for calling using multiprocessing pool
    [pair, zloc, args, tile_ids] = fargs

    dbconnection = utils.make_dbconnection(args['pointmatch'])
    sorter = np.argsort(tile_ids)

    # this dict will get returned
    chunk = {}
    chunk['tiles_used'] = np.zeros(tile_ids.size).astype(bool)
    chunk['data'] = None
    chunk['indices'] = None
    chunk['indptr'] = None
    chunk['weights'] = None
    chunk['nchunks'] = 0
    chunk['zlist'] = []

    pstr = '  proc%d: ' % zloc

    # get point matches
    t0 = time.time()

    matches = utils.get_matches(
            pair['section1'],
            pair['section2'],
            args['pointmatch'],
            dbconnection)

    if len(matches) == 0:
        return chunk

    # extract IDs for fast checking
    pid_set = set(m['pId'] for m in matches)
    qid_set = set(m['qId'] for m in matches)

    tile_set = set(tile_ids)

    pid_set.intersection_update(tile_set)
    qid_set.intersection_update(tile_set)

    matches = [m for m in matches if m['pId']
               in pid_set and m['qId'] in qid_set]

    pids = np.array([m['pId'] for m in matches])
    qids = np.array([m['qId'] for m in matches])

    if len(matches) == 0:
        logger.debug(
            "%sno tile pairs in "
            "stack for pointmatch groupIds %s and %s" % (
                pstr, pair['section1'], pair['section2']))
        return chunk

    logger.debug(
            "%sloaded %d matches, using %d, "
            "for groupIds %s and %s in %0.1f sec "
            "using interface: %s" % (
                pstr,
                len(pid_set.union(qid_set)),
                len(matches),
                pair['section1'],
                pair['section2'],
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
    transform = AlignerTransform(
        args['transformation'],
        fullsize=args['fullsize_transform'],
        order=args['poly_order'])
    nd = (
        transform.nnz_per_row *
        transform.rows_per_ptmatch *
        args['matrix_assembly']['npts_max'] *
        nmatches)
    ni = (
        transform.rows_per_ptmatch *
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
        pair['z1'],
        pair['z2'],
        args['matrix_assembly'])

    for k in np.arange(nmatches):
        # create the CSR sub-matrix for this tile pair
        d, ind, iptr, wts, npts = transform.CSR_from_tilepair(
            matches[k],
            pinds[k],
            qinds[k],
            args['matrix_assembly']['npts_min'],
            args['matrix_assembly']['npts_max'],
            args['matrix_assembly']['choose_random'])

        if d is None:
            continue  # if npts<nmin, or all weights=0

        # note both as used
        chunk['tiles_used'][pinds[k]] = True
        chunk['tiles_used'][qinds[k]] = True

        # add sub-matrix to global matrix
        global_dind = np.arange(
            npts *
            transform.rows_per_ptmatch *
            transform.nnz_per_row) + \
            nrows*transform.nnz_per_row
        data[global_dind] = d
        indices[global_dind] = ind

        global_rowind = \
            np.arange(npts * transform.rows_per_ptmatch) + nrows
        weights[global_rowind] = wts * tilepair_weightfac
        indptr[global_rowind + 1] = iptr + indptr[nrows]

        nrows += wts.size

    del matches
    # truncate, because we allocated conservatively
    data = data[0: nrows * transform.nnz_per_row]
    indices = indices[0: nrows * transform.nnz_per_row]
    indptr = indptr[0: nrows + 1]
    weights = weights[0: nrows]

    chunk['data'] = np.copy(data)
    chunk['weights'] = np.copy(weights)
    chunk['indices'] = np.copy(indices)
    chunk['indptr'] = np.copy(indptr)
    chunk['zlist'].append(pair['z1'])
    chunk['zlist'].append(pair['z2'])
    chunk['zlist'] = np.array(chunk['zlist'])
    del data, indices, indptr, weights

    return chunk


def tilepair_weight(z1, z2, matrix_assembly):
    if matrix_assembly['explicit_weight_by_depth'] is not None:
        ind = matrix_assembly['depth'].index(int(np.abs(z1 - z2)))
        tp_weight = matrix_assembly['explicit_weight_by_depth'][ind]
    else:
        if z1 == z2:
            tp_weight = matrix_assembly['montage_pt_weight']
        else:
            tp_weight = matrix_assembly['cross_pt_weight']
            if matrix_assembly['inverse_dz']:
                tp_weight = tp_weight/(np.abs(z2 - z1) + 1)
    return tp_weight


class EMaligner(argschema.ArgSchemaParser):
    default_schema = EMA_Schema

    def run(self):
        logger.setLevel(self.args['log_level'])
        utils.logger.setLevel(self.args['log_level'])
        t0 = time.time()
        zvals = np.arange(
            self.args['first_section'],
            self.args['last_section'] + 1)


        # read in the tilespecs
        self.resolvedtiles = utils.get_resolved_tilespecs(
            self.args['input_stack'],
            self.args['transformation'],
            self.args['n_parallel_jobs'],
            zvals,
            fullsize=self.args['fullsize_transform'],
            order=self.args['poly_order'])

        # the parallel workers will need this stack ready
        if self.args['output_mode'] == 'stack':
            utils.create_or_set_loading(self.args['output_stack'])

        # montage
        if self.args['solve_type'] == 'montage':
            zvals = utils.get_z_values_for_stack(
                    self.args['input_stack'],
                    zvals)
            for z in zvals:
                self.results = self.assemble_and_solve(np.array([z]))

        # 3D
        elif self.args['solve_type'] == '3D':
            self.results = self.assemble_and_solve(zvals)

        if (self.args['output_mode'] == 'stack') & self.args['close_stack']:
            utils.set_complete(self.args['output_stack'])

        logger.info(' total time: %0.1f' % (time.time() - t0))

    def assemble_and_solve(self, zvals):
        t0 = time.time()
        if self.args['ingest_from_file'] != '':
            assemble_result = self.assemble_from_hdf5(
                self.args['ingest_from_file'],
                zvals,
                read_data=False)
            results = {}
            results['x'] = assemble_result['x']

        else:
            if self.args['assemble_from_file'] != '':
                assemble_result = self.assemble_from_hdf5(
                    self.args['assemble_from_file'],
                    zvals)
            else:
                assemble_result = self.assemble_from_db(zvals)

            self.ntiles_used = np.count_nonzero(assemble_result['tiles_used'])
            logger.info(' A created in %0.1f seconds' % (time.time() - t0))

            if self.args['profile_data_load']:
                raise utils.EMalignerException(
                    "exiting after timing profile")

            # solve
            message, results = \
                self.solve_or_not(
                    assemble_result['A'],
                    assemble_result['weights'],
                    assemble_result['reg'],
                    assemble_result['x'])
            logger.info('\n' + message)
            del assemble_result['A']

        if self.args['output_mode'] == 'stack':
            solved_resolved = utils.update_tilespecs(
                    self.resolvedtiles,
                    results['x'],
                    assemble_result['tiles_used'])
            utils.write_to_new_stack(
                    solved_resolved,
                    self.args['output_stack'],
                    self.args['render_output'],
                    self.args['overwrite_zlayer'])
            if self.args['render_output'] == 'stdout':
                logger.info(message)
        del assemble_result['x']

        return results

    def assemble_from_hdf5(self, filename, zvals, read_data=True):
        assemble_result = {}

        with h5py.File(filename, 'r') as f:
            assemble_result['tids'] = np.array(
                f.get('used_tile_ids')[()]).astype('U')
            assemble_result['unused_tids'] = np.array(
                f.get('unused_tile_ids')[()]).astype('U')
            k = 0
            assemble_result['x'] = []
            while True:
                name = 'transforms_%d' % k
                if name in f.keys():
                    assemble_result['x'].append(f.get(name)[()])
                    k += 1
                else:
                    break

            if len(assemble_result['x']) == 1:
                n = assemble_result['x'][0].size
                assemble_result['x'] = np.array(
                    assemble_result['x']).flatten().reshape((n, 1))
            else:
                assemble_result['x'] = np.transpose(
                    np.array(assemble_result['x']))

            reg = f.get('lambda')[()]
            datafile_names = f.get('datafile_names')[()]
            file_args = json.loads(f.get('input_args')[()][0])

        # get the tile IDs and transforms
        tids = np.array([
            t.tileId for t in self.resolvedtiles.tilespecs])
        assemble_result['tiles_used'] = np.in1d(tids, assemble_result['tids'])

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
        # create A matrix in compressed sparse row (CSR) format
        CSR_A = self.create_CSR_A(self.resolvedtiles)

        assemble_result = {}
        assemble_result['A'] = CSR_A.pop('A')
        assemble_result['weights'] = CSR_A.pop('weights')
        assemble_result['tiles_used'] = CSR_A.pop('tiles_used')
        assemble_result['reg'] = CSR_A.pop('reg')
        assemble_result['x'] = CSR_A.pop('x')

        # output the regularization vectors to hdf5 file
        if self.args['output_mode'] == 'hdf5':
            alltids = np.array([
                t.tileId for t in self.resolvedtiles.tilespecs])

            utils.write_reg_and_tforms(
                dict(self.args),
                CSR_A['metadata'],
                assemble_result['x'],
                assemble_result['reg'],
                alltids[assemble_result['tiles_used']],
                alltids[np.invert(assemble_result['tiles_used'])])

        return assemble_result

    def concatenate_results(self, results):
        result = {}

        if np.all([r['data'] is None for r in results]):
            return {'data': None}

        result['data'] = np.concatenate([
            results[i]['data'] for i in range(len(results))
            if results[i]['data'] is not None]).astype('float64')
        result['weights'] = np.concatenate([
            results[i]['weights'] for i in range(len(results))
            if results[i]['data'] is not None]).astype('float64')
        result['indices'] = np.concatenate([
            results[i]['indices'] for i in range(len(results))
            if results[i]['data'] is not None]).astype('int64')
        result['zlist'] = np.concatenate([
            results[i]['zlist'] for i in range(len(results))
            if results[i]['data'] is not None])
        # Pointers need to be handled differently,
        # since you need to sum the arrays
        result['indptr'] = [results[i]['indptr']
                            for i in range(len(results))
                            if results[i]['data'] is not None]
        indptr_cumends = np.cumsum([i[-1] for i in result['indptr']])
        result['indptr'] = np.concatenate(
            [j if i == 0 else j[1:]+indptr_cumends[i-1] for i, j
             in enumerate(result['indptr'])]).astype('int64')

        return result

    def create_CSR_A(self, resolved):
        func_result = {
            'A': None,
            'x': None,
            'reg': None,
            'weights': None,
            'tiles_used': None,
            'metadata': None}

        pool = multiprocessing.Pool(self.args['n_parallel_jobs'])

        pairs = utils.determine_zvalue_pairs(
                resolved,
                self.args['matrix_assembly']['depth'])

        npairs = len(pairs)
        tile_ids = np.array([t.tileId for t in resolved.tilespecs])
        fargs = [[pairs[i], i, self.args, tile_ids] for i in range(npairs)]

        with renderapi.client.WithPool(self.args['n_parallel_jobs']) as pool:
            results = np.array(pool.map(calculate_processing_chunk, fargs))

        func_result['tiles_used'] = results[0]['tiles_used']
        for result in results[1:]:
            func_result['tiles_used'] = \
                    func_result['tiles_used'] | result['tiles_used']

        func_result['x'] = []
        reg = []
        for t in np.array(resolved.tilespecs)[func_result['tiles_used']]:
            func_result['x'].append(t.tforms[-1].to_solve_vec())
            reg.append(
                    t.tforms[-1].regularization(self.args['regularization']))
        func_result['x'] = np.concatenate(func_result['x'])
        reg = np.concatenate(reg)
        func_result['reg'] = sparse.eye(reg.size, format='csr')
        func_result['reg'].data = reg

        if self.args['output_mode'] == 'hdf5':
            results = np.array(results)

            if self.args['hdf5_options']['chunks_per_file'] == -1:
                proc_chunks = [np.arange(npairs)]
            else:
                proc_chunks = np.array_split(
                    np.arange(npairs),
                    np.ceil(
                        float(npairs) /
                        self.args['hdf5_options']['chunks_per_file']))
            func_result['metadata'] = []
            for pchunk in proc_chunks:
                cat_chunk = self.concatenate_results(results[pchunk])
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
                        utils.write_chunk_to_file(
                            fname,
                            c,
                            cat_chunk['weights']))

        else:
            result = self.concatenate_results(results)
            A = csr_matrix((
                result['data'],
                result['indices'],
                result['indptr']))
            outw = sparse.eye(result['weights'].size, format='csr')
            outw.data = result['weights']
            slice_ind = np.concatenate(
                    [np.repeat(
                        func_result['tiles_used'][i],
                        resolved.tilespecs[i].tforms[-1].DOF_per_tile)
                     for i in range(tile_ids.size)])
            func_result['A'] = A[:, slice_ind]
            func_result['weights'] = outw

        return func_result

    def solve_or_not(self, A, weights, reg, x0):
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
            results = None
        else:
            results = utils.solve(A, weights, reg, x0)
            message = utils.message_from_solve_results(results)

        return message, results


if __name__ == '__main__':
    mod = EMaligner(schema_type=EMA_Schema)
    mod.run()
