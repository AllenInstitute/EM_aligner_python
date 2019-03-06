import numpy as np
import renderapi
import argschema
from .schemas import EMA_Schema
import utils
from .transform.transform import AlignerTransform
import time
import scipy.sparse as sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import factorized
import warnings
import os
import sys
import multiprocessing
import logging
import json
warnings.simplefilter(action='ignore', category=FutureWarning)
import h5py
warnings.resetwarnings()

logger = logging.getLogger(__name__)


def calculate_processing_chunk(fargs):
    # set up for calling using multiprocessing pool
    [pair, zloc, args, tile_ids] = fargs

    dbconnection = utils.make_dbconnection(args['pointmatch'])
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
    matches = utils.get_matches(
            pair['section1'],
            pair['section2'],
            args['pointmatch'],
            dbconnection)

    if len(matches) == 0:
        return chunk

    # extract IDs for fast checking
    pids = np.array([m['pId'] for m in matches])
    qids = np.array([m['qId'] for m in matches])

    # remove matches that don't have both IDs in tile_ids
    instack = np.in1d(pids, tile_ids) & np.in1d(qids, tile_ids)
    matches = np.array(matches)[instack].tolist()
    pids = pids[instack]
    qids = qids[instack]

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
                instack.size,
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
    if transform.fullsize:
        b = np.zeros((ni, 1)).astype('float64')
    else:
        b = np.zeros((ni, 2)).astype('float64')

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
        d, ind, iptr, wts, ib, npts = transform.CSR_from_tilepair(
            matches[k],
            pinds[k],
            qinds[k],
            args['matrix_assembly']['npts_min'],
            args['matrix_assembly']['npts_max'],
            args['matrix_assembly']['choose_random'])

        if d is None:
            continue  # if npts<nmin, or all weights=0

        # add both tile ids to the list
        chunk['tiles_used'].append(matches[k]['pId'])
        chunk['tiles_used'].append(matches[k]['qId'])

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
        for k in range(ib.shape[1]):
            b[global_rowind, k] = ib[:, k]
        indptr[global_rowind + 1] = iptr + indptr[nrows]

        nrows += wts.size


    del matches
    # truncate, because we allocated conservatively
    data = data[0: nrows * transform.nnz_per_row]
    indices = indices[0: nrows * transform.nnz_per_row]
    indptr = indptr[0: nrows + 1]
    weights = weights[0: nrows]
    b = b[0: nrows, :]

    chunk['data'] = np.copy(data)
    chunk['weights'] = np.copy(weights)
    chunk['indices'] = np.copy(indices)
    chunk['indptr'] = np.copy(indptr)
    chunk['b'] = np.copy(b)
    chunk['zlist'].append(pair['z1'])
    chunk['zlist'].append(pair['z2'])
    chunk['zlist'] = np.array(chunk['zlist'])
    del data, indices, indptr, weights, b

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


def mat_stats(m, name):
    shape = m.get_shape()
    mesg = "\n matrix: %s\n" % name
    mesg += " format: %s\n" % m.getformat()
    mesg += " shape: (%d, %d)\n" % (shape[0], shape[1])
    mesg += " nnz: %d" % m.nnz
    logger.debug(mesg)


class EMaligner(argschema.ArgSchemaParser):
    default_schema = EMA_Schema

    def run(self):
        logger.setLevel(self.args['log_level'])
        t0 = time.time()
        zvals = np.arange(
            self.args['first_section'],
            self.args['last_section'] + 1)

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

        self.transform = AlignerTransform(
            name=self.args['transformation'],
            order=self.args['poly_order'],
            fullsize=self.args['fullsize_transform'])

        if self.args['ingest_from_file'] != '':
            assemble_result = self.assemble_from_hdf5(
                self.args['ingest_from_file'],
                zvals,
                read_data=False)
            x = assemble_result['tforms']
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

            print(assemble_result['weights'].shape)
            print(assemble_result['A'].shape)
            print(assemble_result['b'].shape)

            # solve
            message, x, results = \
                self.solve_or_not(
                    assemble_result['A'],
                    assemble_result['weights'],
                    assemble_result['reg'],
                    assemble_result['x'],
                    assemble_result['b'])
            logger.info('\n' + message)
            if assemble_result['A'] is not None:
                results['Ashape'] = assemble_result['A'].shape
            del assemble_result['A']

        if self.args['output_mode'] == 'stack':
            write_to_new_stack(
                self.args['input_stack'],
                self.args['output_stack'],
                self.args['transformation'],
                self.args['fullsize_transform'],
                self.args['poly_order'],
                assemble_result['tspecs'],
                assemble_result['shared_tforms'],
                x,
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
        'b': None,
        'weights': None,
        'reg': None,
        'tspecs': None,
        'tforms': None,
        'tids': None,
        'shared_tforms': None,
        'unused_tids': None}

    def assemble_from_hdf5(self, filename, zvals, read_data=True):
        assemble_result = dict(self.assemble_struct)

        resolved = get_resolved_tilespecs(
            self.args['input_stack'],
            self.args['transformation'],
            zvals,
            fullsize=self.args['fullsize_transform'],
            order=self.args['poly_order'])

        assemble_result['shared_tforms'] = resolved.transforms

        with h5py.File(filename, 'r') as f:
            assemble_result['tids'] = np.array(
                f.get('used_tile_ids')[()]).astype('U')
            assemble_result['unused_tids'] = np.array(
                f.get('unused_tile_ids')[()]).astype('U')
            k=0
            assemble_result['tforms'] = []
            while True:
                name = 'transforms_%d' % k
                if name in f.keys():
                    assemble_result['tforms'].append(f.get(name)[()])
                    k += 1
                else:
                    break

            if len(assemble_result['tforms']) == 1:
                n = assemble_result['tforms'][0].size
                assemble_result['tforms'] = np.array(
                    assemble_result['tforms']).flatten().reshape((n, 1))
            else:
                assemble_result['tforms'] = np.transpose(
                    np.array(assemble_result['tforms']))

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
            ib = [np.array([]), np.array([])]
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
                    
                    k = 0
                    while True:
                        name = 'b_%d' % k
                        if name in f.keys():
                            ib[k] = np.append(ib[k], f.get(name)[()])
                            k += 1
                        else:
                            break

                    logger.info('  %s read' % fname)

            assemble_result['A'] = csr_matrix((data, indices, indptr))
            assemble_result['b'] = ib[0].reshape(-1, 1)
            if ib[1].size > 0:
                assemble_result['b'] = np.hstack((
                    assemble_result['b'],
                    ib[1].reshape(-1, 1)))

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

        assemble_result['resolved'] = utils.get_resolved_tilespecs(
            self.args['input_stack'],
            self.args['transformation'],
            self.args['n_parallel_jobs'],
            zvals,
            fullsize=self.args['fullsize_transform'],
            order=self.args['poly_order'])

        # create A matrix in compressed sparse row (CSR) format
        CSR_A = self.create_CSR_A(assemble_result['resolved'])
        assemble_result['A'] = CSR_A.pop('A')
        assemble_result['b'] = CSR_A.pop('b')
        assemble_result['x'] = CSR_A.pop('x')
        assemble_result['weights'] = CSR_A.pop('weights')
        assemble_result['tiles_used'] = CSR_A.pop('tiles_used')

        # some book-keeping if there were some unused tiles
        tids = np.array([t.tileId for t in assemble_result['resolved'].tilespecs])
        tile_ind = np.in1d(tids, assemble_result['tiles_used'])
        #assemble_result['tspecs'] = np.array(assemble_result['resolved'].tilespecs)[tile_ind]
        assemble_result['tids'] = tids[tile_ind]
        assemble_result['unused_tids'] = tids[np.invert(tile_ind)]

        #assemble_result['tforms'] = from_stack['tforms'][slice_ind, :]
        #del from_stack, CSR_A['tiles_used'], tile_ind

        # create the regularization vectors
        assemble_result['reg'] = self.transform.create_regularization(
            assemble_result['A'].shape[1],
            self.args['regularization'])

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


    def concatenate_results(self, results):
        result = {}
        result['data'] = np.concatenate([
            results[i]['data'] for i in range(len(results))
            if results[i]['data'] is not None]).astype('float64')
        result['b'] = np.concatenate([
            results[i]['b'] for i in range(len(results))
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


    def write_chunks_to_files(self, results):
        # split up into chunks
        npairs = len(results)
        chunks = [np.arange(npairs)]
        if self.args['hdf5_options']['chunks_per_file'] != -1:
            chunks = np.array_split(
                np.arange(npairs),
                np.ceil(
                    float(npairs) /
                    self.args['hdf5_options']['chunks_per_file']))

        metadata = []
        for pchunk in chunks:
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
                metadata.append(
                    write_chunk_to_file(
                        fname,
                        c,
                        cat_chunk['b'],
                        cat_chunk['weights']))
                    
        return metadata


    def create_CSR_A(self, resolved):
        func_result = {
            'A': None,
            'x': None,
            'b': None,
            'weights': None,
            'tiles_used': None,
            'metadata': None}

        pairs = utils.determine_zvalue_pairs(
                resolved.tilespecs,
                self.args['matrix_assembly']['depth'])
        npairs = len(pairs)

        tile_ids = np.array([t.tileId for t in resolved.tilespecs])

        fargs = [[pairs[i], i, self.args, tile_ids] for i in range(npairs)]

        with renderapi.client.WithPool(self.args['n_parallel_jobs']) as pool:
            results = np.array(pool.map(calculate_processing_chunk, fargs))

        func_result['tiles_used'] = np.unique(np.array(
                [item for result in results for item in result['tiles_used']]))

        if self.args['output_mode'] == 'hdf5':
            func_result['metadata'] = self.write_chunks_to_files(results)

        else:
            func_result['x'] = np.concatenate([
                t.tforms[-1].to_solve_vec() for t in resolved.tilespecs
                if t.tileId in func_result['tiles_used']])
            result = self.concatenate_results(results)
            A = csr_matrix((
                result['data'],
                result['indices'],
                result['indptr']))
            outw = sparse.eye(result['weights'].size, format='csr')
            outw.data = result['weights']
            tile_ind = np.in1d(tile_ids, func_result['tiles_used'])
            slice_ind = np.repeat(
                tile_ind,
                self.transform.DOF_per_tile / self.transform.rows_per_ptmatch)
            func_result['A'] = A[:, slice_ind]
            func_result['b'] = result['b']
            func_result['weights'] = outw

        return func_result

    def solve_or_not(self, A, weights, reg, filt_tforms, b):
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
            rhs = np.zeros((A.shape[1], b.shape[1]))
            print(rhs.shape)
            print(reg.shape)
            for i in range(b.shape[1]):
                rhs[:, i] = A.transpose().dot(weights).dot(b[:, i])
            weights.data = np.sqrt(weights.data)
            rtWA = weights.dot(A)
            K = rtWA.transpose().dot(rtWA) + reg

            logger.info(' K created in %0.1f seconds' % (time.time() - t0))
            t0 = time.time()
            del weights, rtWA

            # factorize, then solve, efficient for large affine
            solve = factorized(K)
            if filt_tforms.shape[1] == 2:
                # certain transforms have redundant matrices
                # then applies the LU decomposition to
                # the u and v transforms separately
                Lm = reg.dot(filt_tforms[:, 0]) + rhs[:, 0]
                xu = solve(Lm)
                erru = A.dot(xu) - b[:, 0]
                precisionu = \
                    np.linalg.norm(K.dot(xu) - Lm) / np.linalg.norm(Lm)

                Lm = reg.dot(filt_tforms[:, 1]) + rhs[:, 1]
                xv = solve(Lm)
                errv = A.dot(xv) - b[:, 1]
                precisionv = \
                    np.linalg.norm(K.dot(xv) - Lm) / np.linalg.norm(Lm)
                precision = np.sqrt(precisionu ** 2 + precisionv ** 2)

                # recombine
                x = np.transpose(np.vstack((xu, xv)))
                err = np.hstack((erru, errv))
                del xu, xv, erru, errv, precisionu, precisionv
            else:
                # simpler case for similarity, or
                # affine_fullsize, but 2x larger than affine

                Lm = reg.dot(filt_tforms[:, 0]) + rhs[:, 0]
                x = solve(Lm)
                err = A.dot(x) - b[:, 0]
                precision = \
                    np.linalg.norm(K.dot(x) - Lm) / np.linalg.norm(Lm)
            del K, Lm

            error = np.linalg.norm(err)

            results = {}
            results['time'] = time.time()-t0
            results['precision'] = precision
            results['error'] = error
            results['err'] = [np.abs(err).mean(), np.abs(err).std()]

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

            # get the scales (quick way to look for distortion)
            tforms = self.transform.from_solve_vec(x)
            if isinstance(
                    self.transform,
                    renderapi.transform.Polynomial2DTransform):
                # renderapi does not have scale property
                if self.transform.order > 0:
                    scales = np.array(
                        [[t.params[0, 1], t.params[1, 2]]
                         for t in tforms]).flatten()
                else:
                    scales = np.array([0])
            else:
                scales = np.array([
                    np.array(t.scale) for t in tforms]).flatten()

            results['scale'] = scales.mean()
            message += '\n avg scale = %0.2f +/- %0.2f' % (
                scales.mean(), scales.std())

        return message, x, results


if __name__ == '__main__':
    mod = EMaligner(schema_type=EMA_Schema)
    mod.run()
