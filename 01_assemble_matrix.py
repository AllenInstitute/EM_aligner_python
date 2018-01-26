#!/usr/bin/env python

from pymongo import MongoClient
import numpy as np
import renderapi
import argschema
import copy
import time
from scipy.sparse import csr_matrix
import h5py
import os

class db_params(argschema.ArgSchema):
    owner = argschema.fields.String(default='',description='owner') 
    project = argschema.fields.String(default='',description='project') 
    name = argschema.fields.String(default='',description='name')
    host = argschema.fields.String(default='em-131fs',description='render host')
    port = argschema.fields.Int(default=8080,description='render port')
    mongo_host = argschema.fields.String(default='em-131fs',description='mongodb host')
    mongo_port = argschema.fields.Int(default=27017,description='mongodb port')
    db_interface = argschema.fields.String(default='mongo')
    client_scripts = argschema.fields.String(default='/allen/programs/celltypes/workgroups/em-connectomics/gayathrim/nc-em2/Janelia_Pipeline/render_latest/render-ws-java-client/src/main/scripts',description='render bin path')

class output_options(argschema.ArgSchema):
    output_mode = argschema.fields.String(default='hdf5')
    output_dir = argschema.fields.String(default='/allen/programs/celltypes/workgroups/em-connectomics/danielk/solver_exchange/python/')
    chunks_per_file = argschema.fields.Int(default=5,description='how many sections with upward-looking cross section to write per .h5 file')

class matrix_assembly(argschema.ArgSchema):
    depth = argschema.fields.Int(default=2,description='depth in z for matrix assembly point matches')
    cross_pt_weight = argschema.fields.Float(default=1.0,description='weight of cross section point matches')
    montage_pt_weight = argschema.fields.Float(default=1.0,description='weight of montage point matches')

class regularization(argschema.ArgSchema):
    default_lambda = argschema.fields.Float(0.005,description='regularization factor')
    translation_lambda = argschema.fields.Float(0.005,description='regularization factor')

class pointmatch(db_params):
    collection_type = argschema.fields.String(default='pointmatch',description="'stack' or 'pointmatch'")
class stack(db_params):
    collection_type = argschema.fields.String(default='stack',description="'stack' or 'pointmatch'")

class MySchema(argschema.ArgSchema):
    first_section = argschema.fields.Int(default=1000, description = 'first section for matrix assembly')
    last_section = argschema.fields.Int(default=1000, description = 'last section for matrix assembly')
    input_stack = argschema.fields.Nested(stack)
    pointmatch = argschema.fields.Nested(pointmatch)
    output_options = argschema.fields.Nested(output_options)
    matrix_assembly = argschema.fields.Nested(matrix_assembly)
    regularization = argschema.fields.Nested(regularization)
    showtiming = argschema.fields.Int(default=1,description = 'have the routine showhow long each process takes')
    output_dir = argschema.fields.String(default='/allen/programs/celltypes/workgroups/em-connectomics/danielk/solver_exchange/python/',description='where to send logs and results')

def make_dbconnection(collection):
    #connect to the database
    if collection['db_interface']=='mongo':
        client = MongoClient(host=collection['mongo_host'],port=collection['mongo_port'])
        if collection['collection_type']=='stack':
            mongo_collection_name = collection['owner']+'__'+collection['project']+'__'+collection['name']+'__tile'
            dbconnection = client.render[mongo_collection_name]
        elif collection['collection_type']=='pointmatch':
            mongo_collection_name = collection['owner']+'__'+collection['name']
            dbconnection = client.match[mongo_collection_name]
    elif collection['db_interface']=='render':
        dbconnection = renderapi.connect(**collection)
    else:
        print 'invalid interface in make_dbconnection()'
        return
    return dbconnection

def get_tileids_and_tforms(stack,zvals):
    #connect to the database
    dbconnection = make_dbconnection(stack)

    #loop over z values
    tile_ids = []
    tile_tforms = []
    t0 = time.time()
    for z in zvals:
        #load tile specs from the database
        if stack['db_interface']=='render':
            tspecs = renderapi.tilespec.get_tile_specs_from_z(stack['name'],float(z),render=dbconnection,owner=stack['owner'],project=stack['project'])
        if stack['db_interface']=='mongo':
            cursor = dbconnection.find({'z':float(z)})
            tspecs = list(cursor)

        for ts in tspecs:
            if stack['db_interface']=='render':
                tile_ids.append(ts.tileId)
                tile_tforms.append([ts.tforms[1].M[0,0],ts.tforms[1].M[0,1],ts.tforms[1].M[0,2],ts.tforms[1].M[1,0],ts.tforms[1].M[1,1],ts.tforms[1].M[1,2]])
            if stack['db_interface']=='mongo':
                tile_ids.append(ts['tileId'])
                tile_tforms.append(np.array(ts['transforms']['specList'][1]['dataString'].split()).astype('float')[[0,2,4,1,3,5]])

    print 'loaded %d tile specs from %d zvalues in %0.1f sec using interface: %s'%(len(tile_ids),len(zvals),time.time()-t0,stack['db_interface'])
    return np.array(tile_ids),np.array(tile_tforms)

def write_csrtype_files(collection,matrix_assembly,tile_ids,zvals,output_options):
    #connect to the database
    dbconnection = make_dbconnection(collection)

    file_number=0
    nmax=50

    sorter = np.argsort(tile_ids)
    file_chunks = 0

    if output_options['chunks_per_file']==-1:
        nmod=0.1 # np.mod(n+1,0.1) never == 0
    else:
        nmod = output_options['chunks_per_file']

    for i in np.arange(len(zvals)):
    #for i in np.arange(1):
        print ' upward-looking for z %d'%zvals[i]
        jmax = np.min([i+matrix_assembly['depth']+1,len(zvals)])
        for j in np.arange(i,jmax): #depth, upward looking
            t0=time.time()
            if collection['db_interface']=='render':
                if i==j:
                    matches = renderapi.pointmatch.get_matches_within_group(collection['name'],str(float(zvals[i])),owner=collection['owner'],render=dbconnection)
                else:
                    matches = renderapi.pointmatch.get_matches_from_group_to_group(collection['name'],str(float(zvals[i])),str(float(zvals[j])),owner=collection['owner'],render=dbconnection)
                matches = np.array(matches)

            if collection['db_interface']=='mongo':
                cursor = dbconnection.find({'pGroupId':str(float(zvals[i])),'qGroupId':str(float(zvals[j]))})
                matches = np.array(list(cursor))
                if i!=j:
                    #in principle, this does nothing if zvals[i] < zvals[j], but, just in case
                    cursor = dbconnection.find({'pGroupId':str(float(zvals[j])),'qGroupId':str(float(zvals[i]))})
                    matches = np.append(matches,list(cursor))

            print '  loaded %d matches for z1=%d z2=%d in %0.1f sec using interface: %s'%(len(matches),zvals[i],zvals[j],time.time()-t0,collection['db_interface'])

            t0 = time.time()

            pids = []
            qids = []
            for m in matches:
                pids.append(m['pId'])
                qids.append(m['qId'])
            pids = np.array(pids)
            qids = np.array(qids)

            #for the given point matches, these are the indices in tile_ids that are being called
            pinds = sorter[np.searchsorted(tile_ids,pids,sorter=sorter)]
            qinds = sorter[np.searchsorted(tile_ids,qids,sorter=sorter)]

            #let's keep p<q
            backwards = qinds<pinds
            ptmp = pinds[backwards]
            qtmp = qinds[backwards]
            pinds[backwards] = qtmp
            qinds[backwards] = ptmp
            del qtmp,ptmp

            p0col = pinds*6
            q0col = qinds*6
            
            drow = np.zeros(6).astype('float64')
            dblock = np.zeros(6*nmax).astype('float64')

            #ordering for backwards
            dback=np.array([3,4,5,0,1,2])
            dforw=np.array([0,1,2,3,4,5])
            colx = np.arange(3)
            coly = colx+3

            #conservative pre-allocation
            nmatches = len(matches)
            data = np.zeros(6*nmax*nmatches*2).astype('float64')
            weights = np.zeros(6*nmax*nmatches*2).astype('float16')
            indices = np.zeros(6*nmax*nmatches*2).astype('int64')
            indptr = np.zeros(2*nmax*nmatches+1).astype('int64')
            halfp_ones = np.ones(2*6*nmax).astype('float16')

            indptr[0] = 0
            nrows = 0

            for k in np.arange(len(matches)):
            #for k in np.arange(5400):
                npts = len(matches[k]['matches']['q'][0])
                if npts > nmax: #really dumb filter for limiting size
                    npts=nmax

                #what column indices will we be writing to?
                xindices = np.hstack((colx+p0col[k],colx+q0col[k])).astype('int64')
                yindices = xindices+3

                if backwards[k]:
                    dorder = dback
                else:
                    dorder = dforw

                if i==j:
                    matchweight = matrix_assembly['montage_pt_weight']
                else:
                    matchweight = matrix_assembly['cross_pt_weight']

                m = np.arange(npts)
                m6 = m*6
                dblock[dorder[0]+m6] = np.array(matches[k]['matches']['p'][0])[m]
                dblock[dorder[1]+m6] = np.array(matches[k]['matches']['p'][1])[m]
                dblock[dorder[2]+m6] = 1.0
                dblock[dorder[3]+m6] = -1.0*np.array(matches[k]['matches']['q'][0])[m]
                dblock[dorder[4]+m6] = -1.0*np.array(matches[k]['matches']['q'][1])[m]
                dblock[dorder[5]+m6] = -1.0
                data[(nrows*6):(nrows*6+6*npts*2)] = np.tile(dblock[0:npts*6],2)  #xrows, then duplicate for y rows
                weights[(nrows*6):(nrows*6+6*npts*2)] = matchweight*halfp_ones[0:2*npts*6]  #xrows, then duplicate for y rows
                indices[(nrows*6):(nrows*6+6*npts)] = np.tile(xindices,npts) #xrow column indices
                indices[(nrows*6+6*npts):(nrows*6+6*npts*2)] = np.tile(yindices,npts) #yrow column indices
                indptr[(nrows+1):(nrows+1+2*npts)] = np.arange(1,2*npts+1)*6+indptr[nrows]

                nrows += 2*npts
            
            #truncate, because we allocated conservatively
            data = data[0:nrows*6]
            weights = weights[0:nrows*6]
            indices = indices[0:nrows*6]
            indptr = indptr[0:nrows+1]

            c = csr_matrix((data,indices,indptr))
            print '  created submatrix in %0.1f sec.'%(time.time()-t0),'canonical format: ',c.has_canonical_format,', shape: ',c.shape,' nnz: ',c.nnz
            del c
        
            if file_chunks==0:
                file_data = copy.deepcopy(data)
                file_weights = copy.deepcopy(weights)
                file_indices = copy.deepcopy(indices)
                file_indptr = copy.deepcopy(indptr)
            else:
                file_data = np.append(file_data,data)
                file_weights = np.append(file_weights,weights)
                file_indices = np.append(file_indices,indices)
                lastptr = file_indptr[-1]
                file_indptr = np.append(file_indptr,indptr[1:]+lastptr)
            file_chunks += 1

        if (np.mod(i+1,nmod)==0)|(i==len(zvals)-1):
            fname = output_options['output_dir']+'/%d.h5'%file_number
            c = csr_matrix((file_data,file_indices,file_indptr))
            print ' from %d chunks, canonical format: '%output_options['chunks_per_file'],c.has_canonical_format,', shape: ',c.shape,' nnz: ',c.nnz
            print ' writing to file: %s'%fname
            f = h5py.File(fname,"w")
            dset = f.create_dataset("indptr",(c.indptr.size,),dtype='int32')
            dset[:] = c.indptr
            dset = f.create_dataset("indices",(c.indices.size,),dtype='int32')
            dset[:] = c.indices
            dset = f.create_dataset("data",(c.data.size,),dtype='float64')
            dset[:] = c.data
            dset = f.create_dataset("weights",(file_weights.size,),dtype='float16')
            dset[:] = file_weights
            f.close()
            print 'wrote %s\n%0.2fGB on disk'%(fname,os.path.getsize(fname)/(2.**30))
            del c
            file_chunks = 0
            file_number += 1

    return 1

if __name__=='__main__':
    t0 = time.time()
    mod = argschema.ArgSchemaParser(schema_type=MySchema)

    #specify the z values
    zvals = np.arange(mod.args['first_section'],mod.args['last_section']+1)

    #get the tile IDs and transforms
    tile_ids,tile_tforms = get_tileids_and_tforms(mod.args['input_stack'],zvals)

    #create compressed sparse row format files
    write_csrtype_files(mod.args['pointmatch'],mod.args['matrix_assembly'],tile_ids,zvals,mod.args['output_options'])

    print 'total time: %0.1f'%(time.time()-t0)

