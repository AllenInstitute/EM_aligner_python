#!/usr/bin/env python

from pymongo import MongoClient
import numpy as np
import renderapi
from assembly_schema import *
import copy
import time
import scipy.sparse as sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import h5py
import os

def make_dbconnection(collection,which='tile'):
    #connect to the database
    if collection['db_interface']=='mongo':
        client = MongoClient(host=collection['mongo_host'],port=collection['mongo_port'])
        if collection['collection_type']=='stack':
            #for getting shared transforms, which='transform'
            mongo_collection_name = collection['owner']+'__'+collection['project']+'__'+collection['name']+'__'+which
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

    tile_ids = []
    tile_tforms = []
    t0 = time.time()

    for z in zvals:
        #load tile specs from the database
        if stack['db_interface']=='render':
            tmp = renderapi.resolvedtiles.get_resolved_tiles_from_z(stack['name'],float(z),render=dbconnection,owner=stack['owner'],project=stack['project'])
            tspecs = tmp.tilespecs
            shared_tforms = tmp.transforms
        if stack['db_interface']=='mongo':
            #cursor = dbconnection.find({'z':float(z)}) #no order
            cursor = dbconnection.find({'z':float(z)}).sort([('layout.imageRow',1),('layout.imageCol',1)])
            tspecs = list(cursor)
            dbconnection2 = make_dbconnection(stack,which='transform')
            cursor = dbconnection2.find({})
            shared_tforms = list(cursor)
            #will import these to new stack using renderapi
            for j in np.arange(len(shared_tforms)):
                shared_tforms[j] = renderapi.transform.load_transform_json(shared_tforms[j])

        #make lists of IDs and transforms
        for k in np.arange(len(tspecs)):
            if stack['db_interface']=='render':
                tile_ids.append(tspecs[k].tileId)
                tile_tforms.append([tspecs[k].tforms[-1].M[0,0],tspecs[k].tforms[-1].M[0,1],tspecs[k].tforms[-1].M[0,2],tspecs[k].tforms[-1].M[1,0],tspecs[k].tforms[-1].M[1,1],tspecs[k].tforms[-1].M[1,2]])
            if stack['db_interface']=='mongo':
                tile_ids.append(tspecs[k]['tileId'])
                tile_tforms.append(np.array(tspecs[k]['transforms']['specList'][-1]['dataString'].split()).astype('float')[[0,2,4,1,3,5]])
                tspecs[k] = renderapi.tilespec.TileSpec(json=tspecs[k]) #move to renderapi object

    print 'loaded %d tile specs from %d zvalues in %0.1f sec using interface: %s'%(len(tile_ids),len(zvals),time.time()-t0,stack['db_interface'])
    return np.array(tile_ids),np.array(tile_tforms).flatten(),np.array(tspecs),shared_tforms

def create_CSR_A(collection,matrix_assembly,tile_ids,zvals,output_options):
    #connect to the database
    dbconnection = make_dbconnection(collection)

    file_number=0
    nmax=500

    sorter = np.argsort(tile_ids)
    file_chunks = 0
    tiles_used = []

    if output_options['chunks_per_file']==-1:
        nmod=0.1 # np.mod(n+1,0.1) never == 0
    else:
        nmod = output_options['chunks_per_file']

    file_zlist=[]

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
                    #cursor = dbconnection.find({'pGroupId':str(float(zvals[j])),'qGroupId':str(float(zvals[i]))})
                    cursor = dbconnection.find({'pGroupId':str(float(zvals[j])),'qGroupId':str(float(zvals[i]))}).sort({"item.layout.imgaeRow":1})
                    matches = np.append(matches,list(cursor))

            if len(matches)==0:
                print 'WARNING: %d matches for z1=%d z2=%d in pointmatch collection'%(len(matches),zvals[i],zvals[j])
                continue

            #extract IDs for checking
            pids = []
            qids = []
            for m in matches:
                pids.append(m['pId'])
                qids.append(m['qId'])
            pids = np.array(pids)
            qids = np.array(qids)

            #remove matches that don't have both IDs in tile_ids
            instack = np.in1d(pids,tile_ids)&np.in1d(qids,tile_ids)
            matches = matches[instack]
            pids = pids[instack]
            qids = qids[instack]
 
            if len(matches)==0:
                print 'WARNING: no tile pairs in stack for pointmatches in z1=%d z2=%d'%(zvals[i],zvals[j])
                continue

            print '  loaded %d matches, using %d, for z1=%d z2=%d in %0.1f sec using interface: %s'%(instack.size,len(matches),zvals[i],zvals[j],time.time()-t0,collection['db_interface'])
        
            t0 = time.time()
            
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
            indices = np.zeros(6*nmax*nmatches*2).astype('int64')
            indptr = np.zeros(2*nmax*nmatches+1).astype('int64')
            weights = np.zeros(2*nmax*nmatches).astype('float64')
            halfp_ones = np.ones(2*nmax).astype('float64')

            indptr[0] = 0
            nrows = 0

            for k in np.arange(nmatches):
                npts = len(matches[k]['matches']['q'][0])
                if npts > nmax: #really dumb filter for limiting size
                    npts=nmax
                if npts==0:
                    print 'zero length point match entry?'
                    continue

                #add both tile ids to the list
                tiles_used.append(matches[k]['pId'])
                tiles_used.append(matches[k]['qId'])

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
                indices[(nrows*6):(nrows*6+6*npts)] = np.tile(xindices,npts) #xrow column indices
                indices[(nrows*6+6*npts):(nrows*6+6*npts*2)] = np.tile(yindices,npts) #yrow column indices
                indptr[(nrows+1):(nrows+1+2*npts)] = np.arange(1,2*npts+1)*6+indptr[nrows]
                weights[(nrows):(nrows+2*npts)] = matchweight*halfp_ones[0:2*npts]  #xrows, then duplicate for y rows

                nrows += 2*npts
           
            del matches 
            #truncate, because we allocated conservatively
            data = data[0:nrows*6]
            indices = indices[0:nrows*6]
            indptr = indptr[0:nrows+1]
            weights = weights[0:nrows]

            #c = csr_matrix((data,indices,indptr))
            #print '  created submatrix in %0.1f sec.'%(time.time()-t0),'canonical format: ',c.has_canonical_format,', shape: ',c.shape,' nnz: ',c.nnz
            #del c
        
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
            file_zlist.append(zvals[i])
            del data,indices,indptr,weights

        if (np.mod(i+1,nmod)==0)|(i==len(zvals)-1):
            fname = '%d_%d.h5'%(file_zlist[0],file_zlist[-1])
            fullname = output_options['output_dir']+'/'+fname
            c = csr_matrix((file_data,file_indices,file_indptr))
            del file_data,file_indices,file_indptr
            print ' from %d chunks, canonical format: '%output_options['chunks_per_file'],c.has_canonical_format,', shape: ',c.shape,' nnz: ',c.nnz
            cnnz = np.argwhere(c.getnnz(0)==0).flatten().size
            print ' matrix contains %d all-zero columns (%d tiles in stack were not matched)'%(cnnz,cnnz/6)
            if output_options['output_mode']=='hdf5':
                print ' writing to file: %s'%fullname
                f = h5py.File(fullname,"w")
                dset = f.create_dataset("indptr",(c.indptr.size,),dtype='int32')
                dset[:] = c.indptr
                dset = f.create_dataset("indices",(c.indices.size,),dtype='int32')
                dset[:] = c.indices
                dset = f.create_dataset("data",(c.data.size,),dtype='float64')
                dset[:] = c.data
                dset = f.create_dataset("weights",(file_weights.size,),dtype='float64')
                dset[:] = file_weights
                f.close()
                print 'wrote %s\n%0.2fGB on disk'%(fname,os.path.getsize(fullname)/(2.**30))
                fmode='a'
                if file_number==0:
                    fmode='w'
                f=open(output_options['output_dir']+'/index.txt',fmode)
                f.write('file %s contains %ld rows\n'%(fname,c.shape[0]))
                f.close()
            file_chunks = 0
            file_number += 1
            file_zlist = []
 
    outw = sparse.eye(file_weights.size,format='csr')
    outw.data = file_weights
    del file_weights 
    return c,outw,np.unique(np.array(tiles_used))

def create_regularization(regularization,tile_tforms,output_options):
    #create a regularization vector
    reg = np.ones_like(tile_tforms).astype('float64')*regularization['default_lambda']
    reg[2::3] = regularization['translation_lambda']

    if output_options['output_mode']=='hdf5':
        #write the input transforms to disk
        fname = output_options['output_dir']+'/regularization.h5'
        f = h5py.File(fname,"w")
        dset = f.create_dataset("transforms",(tile_tforms.size,),dtype='float64')
        dset[:] = tile_tforms

        #create a regularization vector
        dset = f.create_dataset("lambda",(reg.size,),dtype='float64')
        dset[:] = reg
        f.close()
        print 'wrote %s'%fname

    outr = sparse.eye(reg.size,format='csr')
    outr.data = reg

    return outr

if __name__=='__main__':
    t0 = time.time()
    mod = argschema.ArgSchemaParser(schema_type=MySchema)

    #specify the z values
    zvals = np.arange(mod.args['first_section'],mod.args['last_section']+1)

    if mod.args['solve_type']=='montage':
        output = mod.args['output_stack'] 
        ingestconn = make_dbconnection(output)
        renderapi.stack.create_stack(output['name'],render=ingestconn)

        for z in zvals:
            #get the tile IDs and transforms
            tile_ids,tile_tforms,tile_tspecs,shared_tforms = get_tileids_and_tforms(mod.args['input_stack'],[z])

            #create A matrix in compressed sparse row (CSR) format
            A,weights,tiles_used = create_CSR_A(mod.args['pointmatch'],mod.args['matrix_assembly'],tile_ids,[z],mod.args['output_options'])
            tile_ind = np.in1d(tile_ids,tiles_used)
            mont_tspecs = tile_tspecs[tile_ind]
            mont_tforms = tile_tforms[np.repeat(tile_ind,6)]
           
            del tile_ids,tiles_used,tile_tforms,tile_ind,tile_tspecs
            
            #create the regularization vectors
            mont_reg = create_regularization(mod.args['regularization'],mont_tforms,mod.args['output_options'])

            #regularized least squares
            ATW = A.transpose().dot(weights)
            K = ATW.dot(A) + mont_reg
            print ' created K: %d x %d'%K.shape
            Lm = mont_reg.dot(mont_tforms)

            del weights,mont_tforms,mont_reg
     
            #solve
            mont_x = spsolve(K,Lm)
            print ' assembled and solved in %0.1f sec'%(time.time()-t0)
            print ' precision [norm(Kx-Lm)/norm(Lm)] = %0.1e'%(np.linalg.norm(K.dot(mont_x)-Lm)/np.linalg.norm(Lm))
            print ' error     [norm(Ax-b)] = %0.3f'%(np.linalg.norm(A.dot(mont_x)))
            del A,K,ATW,Lm

            #replace the last transform in the tilespec with the new one
            for m in np.arange(len(mont_tspecs)):
                mont_tspecs[m].tforms[-1].M[0,0] = mont_x[m*6+0]
                mont_tspecs[m].tforms[-1].M[0,1] = mont_x[m*6+1]
                mont_tspecs[m].tforms[-1].M[0,2] = mont_x[m*6+2]
                mont_tspecs[m].tforms[-1].M[1,0] = mont_x[m*6+3]
                mont_tspecs[m].tforms[-1].M[1,1] = mont_x[m*6+4]
                mont_tspecs[m].tforms[-1].M[1,2] = mont_x[m*6+5]

            #renderapi.client.import_tilespecs(output['name'],tspout,sharedTransforms=shared_tforms,render=ingestconn)
            renderapi.client.import_tilespecs_parallel(output['name'],mont_tspecs.tolist(),sharedTransforms=shared_tforms,render=ingestconn,close_stack=False)
            del shared_tforms,mont_x,mont_tspecs

        if mod.args['close_stack']:
            renderapi.stack.set_stack_state(output['name'],state='COMPLETE',render=ingestconn)

    else:
        #get the tile IDs and transforms
        tile_ids,tile_tforms,tile_tspecs = get_tileids_and_tforms(mod.args['input_stack'],zvals)

        #create A matrix in compressed sparse row (CSR) format
        c,weights,tiles_used = create_CSR_A(mod.args['pointmatch'],mod.args['matrix_assembly'],tile_ids,zvals,mod.args['output_options'])

        #create the regularization vectors
        create_regularization(mod.args['regularization'],tile_tforms,mod.args['output_options'])

    print 'total time: %0.1f'%(time.time()-t0)

