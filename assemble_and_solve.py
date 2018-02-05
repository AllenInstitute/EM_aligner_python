#!/usr/bin/env python

from pymongo import MongoClient
import numpy as np
import renderapi
from EM_aligner_python_schema import *
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

def get_tileids_and_tforms(stack,tform_obj,zvals):
    #connect to the database
    dbconnection = make_dbconnection(stack)

    tile_ids = []
    tile_tforms = []
    tile_tspecs = []
    shared_tforms = []
    t0 = time.time()

    for z in zvals:
        #load tile specs from the database
        if stack['db_interface']=='render':
            tmp = renderapi.resolvedtiles.get_resolved_tiles_from_z(stack['name'],float(z),render=dbconnection,owner=stack['owner'],project=stack['project'])
            tspecs = tmp.tilespecs
            for st in tmp.transforms:
                shared_tforms.append(st)
        if stack['db_interface']=='mongo':
            #cursor = dbconnection.find({'z':float(z)}) #no order
            cursor = dbconnection.find({'z':float(z)}).sort([('layout.imageRow',1),('layout.imageCol',1)]) #like renderapi order?
            tspecs = list(cursor)
            refids = []
            for ts in tspecs:
                for m in np.arange(len(ts['transforms']['specList'])):
                    if 'refId' in ts['transforms']['specList'][m]:
                        refids.append(ts['transforms']['specList'][m]['refId'])
            refids = np.unique(np.array(refids))

            #be selective of which transforms to pass on to the new stack
            dbconnection2 = make_dbconnection(stack,which='transform')
            for refid in refids:
                shared_tforms.append(renderapi.transform.load_transform_json(list(dbconnection2.find({"id":refid}))[0]))

        #make lists of IDs and transforms
        for k in np.arange(len(tspecs)):
            if stack['db_interface']=='render':
                tile_ids.append(tspecs[k].tileId)
                if tform_obj.name=='affine':
                    tile_tforms.append([tspecs[k].tforms[-1].M[0,0],tspecs[k].tforms[-1].M[0,1],tspecs[k].tforms[-1].M[0,2],tspecs[k].tforms[-1].M[1,0],tspecs[k].tforms[-1].M[1,1],tspecs[k].tforms[-1].M[1,2]])
                elif tform_obj.name=='rigid':
                    tile_tforms.append([tspecs[k].tforms[-1].M[0,0],tspecs[k].tforms[-1].M[0,1],tspecs[k].tforms[-1].M[0,2],tforms[-1].M[1,2]])
            if stack['db_interface']=='mongo':
                tile_ids.append(tspecs[k]['tileId'])
                if tform_obj.name=='affine':
                    tile_tforms.append(np.array(tspecs[k]['transforms']['specList'][-1]['dataString'].split()).astype('float')[[0,2,4,1,3,5]])
                elif tform_obj.name=='rigid':
                    tile_tforms.append(np.array(tspecs[k]['transforms']['specList'][-1]['dataString'].split()).astype('float')[[0,2,4,5]])
                tspecs[k] = renderapi.tilespec.TileSpec(json=tspecs[k]) #move to renderapi object
            tile_tspecs.append(tspecs[k])

    print 'loaded %d tile specs from %d zvalues in %0.1f sec using interface: %s'%(len(tile_ids),len(zvals),time.time()-t0,stack['db_interface'])
    return np.array(tile_ids),np.array(tile_tforms).flatten(),np.array(tile_tspecs).flatten(),shared_tforms

class transform_csr:
    #class to convert a tile pair pointmatch dict into a CSR
    def __init__(self,name,nmin,nmax):
        self.nmin = nmin
        self.nmax = nmax
        self.name=name
        if self.name=='affine':
            self.DOF_per_tile=6
            self.nnz_per_row=6
            self.rows_per_ptmatch=2
        if self.name=='rigid':
            self.DOF_per_tile=4
            self.nnz_per_row=6
            self.rows_per_ptmatch=4

        #allocate some space
        self.data = np.zeros(nmax*self.nnz_per_row*self.rows_per_ptmatch).astype('float64')
        self.indices = np.zeros(nmax*self.nnz_per_row*self.rows_per_ptmatch).astype('int64')
        self.indptr = np.zeros(nmax*self.rows_per_ptmatch).astype('int64')
        self.weights = np.zeros(nmax*self.rows_per_ptmatch).astype('float64')

    def CSR_tile_pair(self,match,tile_ind1,tile_ind2):
       npts = len(match['matches']['q'][0])
       if npts > self.nmax: #really dumb filter for limiting size
           npts=self.nmax
       if npts<self.nmin:
           return None,None,None,None
       self.npts=npts

       m = np.arange(npts)
       mstep = m*self.nnz_per_row
       if self.name=='affine':
           #u=ax+by+c
           self.data[0+mstep] = np.array(match['matches']['p'][0])[m]
           self.data[1+mstep] = np.array(match['matches']['p'][1])[m]
           self.data[2+mstep] = 1.0
           self.data[3+mstep] = -1.0*np.array(match['matches']['q'][0])[m]
           self.data[4+mstep] = -1.0*np.array(match['matches']['q'][1])[m]
           self.data[5+mstep] = -1.0
           uindices = np.hstack((tile_ind1*self.DOF_per_tile+np.array([0,1,2]),tile_ind2*self.DOF_per_tile+np.array([0,1,2])))
           self.indices[0:npts*self.nnz_per_row] = np.tile(uindices,npts) 
           #v=dx+ey+f
           self.data[(npts*self.nnz_per_row):(2*npts*self.nnz_per_row)] = self.data[0:npts*self.nnz_per_row]
           self.indices[npts*self.nnz_per_row:2*npts*self.nnz_per_row] = np.tile(uindices+3,npts) 
           self.indptr[0:2*npts] = np.arange(1,2*npts+1)*self.nnz_per_row
           self.weights[0:2*npts] = np.tile(np.array(match['matches']['w'])[m],2)

       if self.name=='rigid':
           px = np.array(match['matches']['p'][0])[m]
           py = np.array(match['matches']['p'][1])[m]
           qx = np.array(match['matches']['q'][0])[m]
           qy = np.array(match['matches']['q'][1])[m]
           #u=ax+by+c
           self.data[0+mstep] = px
           self.data[1+mstep] = py
           self.data[2+mstep] = 1.0
           self.data[3+mstep] = -1.0*qx
           self.data[4+mstep] = -1.0*qy
           self.data[5+mstep] = -1.0
           uindices = np.hstack((tile_ind1*self.DOF_per_tile+np.array([0,1,2]),tile_ind2*self.DOF_per_tile+np.array([0,1,2])))
           self.indices[0:npts*self.nnz_per_row] = np.tile(uindices,npts) 
           #v=-bx+ay+d
           self.data[0+mstep+npts*self.nnz_per_row] = -1.0*px
           self.data[1+mstep+npts*self.nnz_per_row] = py
           self.data[2+mstep+npts*self.nnz_per_row] = 1.0
           self.data[3+mstep+npts*self.nnz_per_row] = 1.0*qx
           self.data[4+mstep+npts*self.nnz_per_row] = -1.0*qy
           self.data[5+mstep+npts*self.nnz_per_row] = -1.0
           vindices = np.hstack((tile_ind1*self.DOF_per_tile+np.array([1,0,3]),tile_ind2*self.DOF_per_tile+np.array([1,0,3])))
           self.indices[npts*self.nnz_per_row:2*npts*self.nnz_per_row] = np.tile(vindices,npts) 
           #du
           self.data[0+mstep+2*npts*self.nnz_per_row] = px-px.mean()
           self.data[1+mstep+2*npts*self.nnz_per_row] = py-py.mean()
           self.data[2+mstep+2*npts*self.nnz_per_row] = 0.0
           self.data[3+mstep+2*npts*self.nnz_per_row] = -1.0*(qx-qx.mean())
           self.data[4+mstep+2*npts*self.nnz_per_row] = -1.0*(qy-qy.mean())
           self.data[5+mstep+2*npts*self.nnz_per_row] = -0.0
           self.indices[2*npts*self.nnz_per_row:3*npts*self.nnz_per_row] = np.tile(uindices,npts) 
           #dv
           self.data[0+mstep+3*npts*self.nnz_per_row] = -1.0*(px-px.mean())
           self.data[1+mstep+3*npts*self.nnz_per_row] = py-py.mean()
           self.data[2+mstep+3*npts*self.nnz_per_row] = 0.0
           self.data[3+mstep+3*npts*self.nnz_per_row] = 1.0*(qx-qx.mean())
           self.data[4+mstep+3*npts*self.nnz_per_row] = -1.0*(qy-qy.mean())
           self.data[5+mstep+3*npts*self.nnz_per_row] = -0.0
           self.indices[3*npts*self.nnz_per_row:4*npts*self.nnz_per_row] = np.tile(uindices,npts) 
  
           self.indptr[0:self.rows_per_ptmatch*npts] = np.arange(1,self.rows_per_ptmatch*npts+1)*self.nnz_per_row
           self.weights[0:self.rows_per_ptmatch*npts] = np.tile(np.array(match['matches']['w'])[m],self.rows_per_ptmatch)

       return self.data[0:npts*self.rows_per_ptmatch*self.nnz_per_row],self.indices[0:npts*self.rows_per_ptmatch*self.nnz_per_row],self.indptr[0:npts*self.rows_per_ptmatch],self.weights[0:npts*self.rows_per_ptmatch]


def create_CSR_A(collection,matrix_assembly,tform_obj,tile_ids,zvals,output_options):
    #connect to the database
    dbconnection = make_dbconnection(collection)

    file_number=0

    sorter = np.argsort(tile_ids)
    file_chunks = 0
    tiles_used = []

    if output_options['chunks_per_file']==-1:
        nmod=0.1 # np.mod(n+1,0.1) never == 0
    else:
        nmod = output_options['chunks_per_file']

    file_zlist=[]

    for i in np.arange(len(zvals)):
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

            #conservative pre-allocation
            nmatches = len(matches)
            data = np.zeros(tform_obj.nnz_per_row*tform_obj.rows_per_ptmatch*matrix_assembly['npts_max']*nmatches).astype('float64')
            indices = np.zeros(tform_obj.nnz_per_row*tform_obj.rows_per_ptmatch*matrix_assembly['npts_max']*nmatches).astype('int64')
            indptr = np.zeros(tform_obj.rows_per_ptmatch*matrix_assembly['npts_max']*nmatches+1).astype('int64')
            weights = np.zeros(tform_obj.rows_per_ptmatch*matrix_assembly['npts_max']*nmatches).astype('float64')

            indptr[0] = 0
            nrows = 0

            for k in np.arange(nmatches):
                #create the CSR sub-matrix for this tile pair
                d,ind,iptr,wts = tform_obj.CSR_tile_pair(matches[k],pinds[k],qinds[k])
                npts = tform_obj.npts

                if d is None:
                    continue #if npts<nmin, for example

                if i==j:
                    matchweight = matrix_assembly['montage_pt_weight']
                else:
                    matchweight = matrix_assembly['cross_pt_weight']
                    if matrix_assembly['inverse_dz']:
                        matchweight = matchweight/np.abs(j-i+1)

                #add both tile ids to the list
                tiles_used.append(matches[k]['pId'])
                tiles_used.append(matches[k]['qId'])

                #add sub-matrix to global matrix
                global_dind = np.arange(npts*tform_obj.rows_per_ptmatch*tform_obj.nnz_per_row)+nrows*tform_obj.nnz_per_row
                data[global_dind] = d
                indices[global_dind] = ind

                global_rowind = np.arange(npts*tform_obj.rows_per_ptmatch)+nrows
                weights[global_rowind] = wts*matchweight
                indptr[global_rowind+1] = iptr+indptr[nrows]

                nrows += wts.size
           
            del matches 
            #truncate, because we allocated conservatively
            data = data[0:nrows*tform_obj.nnz_per_row]
            indices = indices[0:nrows*tform_obj.nnz_per_row]
            indptr = indptr[0:nrows+1]
            weights = weights[0:nrows]

            #can check CSR form here, but seems to be working
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

def create_regularization(regularization,tform_obj,tile_tforms,output_options):
    #create a regularization vector
    reg = np.ones_like(tile_tforms).astype('float64')*regularization['default_lambda']
    if tform_obj.name=='affine':
        reg[2::3] = reg[2::3]*regularization['translation_factor']
    elif tform_obj.name=='rigid':
        reg[2::4] = reg[2::4]*regularization['translation_factor']
        reg[3::4] = reg[3::4]*regularization['translation_factor']
    if regularization['freeze_first_tile']:
        reg[0:tform_obj.DOF_per_tile] = 1e15

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

def assemble_and_solve(mod,zvals,ingestconn):
    t0 = time.time()
    #make a transform object
    tform_obj = transform_csr(mod.args['transformation'],mod.args['matrix_assembly']['npts_min'],mod.args['matrix_assembly']['npts_max'])

    #get the tile IDs and transforms
    tile_ids,tile_tforms,tile_tspecs,shared_tforms = get_tileids_and_tforms(mod.args['input_stack'],tform_obj,zvals)

    #create A matrix in compressed sparse row (CSR) format
    A,weights,tiles_used = create_CSR_A(mod.args['pointmatch'],mod.args['matrix_assembly'],tform_obj,tile_ids,zvals,mod.args['output_options'])
    tile_ind = np.in1d(tile_ids,tiles_used)
    filt_tspecs = tile_tspecs[tile_ind]
    filt_tforms = tile_tforms[np.repeat(tile_ind,tform_obj.DOF_per_tile)]
  
    del tile_ids,tiles_used,tile_tforms,tile_ind,tile_tspecs
    
    #create the regularization vectors
    mont_reg = create_regularization(mod.args['regularization'],tform_obj,filt_tforms,mod.args['output_options'])

    #regularized least squares
    ATW = A.transpose().dot(weights)
    K = ATW.dot(A) + mont_reg
    print ' created K: %d x %d'%K.shape
    Lm = mont_reg.dot(filt_tforms)

    del weights,filt_tforms,mont_reg

    #solve
    x = spsolve(K,Lm)
    err = A.dot(x)

    message = '***-------------***\n'
    message = message + ' assembled and solved in %0.1f sec\n'%(time.time()-t0)
    message = message + ' precision [norm(Kx-Lm)/norm(Lm)] = %0.1e\n'%(np.linalg.norm(K.dot(x)-Lm)/np.linalg.norm(Lm))
    message = message + ' error     [norm(Ax-b)] = %0.3f\n'%(np.linalg.norm(err))
    message = message + ' avg cartesian projection displacement per point [mean(|Ax|)+/-std(|Ax|)] : %0.1f +/- %0.1f pixels'%(np.abs(err).mean(),np.abs(err).std())
    print message
    del A,K,ATW,Lm

    #replace the last transform in the tilespec with the new one
    for m in np.arange(len(filt_tspecs)):
        if mod.args['transformation']=='affine':
            filt_tspecs[m].tforms[-1].M[0,0] = x[m*6+0]
            filt_tspecs[m].tforms[-1].M[0,1] = x[m*6+1]
            filt_tspecs[m].tforms[-1].M[0,2] = x[m*6+2]
            filt_tspecs[m].tforms[-1].M[1,0] = x[m*6+3]
            filt_tspecs[m].tforms[-1].M[1,1] = x[m*6+4]
            filt_tspecs[m].tforms[-1].M[1,2] = x[m*6+5]
        elif mod.args['transformation']=='rigid':
            filt_tspecs[m].tforms[-1].M[0,0] = x[m*4+0]
            filt_tspecs[m].tforms[-1].M[0,1] = x[m*4+1]
            filt_tspecs[m].tforms[-1].M[0,2] = x[m*4+2]
            filt_tspecs[m].tforms[-1].M[1,0] = -x[m*4+1]
            filt_tspecs[m].tforms[-1].M[1,1] = x[m*4+0]
            filt_tspecs[m].tforms[-1].M[1,2] = x[m*4+3]

    #renderapi.client.import_tilespecs(output['name'],tspout,sharedTransforms=shared_tforms,render=ingestconn)
    if ingestconn!=None:
        renderapi.client.import_tilespecs_parallel(mod.args['output_stack']['name'],filt_tspecs.tolist(),sharedTransforms=shared_tforms,render=ingestconn,close_stack=False)
        print message
    del shared_tforms,x,filt_tspecs
    
if __name__=='__main__':
    t0 = time.time()
    mod = argschema.ArgSchemaParser(schema_type=EMA_Schema)

    #specify the z values
    zvals = np.arange(mod.args['first_section'],mod.args['last_section']+1)

    ingestconn=None
    #make a connection to the new stack
    if mod.args['output_options']['output_mode']=='stack':
        ingestconn = make_dbconnection(mod.args['output_stack'])
        renderapi.stack.create_stack(mod.args['output_stack']['name'],render=ingestconn)

    if mod.args['solve_type']=='montage':
        for z in zvals:
            assemble_and_solve(mod,[z],ingestconn)
    elif mod.args['solve_type']=='3D':
        assemble_and_solve(mod,zvals,ingestconn)
     
    if ingestconn!=None:
        if mod.args['close_stack']:
            renderapi.stack.set_stack_state(mod.args['output_stack']['name'],state='COMPLETE',render=ingestconn)

    print 'total time: %0.1f'%(time.time()-t0)

