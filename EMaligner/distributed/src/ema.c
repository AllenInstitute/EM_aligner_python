/** @file ema.c
  *
  *
*/
#include "ema.h"

/*! Rank 0 process counts the number of files from index.txt and broadcasts the result */
PetscErrorCode CountFiles(MPI_Comm COMM, char indexname[], int *nfiles){
/**
 * @param[in] COMM The MPI communicator, probably PETSC_COMM_WORLD.
 * @param[in] indexname The full path to index.txt, given as command-line argument with -f.
 * @param[out] *nfiles The result of the count, returned to main().
*/
  PetscErrorCode ierr;
  PetscMPIInt rank;
  hid_t       file,space,dset;
  herr_t      status;
  hsize_t     dims[1];
  int ndims;

  ierr = MPI_Comm_rank(COMM,&rank);CHKERRQ(ierr);
  if (rank==0){
    file = H5Fopen(indexname, H5F_ACC_RDONLY, H5P_DEFAULT);
    dset = H5Dopen(file,"datafile_names",H5P_DEFAULT);
    space = H5Dget_space(dset);
    ndims = H5Sget_simple_extent_dims (space, dims, NULL);
    *nfiles = dims[0];
    status = H5Dclose (dset);
    status = H5Sclose (space);
    status = H5Fclose (file);
  }
  MPI_Bcast(nfiles,1,MPI_INT,0,COMM);
  return ierr;
}

/*! @brief Rank 0 processor reads metadata and broadcasts */
PetscErrorCode ReadMetadata(MPI_Comm COMM, char indexname[], int nfiles, char *csrnames[], PetscInt **metadata){
/**
 * @param[in] COMM The MPI communicator, probably PETSC_COMM_WORLD.
 * @param[in] indexname The full path to solution_input.h5, given as command-line argument with -f.
 * @param[in] nfiles The number of files listed in solution_input.h5 file, read from previous CountFiles() call.
 * @param[out] *csrnames An array of file names from solution_input.h5, populated by this function.
 * @param[out] *metadata An array of metadata from index.txt, populated by this function.
*/
  PetscErrorCode ierr;
  PetscViewer viewer;
  PetscMPIInt rank;
  PetscInt junk;
  int i,j;
  char *dir,*tmp,tmpname[200];
  char **rdata;
  hid_t       file, filetype, memtype, space, dset;
  herr_t      status;
  hsize_t     dims[1];
  int         ndims;

  tmp = strdup(indexname);
  dir = strdup(dirname(tmp));
  ierr = MPI_Comm_rank(COMM,&rank);CHKERRQ(ierr);
  if (rank==0){
    file = H5Fopen(indexname, H5F_ACC_RDONLY, H5P_DEFAULT);
    dset = H5Dopen(file,"datafile_names",H5P_DEFAULT);
    filetype = H5Dget_type (dset);
    space = H5Dget_space(dset);
    ndims = H5Sget_simple_extent_dims(space, dims, NULL);
    rdata = (char **)malloc(dims[0]*sizeof (char *));
    memtype = H5Tcopy(H5T_C_S1);
    status = H5Tset_size (memtype, H5T_VARIABLE);
    status = H5Dread (dset, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, rdata);
    for (i=0;i<nfiles;i++){
        sprintf(csrnames[i],"%s/%s",dir,rdata[i]);
    }
    status = H5Dvlen_reclaim(memtype, space, H5P_DEFAULT, rdata);
    free (rdata);
    status = H5Dclose (dset);
    status = H5Sclose (space);

    PetscInt *row,*mxcol,*mncol,*nnz;
    row = (PetscInt *)malloc(nfiles*sizeof(PetscInt));
    mxcol = (PetscInt *)malloc(nfiles*sizeof(PetscInt));
    mncol = (PetscInt *)malloc(nfiles*sizeof(PetscInt));
    nnz = (PetscInt *)malloc(nfiles*sizeof(PetscInt));
    dset = H5Dopen (file, "datafile_nrows", H5P_DEFAULT);
    status = H5Dread (dset, H5T_NATIVE_LONG, H5S_ALL, H5S_ALL, H5P_DEFAULT,row);
    dset = H5Dopen (file, "datafile_mincol", H5P_DEFAULT);
    status = H5Dread (dset, H5T_NATIVE_LONG, H5S_ALL, H5S_ALL, H5P_DEFAULT,mncol);
    dset = H5Dopen (file, "datafile_maxcol", H5P_DEFAULT);
    status = H5Dread (dset, H5T_NATIVE_LONG, H5S_ALL, H5S_ALL, H5P_DEFAULT,mxcol);
    dset = H5Dopen (file, "datafile_nnz", H5P_DEFAULT);
    status = H5Dread (dset, H5T_NATIVE_LONG, H5S_ALL, H5S_ALL, H5P_DEFAULT,nnz);
    for (i=0;i<nfiles;i++){
        metadata[i][0] = row[i];
        metadata[i][1] = mncol[i];
        metadata[i][2] = mxcol[i];
        metadata[i][3] = nnz[i];
    }
    free(row);
    free(mncol);
    free(mxcol);
    free(nnz);
    status = H5Dclose (dset);
    status = H5Tclose (filetype);
    status = H5Tclose (memtype);
    status = H5Fclose (file);
  }
  for (i=0;i<nfiles;i++){
    MPI_Bcast(csrnames[i],PETSC_MAX_PATH_LEN,MPI_CHAR,0,COMM);
    MPI_Bcast(metadata[i],4,MPIU_INT,0,COMM);
  }
  free(dir);
  free(tmp);
  return ierr;
}

PetscErrorCode CopyDataSetstoSolutionOut(MPI_Comm COMM, char indexname [],char outputname []){
  hid_t       filein, fileout, filetype, memtype, space, dset,dsetout;
  herr_t      status;
  hsize_t     dims[1];
  int nds = 10;
  char *copyds[10] = {"input_args", "used_tile_ids", "unused_tile_ids", 
	  "datafile_maxcol", "datafile_mincol", "datafile_names",
	  "datafile_nnz", "datafile_nrows", "lambda", "transform_list"};
  char **rdata;
  int ndims,i;

  filein = H5Fopen (indexname, H5F_ACC_RDONLY, H5P_DEFAULT);
  fileout = H5Fcreate (outputname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  for (i=0;i<nds;i++){ 
      dset = H5Dopen (filein,copyds[i],H5P_DEFAULT);
      filetype = H5Dget_type (dset);
      space = H5Dget_space (dset);
      ndims = H5Sget_simple_extent_dims (space, dims, NULL);
      rdata = (char **) malloc (dims[0] * sizeof (char *));
      memtype = H5Tcopy (H5T_C_S1);
      status = H5Tset_size (memtype, H5T_VARIABLE);
      status = H5Dread (dset, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, rdata);
      dsetout = H5Dcreate (fileout, copyds[i], filetype, space, H5P_DEFAULT, H5P_DEFAULT,H5P_DEFAULT);
      status = H5Dwrite (dsetout, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, rdata);
      status = H5Dclose (dset);
      status = H5Dclose (dsetout);
      status = H5Tclose (filetype);
      status = H5Tclose (memtype);
      status = H5Dvlen_reclaim(memtype, space, H5P_DEFAULT, rdata);
      free (rdata);
  }
  status = H5Fclose (filein);
  status = H5Fclose (fileout);
}

/*! Split the list of files roughly evenly amongst all the workers. */
PetscErrorCode SetFiles(MPI_Comm COMM, int nfiles, PetscInt *firstfile,PetscInt *lastfile){
/**
 * @param[in] COMM The MPI communicator, probably PETSC_COMM_WORLD.
 * @param[in] nfiles The number of lines in index.txt, read from previous CountFiles() call.
 * @param[out] *firstfile The index of the first file for this worker.
 * @param[out] *lastfile The index of the first file for this worker.
*/
  PetscErrorCode ierr;
  PetscMPIInt rank,size;
  float avg;
  int i,rem,last;

  ierr = MPI_Comm_rank(COMM,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(COMM,&size);CHKERRQ(ierr);
  if (nfiles<=size){
    for (i=0;i<size;i++){
      if (rank==i){
        *firstfile=i;
        if (i>nfiles-1){
          *lastfile=i-1;
        }
        else{
          *lastfile=i;
        }
      }
    }
  }
  else{
    avg = nfiles/(float)size;
    rem = nfiles%size;
    last=0;
    for (i=0;i<size;i++){
      if (rank==i){
        *firstfile=last;
      }
      if (rem-->0){
        last=last+ceil(avg);
      }
      else{
        last=last+floor(avg);
      }
      if (rank==i){
        *lastfile=last-1;
      }
    }
  }
  return ierr;
}

/*! Read data from a PetscViewer into a Vec (scalar) object. This version good for single-rank reads.*/
PetscErrorCode ReadVec(MPI_Comm COMM,PetscViewer viewer,char *varname,Vec *newvec,PetscInt *n){
/**
 * @param[in] COMM The MPI communicator, PETSC_COMM_SELF or PETSC_COMM_WORLD.
 * @param[in] viewer A PetscViewer instance.
 * @param[in] *varname The new vector will have this name. For reading from hdf5 files, this name must match the dataset name in the file.
 * @param[out] *newvec The new vector object.
 * @param[out] *n The number of entries in the new object.
*/
  PetscErrorCode ierr;
  char           name[256];
  sprintf(name,"%s",varname);
  ierr = VecCreate(COMM,newvec);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)*newvec,name);CHKERRQ(ierr);
  ierr = VecLoad(*newvec,viewer);CHKERRQ(ierr);
  ierr = VecGetSize(*newvec,n);CHKERRQ(ierr);
  return ierr;
}

/*! Read data from a PetscViewer into a Vec (scalar) object. This version good for anticipating allocation across nodes.*/
PetscErrorCode ReadVecWithSizes(MPI_Comm COMM,PetscViewer viewer,char *varname,Vec *newvec,PetscInt *n,PetscInt nlocal,PetscInt nglobal,PetscBool trunc){
/**
 * @param[in] COMM The MPI communicator, PETSC_COMM_SELF or PETSC_COMM_WORLD.
 * @param[in] viewer A PetscViewer instance.
 * @param[in] *varname The new vector will have this name. For reading from hdf5 files, this name must match the dataset name in the file.
 * @param[out] *newvec The new vector object.
 * @param[out] *n The number of entries in the new object.
 * @param[in] nlocal The number of entries the local rank will own.
 * @param[in] nglobal The total number of expected entries in newvec.
 * @param[in] trunc Boolean whether to truncate the imported data.
*/
  PetscErrorCode ierr;
  Vec            tmpvec;
  PetscInt       i,ntmp,low,high,*indx;
  PetscScalar    *vtmp;
  char           name[256];

  sprintf(name,"%s",varname);
  ierr = VecCreate(COMM,newvec);CHKERRQ(ierr);
  ierr = VecSetSizes(*newvec,nlocal,nglobal);CHKERRQ(ierr);
  ierr = VecSetType(*newvec,VECMPI);CHKERRQ(ierr);
  if(trunc){
    ierr = VecCreate(MPI_COMM_SELF,&tmpvec);CHKERRQ(ierr);
    ierr = VecSetType(tmpvec,VECSEQ);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)tmpvec,name);CHKERRQ(ierr);
    ierr = VecLoad(tmpvec,viewer);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(*newvec,&low,&high);CHKERRQ(ierr);
    indx = (PetscInt *)malloc((high-low)*sizeof(PetscInt));
    vtmp = (PetscScalar *)malloc((high-low)*sizeof(PetscScalar));
    for (i=low;i<high;i++){
      indx[i-low]=i;
    }
    ierr = VecGetValues(tmpvec,high-low,indx,vtmp);CHKERRQ(ierr);
    ierr = VecSetValues(*newvec,high-low,indx,vtmp,INSERT_VALUES);CHKERRQ(ierr);
    free(indx);
    free(vtmp);
    ierr = VecAssemblyBegin(*newvec);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(*newvec);CHKERRQ(ierr);
    ierr = VecDestroy(&tmpvec);CHKERRQ(ierr);
  }
  else{
    ierr = PetscObjectSetName((PetscObject)*newvec,name);CHKERRQ(ierr);
    ierr = VecLoad(*newvec,viewer);CHKERRQ(ierr);
  }
  ierr = VecGetSize(*newvec,&ntmp);CHKERRQ(ierr);
  return ierr;
}

/*! Read data from a PetscViewer into an IS (Index Set, i.e. integer) object.*/
PetscErrorCode ReadIndexSet(MPI_Comm COMM,PetscViewer viewer,char *varname,IS *newIS,PetscInt *n){
/**
 * @param[in] COMM The MPI communicator, PETSC_COMM_SELF or PETSC_COMM_WORLD.
 * @param[in] viewer A PetscViewer instance.
 * @param[in] *varname The new IS will have this name. For reading from hdf5 files, this name must match the dataset name in the file.
 * @param[out] *newIS The new IS object.
 * @param[out] *n The number of entries in the new object.
*/
  PetscErrorCode ierr;
  char           name[256];
  sprintf(name,"%s",varname);
  ierr = ISCreate(COMM,newIS);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)*newIS,name);CHKERRQ(ierr);
  ierr = ISLoad(*newIS,viewer);CHKERRQ(ierr);
  ierr = ISGetSize(*newIS,n);CHKERRQ(ierr);
  return ierr;
}

/*! Print to stdout MatInfo for a Mat object.*/
PetscErrorCode ShowMatInfo(MPI_Comm COMM,Mat *m,const char *mesg){
/**
 * @param[in] COMM The MPI communicator PETSC_COMM_WORLD.
 * @param[in] *m The matrix.
 * @param[in] *mesg Name or some other string to prepend the output.
*/
  PetscErrorCode ierr;
  MatInfo info;
  PetscBool      isassembled;
  PetscInt       rowcheck,colcheck;
  PetscMPIInt rank;

  ierr = MPI_Comm_rank(COMM,&rank);CHKERRQ(ierr);
  ierr = MatGetInfo(*m,MAT_GLOBAL_SUM,&info);CHKERRQ(ierr);
  if (rank==0){
    printf("%s info from rank %d:\n",mesg,rank);
    ierr = MatAssembled(*m,&isassembled);CHKERRQ(ierr);
    printf(" is assembled: %d\n",isassembled);
    ierr = MatGetSize(*m,&rowcheck,&colcheck);CHKERRQ(ierr);
    printf(" global size %ld x %ld\n",rowcheck,colcheck);
    //ierr = MatGetInfo(*m,MAT_GLOBAL_SUM,&info);CHKERRQ(ierr);
    printf(" block_size: %f\n",info.block_size);
    printf(" nz_allocated: %f\n",info.nz_allocated);
    printf(" nz_used: %f\n",info.nz_used);
    printf(" nz_unneeded: %f\n",info.nz_unneeded);
    printf(" memory: %f\n",info.memory);
    printf(" assemblies: %f\n",info.assemblies);
    printf(" mallocs: %f\n",info.mallocs);
    printf(" fill_ratio_given: %f\n",info.fill_ratio_given);
    printf(" fill_ratio_needed: %f\n",info.fill_ratio_needed);
  }
  ierr = MatGetInfo(*m,MAT_LOCAL,&info);CHKERRQ(ierr);
  if (rank==0){
    printf("%s local info from rank %d:\n",mesg,rank);
    ierr = MatAssembled(*m,&isassembled);CHKERRQ(ierr);
    printf(" is assembled: %d\n",isassembled);
    ierr = MatGetSize(*m,&rowcheck,&colcheck);CHKERRQ(ierr);
    printf(" global size %ld x %ld\n",rowcheck,colcheck);
    //ierr = MatGetInfo(*m,MAT_GLOBAL_SUM,&info);CHKERRQ(ierr);
    printf(" block_size: %f\n",info.block_size);
    printf(" nz_allocated: %f\n",info.nz_allocated);
    printf(" nz_used: %f\n",info.nz_used);
    printf(" nz_unneeded: %f\n",info.nz_unneeded);
    printf(" memory: %f\n",info.memory);
    printf(" assemblies: %f\n",info.assemblies);
    printf(" mallocs: %f\n",info.mallocs);
    printf(" fill_ratio_given: %f\n",info.fill_ratio_given);
    printf(" fill_ratio_needed: %f\n",info.fill_ratio_needed);
  }
  return ierr;
}

/*! Use metadata to determine global and local sizes and indices.*/
void GetGlobalLocalCounts(int nfiles, PetscInt **metadata, int local_firstfile, int local_lastfile, PetscInt *global_nrow, PetscInt *global_ncol, PetscInt *global_nnz, PetscInt *local_nrow, PetscInt *local_nnz, PetscInt *local_row0){
/**
 * @param[in] nfiles The number of CSR.hdf5 files, from a previous call to CountFiles()
 * @param[in] **metadata Metadata from index.txt from a previous call to ReadIndex()
 * @param[in] local_firstfile, local_lastfile Indices in the list of files for the local worker.
 * @param[out] global_nrow,global_ncol,global_nnz Read from the metadata, these describe A, the matrix to be read from the hdf5 files.
 * @param[out] local_nrow,local_nnz,local_row0 Read from the metadata, these are concatentations of the metadata for all the owned files for one worker.
*/
  PetscInt cmin,cmax;
  int i;

  *global_nrow=0;
  *global_ncol=0;
  *global_nnz=0;
  *local_nrow=0;
  *local_nnz=0;
  *local_row0=0;
  cmin=0;
  cmax=0;
  for (i=0;i<nfiles;i++){
    *global_nrow+=metadata[i][0];
    *global_nnz+=metadata[i][3];
    if (i<local_firstfile){
      *local_row0+=metadata[i][0];
    }
    if (i==0){
      cmin=metadata[i][1];
      cmax=metadata[i][2];
    }
    else{
      if (metadata[i][1]<cmin){
        cmin = metadata[i][1];
      }
      if (metadata[i][2]>cmax){
        cmax = metadata[i][2];
      }
    }
  }
  *global_ncol = cmax-cmin+1;
  for (i=local_firstfile;i<=local_lastfile;i++){
    *local_nnz+=metadata[i][3];
    *local_nrow+=metadata[i][0];
  }
  //printf(" files %d through %d of %d",local_firstfile,local_lastfile,nfiles);
  //printf(" nnz %ld of %ld\n",*local_nnz,*global_nnz);
  //printf(" nrow %ld of %ld\n",*local_nrow,*global_nrow);
  //printf(" localrow %ld\n",*local_row0);
  return;
}

/*! Build local CSR block by sequentially reading in local hdf5 files.*/
PetscErrorCode ReadLocalCSR(MPI_Comm COMM, char *csrnames[], int local_firstfile, int local_lastfile, PetscInt *local_indptr, PetscInt *local_jcol, PetscScalar *local_data, PetscScalar *local_weights){
/**
 * @param[in] COMM The MPI communicator, PETSC_COMM_SELF.
 * @param[in] *csrnames[] The names of the CSR.hdf5 files
 * @param[in] local_firstfile,local_lastfile Indices of which files are handled by which rank.
 * @param[out] *local_indptr,*local_jcol,*local_data Holds the concatenated CSR arrays for this rank.
 * @param[out] *local_weights Holds the concatenated weights for this rank.
*/

  PetscViewer    viewer;                               //viewer object for reading files
  IS             indices,indptr;
  Vec            data,weights;
  PetscErrorCode ierr;
  int i,j;

  //create the distributed matrix A
  PetscInt vcnt,niptr;
  PetscInt roff=0,roff2=0,zoff=0;
  const PetscInt *iptr,*jcol;
  PetscInt poff=0;
  PetscScalar *a,*w;
  //these will concat the multiple files per processor
  for (i=local_firstfile;i<=local_lastfile;i++){
    //open the file
    ierr = PetscViewerHDF5Open(COMM,csrnames[i],FILE_MODE_READ,&viewer);CHKERRQ(ierr);
    //indptr
    ierr = ReadIndexSet(COMM,viewer,(char *)"indptr",&indptr,&niptr);CHKERRQ(ierr);
    ISGetIndices(indptr,&iptr);
    for(j=1;j<niptr;j++){
      local_indptr[j+roff] = iptr[j]+poff;
    }
    ISRestoreIndices(indptr,&iptr);
    poff = local_indptr[niptr-1+roff];
    roff += niptr-1;

    //indices
    ierr = ReadIndexSet(COMM,viewer,(char *)"indices",&indices,&vcnt);CHKERRQ(ierr);
    ISGetIndices(indices,&jcol);
    memcpy(&local_jcol[zoff],jcol,vcnt*sizeof(PetscInt));
    ISRestoreIndices(indices,&jcol);

    //data
    ierr = ReadVec(COMM,viewer,(char *)"data",&data,&vcnt);CHKERRQ(ierr);
    VecGetArray(data,&a);
    memcpy(&local_data[zoff],a,vcnt*sizeof(PetscScalar));
    VecRestoreArray(data,&a);
    zoff+=vcnt;

    //weights
    ierr = ReadVec(COMM,viewer,(char *)"weights",&weights,&vcnt);CHKERRQ(ierr);
    VecGetArray(weights,&w);
    memcpy(&local_weights[roff2],w,vcnt*sizeof(PetscScalar));
    VecRestoreArray(weights,&w);
    roff2 += vcnt;

    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ISDestroy(&indptr);
    VecDestroy(&data);
    VecDestroy(&weights);
    ISDestroy(&indices);
  }

  return ierr;
}

/*! Creates a diagonal matrix with the weights as the entries.*/
PetscErrorCode CreateW(MPI_Comm COMM,PetscScalar *local_weights,PetscInt local_nrow,PetscInt local_row0,PetscInt global_nrow,Mat *W){
/**
 * @param[in] COMM The MPI communicator, PETSC_COMM_SELF.A
 * @param[in] *local_weights Passed into this function to get built into W.
 * @param[in] local_nrow Number of rows for this rank
 * @param[in] local_row0 The starting row for this rank.
 * @param[in] global_nrow The total length of the weights vector (N). W is NxN.
 * @param[out] *W The matrix created by this function.
*/
  PetscErrorCode ierr;
  Vec global_weights;
  PetscInt *indx;
  int i;

  ierr = VecCreate(COMM,&global_weights);CHKERRQ(ierr);
  ierr = VecSetSizes(global_weights,local_nrow,global_nrow);CHKERRQ(ierr);
  ierr = VecSetType(global_weights,VECMPI);CHKERRQ(ierr);
  indx = (PetscInt *)malloc(local_nrow*sizeof(PetscInt));
  for (i=0;i<local_nrow;i++){
    indx[i] = local_row0+i;
  }
  ierr = VecSetValues(global_weights,local_nrow,indx,local_weights,INSERT_VALUES);CHKERRQ(ierr);
  free(indx);

  ierr = MatCreate(PETSC_COMM_WORLD,W);CHKERRQ(ierr);
  ierr = MatSetSizes(*W,local_nrow,local_nrow,global_nrow,global_nrow);CHKERRQ(ierr);
  ierr = MatSetType(*W,MATMPIAIJ);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(*W,1,NULL,0,NULL);CHKERRQ(ierr);
  ierr = MatDiagonalSet(*W,global_weights,INSERT_VALUES);CHKERRQ(ierr);

  ierr = VecDestroy(&global_weights);CHKERRQ(ierr);
  return ierr;
}

/*! Creates a diagonal matrix with the weights as the entries.*/
PetscErrorCode CreateL(MPI_Comm COMM,char *dir,PetscInt local_nrow,PetscInt global_nrow,PetscBool trunc,Mat *L){
/**
 * @param[in] COMM The MPI communicator, PETSC_COMM_SELF.A
 * @param[in] local_nrow Number of rows for this rank
 * @param[in] local_row0 The starting row for this rank.
 * @param[in] global_nrow The total length of the weights vector (N). L is NxN.
 * @param[out] *L The matrix created by this function.
*/
  PetscErrorCode ierr;
  PetscViewer viewer;
  Vec global_reg;
  PetscMPIInt rank;
  PetscInt junk;
  char tmp[200];

  ierr = MPI_Comm_rank(COMM,&rank);CHKERRQ(ierr);

  //VecCreate(COMM,&global_reg);
  //VecSetSizes(global_reg,local_nrow,global_nrow);
  //VecSetType(global_reg,VECMPI);
  sprintf(tmp,"%s/solution_input.h5",dir);
  ierr = PetscViewerHDF5Open(COMM,tmp,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = ReadVecWithSizes(COMM,viewer,(char *)"lambda",&global_reg,&junk,local_nrow,global_nrow,trunc);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,L);CHKERRQ(ierr);
  ierr = MatSetSizes(*L,local_nrow,local_nrow,global_nrow,global_nrow);CHKERRQ(ierr);
  ierr = MatSetType(*L,MATMPIAIJ);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(*L,1,NULL,0,NULL);CHKERRQ(ierr);
  ierr = MatDiagonalSet(*L,global_reg,INSERT_VALUES);CHKERRQ(ierr);
  MatAssemblyBegin(*L,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(*L,MAT_FINAL_ASSEMBLY);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  VecDestroy(&global_reg);
  return ierr;
}

/*! Counts how many RHS vectors are stored in the regularization file. */
PetscErrorCode CountRHS(MPI_Comm COMM,char *dir,PetscInt *nRHS){
/**
 * @param[in] COMM The MPI communicator, PETSC_COMM_SELF.A
 * @param[in] dir string containing the directory
 * @param[out] number of RHS vectors
*/
  PetscErrorCode ierr;
  PetscViewer viewer;
  PetscInt junk;
  IS test;
  char tmp[200];

  *nRHS=0;
  sprintf(tmp,"%s/solution_input.h5",dir);
  ierr = PetscViewerHDF5Open(COMM,tmp,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = ReadIndexSet(COMM,viewer,(char *)"transform_list",&test,&junk);CHKERRQ(ierr);
  *nRHS=junk;
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  return ierr;
}

/*! Read the RHS vectors stored in the regularization file. */
PetscErrorCode ReadRHS(MPI_Comm COMM,char *dir,PetscInt local_nrow,PetscInt global_nrow,PetscInt nrhs,PetscBool trunc,Vec rhs[]){
/**
 * @param[in] COMM The MPI communicator, PETSC_COMM_SELF.A
 * @param[in] dir string containing the directory
 * @param[in] local_nrow Number of local rows this rank will own
 * @param[in] global_nrow Number of global rows.
 * @param[in] nrhs Number of rhs vectors to be read.
 * @param[out] rhs[] RHS vectors
*/
  PetscErrorCode ierr;
  PetscViewer viewer;
  PetscInt junk;
  char tmp[200];
  int i;

  sprintf(tmp,"%s/solution_input.h5",dir);
  ierr = PetscViewerHDF5Open(COMM,tmp,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  for(i=0;i<nrhs;i++){
    sprintf(tmp,"transforms_%d",i);
    //ierr = VecCreate(COMM,&rhs[i]);CHKERRQ(ierr);
    //ierr = VecSetSizes(rhs[i],local_nrow,global_nrow);CHKERRQ(ierr);
    //ierr = VecSetType(rhs[i],VECMPI);CHKERRQ(ierr);
    ierr = ReadVecWithSizes(COMM,viewer,tmp,&rhs[i],&junk,local_nrow,global_nrow,trunc);CHKERRQ(ierr);
  }
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  return ierr;
}
