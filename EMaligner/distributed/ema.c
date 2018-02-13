/** @file ema.c
  * @brief functions used by em_dist_solve
  *
  * 
*/
#include "ema.h"

PetscErrorCode CountFiles(MPI_Comm COMM, char indexname[], int *nfiles){
  PetscErrorCode ierr;
  FILE *fp; 
  int ch;
  PetscMPIInt rank;
  ierr = MPI_Comm_rank(COMM,&rank);CHKERRQ(ierr);
  if (rank==0){
    *nfiles = 0;
    ierr = PetscFOpen(COMM,indexname,"r",&fp);CHKERRQ(ierr);
    while(!feof(fp)){
      ch = fgetc(fp);
      if(ch == '\n'){
        *nfiles+=1;
      }
    }
    ierr = PetscFClose(COMM,fp);CHKERRQ(ierr);
  }
  MPI_Bcast(nfiles,1,MPI_INT,0,COMM);
  return ierr;
}

PetscErrorCode ReadIndex(MPI_Comm COMM, char indexname[], int nfiles, char *csrnames[], PetscInt **metadata){
  PetscErrorCode ierr;
  FILE *fp; 
  PetscMPIInt rank;
  int i,j;
  char *dir,*tmp,tmpname[200];
  tmp = strdup(indexname);
  dir = strdup(dirname(tmp));
  ierr = MPI_Comm_rank(COMM,&rank);CHKERRQ(ierr);
  if (rank==0){
    ierr = PetscFOpen(COMM,indexname,"r",&fp);CHKERRQ(ierr);
    for (i=0;i<nfiles;i++){
      fscanf(fp,"%*s %s",tmpname);
      sprintf(csrnames[i],"%s/%s",dir,tmpname);
      for (j=0;j<4;j++){
        fscanf(fp," %*s %ld",&metadata[i][j]);
        fscanf(fp,"\n");
      }
    }
    ierr = PetscFClose(COMM,fp);CHKERRQ(ierr);
  }
  for (i=0;i<nfiles;i++){
    MPI_Bcast(csrnames[i],PETSC_MAX_PATH_LEN,MPI_CHAR,0,COMM);
    MPI_Bcast(metadata[i],4,MPIU_INT,0,COMM);
  }
  free(dir);
  free(tmp);
  return ierr;
}

PetscErrorCode SetFiles(MPI_Comm COMM, int nfiles, PetscInt *firstfile,PetscInt *lastfile){
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

PetscErrorCode ReadVec(MPI_Comm COMM,PetscViewer viewer,char *varname,Vec *newvec,long *n){
  PetscErrorCode ierr;
  char           name[256];
  sprintf(name,"%s",varname);
  ierr = VecCreate(COMM,newvec);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)*newvec,name);CHKERRQ(ierr);
  ierr = VecLoad(*newvec,viewer);CHKERRQ(ierr);
  ierr = VecGetSize(*newvec,n);CHKERRQ(ierr);
  return ierr;
}

PetscErrorCode ReadIndexSet(MPI_Comm COMM,PetscViewer viewer,char *varname,IS *newIS,long *n){
  PetscErrorCode ierr;
  char           name[256];
  sprintf(name,"%s",varname);
  ierr = ISCreate(COMM,newIS);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)*newIS,name);CHKERRQ(ierr);
  ierr = ISLoad(*newIS,viewer);CHKERRQ(ierr);
  ierr = ISGetSize(*newIS,n);CHKERRQ(ierr);
  return ierr;
}

PetscErrorCode ShowMatInfo(Mat *m,const char *mesg){
  PetscErrorCode ierr;
  MatInfo info;
  PetscBool      isassembled;
  PetscInt       rowcheck,colcheck;  
  
  printf("%s:\n",mesg);
  ierr = MatAssembled(*m,&isassembled);CHKERRQ(ierr);
  printf(" is assembled: %d\n",isassembled);
  ierr = MatGetSize(*m,&rowcheck,&colcheck);CHKERRQ(ierr);
  printf(" is of size %ld x %ld\n",rowcheck,colcheck);
  ierr = MatGetInfo(*m,MAT_LOCAL,&info);CHKERRQ(ierr);
  printf(" block_size: %f\n",info.block_size); 
  printf(" nz_allocated: %f\n",info.nz_allocated);
  printf(" nz_used: %f\n",info.nz_used);
  printf(" nz_unneeded: %f\n",info.nz_unneeded);
  printf(" memory: %f\n",info.memory); 
  printf(" assemblies: %f\n",info.assemblies);
  printf(" mallocs: %f\n",info.mallocs);
  printf(" fill_ratio_given: %f\n",info.fill_ratio_given);
  printf(" fill_ratio_needed: %f\n",info.fill_ratio_needed);
  return ierr;
}

void GetGlobalLocalCounts(int nfiles, PetscInt **metadata, int local_firstfile, int local_lastfile, PetscInt *global_nrow, PetscInt *global_ncol, PetscInt *global_nnz, PetscInt *local_nrow, PetscInt *local_nnz, PetscInt *local_row0){
  PetscInt cmin,cmax;
  int i;

  *global_nrow=0;
  *global_ncol=0;
  *global_nnz=0;
  *local_nrow=0;
  *local_nnz=0;
  *local_row0=0;

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

PetscErrorCode ReadLocalCSR(MPI_Comm COMM, char *csrnames[], int local_firstfile, int local_lastfile, PetscInt *local_indptr, PetscInt *local_jcol, PetscScalar *local_data, PetscScalar *local_weights){

  PetscViewer    viewer;                               //viewer object for reading files
  IS             indices,indptr;
  Vec            data,weights;
  PetscErrorCode ierr;
  int i,j;

  //create the distributed matrix A
  PetscInt vcnt,niptr;
  PetscInt roff=0,roff2=0,zoff=0;
  PetscInt *iptr,*iptrcpy,*jcol,poff=0;
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
    roff2 += niptr;

    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ISDestroy(&indptr);
    VecDestroy(&data);
    VecDestroy(&weights);
    ISDestroy(&indices);
  }

  return ierr;
}

PetscErrorCode CreateW(MPI_Comm COMM,PetscScalar *local_weights,PetscInt local_nrow,PetscInt local_row0,PetscInt global_nrow,Mat *W){
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

  return ierr;
}

PetscErrorCode CreateL(MPI_Comm COMM,char *dir,PetscInt local_nrow,PetscInt global_nrow,Mat *L){
  PetscErrorCode ierr;
  PetscViewer viewer;
  Vec global_reg;
  PetscMPIInt rank;
  PetscInt junk;
  char tmp[200];

  ierr = MPI_Comm_rank(COMM,&rank);CHKERRQ(ierr);

  VecCreate(COMM,&global_reg);
  VecSetSizes(global_reg,local_nrow,global_nrow);
  VecSetType(global_reg,VECMPI);
  sprintf(tmp,"%s/regularization.h5",dir);
  ierr = PetscViewerHDF5Open(COMM,tmp,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = ReadVec(COMM,viewer,(char *)"lambda",&global_reg,&junk);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,L);CHKERRQ(ierr);
  ierr = MatSetSizes(*L,local_nrow,local_nrow,global_nrow,global_nrow);CHKERRQ(ierr);
  ierr = MatSetType(*L,MATMPIAIJ);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(*L,1,NULL,0,NULL);CHKERRQ(ierr);
  ierr = MatDiagonalSet(*L,global_reg,INSERT_VALUES);CHKERRQ(ierr);

  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  VecDestroy(&global_reg);
  return ierr;
}

