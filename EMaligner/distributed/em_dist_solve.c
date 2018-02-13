/** @file em_dist_solve.c
  * @brief Main method for distributed solve of Kx=Lm
  *
  * Reads A from file, reads regularizations and constraints, computes K, solves for x.
*/
static char help[] = "Testing hdf5 I/O\n\n";

#include <stdio.h>
#include <libgen.h>
#include <petsctime.h>
#include "ema.h"

/** @brief em_dist_solve main
  * 
*/
int main(int argc,char **args)
{
  //KSP            ksp;      /* linear solver context */
  PetscViewer    viewer;                               //viewer object for reading files
  PetscMPIInt    rank,size;                            //MPI rank and size
  char           filearg[PETSC_MAX_PATH_LEN];          //input file name
  char           *dir,*indexname,*tmp,**csrnames;      
  int            nfiles;                               //number of files
  PetscInt       **metadata;                           //metadata read from index.txt
  PetscInt       local_firstfile,local_lastfile;       //local file indices
  PetscInt       global_nrow,global_ncol,global_nnz;   //global index info
  PetscInt       local_nrow,local_nnz,local_row0;      //local  index info
  PetscInt       *local_indptr,*local_jcol;            //index arrays for local CSR
  PetscScalar    *local_data,*local_wts;               //data for local CSR and weights
  PetscBool      flg;
  PetscErrorCode ierr;
  IS             indices,indptr;
  Vec            data,weights,lambda;
  //Vec            data,weights,lambda,tforms0,tforms1,Lm0,Lm1,x0,x1,err0,err1;
  //const PetscInt *i,*j;
  //PetscScalar    *a,s0,s1;
  Mat              A,W,K,L;
  //Mat            A,W,K,L;
  //PetscReal      norm0,norm1,tmp0,tmp1;
  //PetscLogDouble t0,t1,t2,tall0,tall1;
  PetscLogDouble tall0,tall1;
  FILE *fp;
  int i,j;

  /*  Command line handling and setup  */
  PetscTime(&tall0);
  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetString(NULL,NULL,"-f",filearg,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  indexname = strdup(filearg);
  dir = strdup(dirname(filearg));
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

  /*  count the numberof hdf5 CSR files  */
  ierr = CountFiles(PETSC_COMM_WORLD,indexname,&nfiles);CHKERRQ(ierr);
  /*  allocate for file names and metadata  */
  csrnames = malloc(nfiles*sizeof(char *));
  metadata = malloc(nfiles*sizeof(PetscInt *));
  for (i=0;i<nfiles;i++){
    csrnames[i] = malloc(PETSC_MAX_PATH_LEN*sizeof(char));
    metadata[i] = malloc(4*sizeof(PetscInt));
  }
  /*  read in the metadata  */
  ierr = ReadIndex(PETSC_COMM_WORLD,indexname,nfiles,csrnames,metadata);CHKERRQ(ierr);
  /*  what files will this rank read  */
  ierr = SetFiles(PETSC_COMM_WORLD,nfiles,&local_firstfile,&local_lastfile);CHKERRQ(ierr);
  /*  how many rows and nnz per worker  */
  GetGlobalLocalCounts(nfiles,metadata,local_firstfile,local_lastfile,&global_nrow,&global_ncol,&global_nnz,&local_nrow,&local_nnz,&local_row0);
  //printf("rank %d will handle files %d through %d of %d\n",rank,local_firstfile,local_lastfile,nfiles);

  /*  allocate space for local CSR arrays  */
  local_indptr = (PetscInt *)calloc(local_nrow+1,sizeof(PetscInt));
  local_jcol = (PetscInt *)calloc(local_nnz,sizeof(PetscInt));
  local_data = (PetscScalar *)calloc(local_nnz,sizeof(PetscScalar));
  local_wts = (PetscScalar *)calloc(local_nrow,sizeof(PetscScalar));
  /*  read in local hdf5 files and concatenate into CSR arrays  */
  ierr = ReadLocalCSR(PETSC_COMM_SELF,csrnames,local_firstfile,local_lastfile,local_indptr,local_jcol,local_data,local_wts);CHKERRQ(ierr);

 
//  //create the distributed matrix A
//  PetscInt k,vcnt,niptr,innz=0;
//  PetscInt *iptr;
//  PetscInt *jcol;
//  PetscScalar *a,*w;
//  //these will concat the multiple files per processor
//  PetscInt roff=0,roff2=0,zoff=0,poff=0,pstart;
//  for (i=local_firstfile;i<=local_lastfile;i++){
//    ierr = PetscViewerHDF5Open(PETSC_COMM_SELF,csrnames[i],FILE_MODE_READ,&viewer);CHKERRQ(ierr);
//
//    //indptr
//    ierr = ReadIndexSet(PETSC_COMM_SELF,viewer,(char *)"indptr",&indptr,&niptr);CHKERRQ(ierr);
//    iptr = malloc(niptr*sizeof(PetscInt));
//    ISGetIndices(indptr,&iptr);
//    printf("rank %d w[%ld] =  %ld\n",rank,roff,local_indptr[roff]);
//    printf("niptr %ld\n",niptr);
//    for(j=1;j<niptr;j++){
//      local_indptr[j+roff] = iptr[j]+poff;
//    }
//    poff = local_indptr[niptr-1+roff];
//    printf("j+roff %ld\n",j+roff);
//    printf("rank %d w[%ld] =  %ld\n",rank,niptr,local_indptr[niptr]);
//    roff += niptr-1;
//
//    //indices
//    ierr = ReadIndexSet(PETSC_COMM_SELF,viewer,(char *)"indices",&indices,&vcnt);CHKERRQ(ierr);
//    jcol = malloc(vcnt*sizeof(PetscInt));
//    ISGetIndices(indices,&jcol);
//    printf("vcnt %ld\n",vcnt);
//    for (j=0;j<vcnt;j++){
//      local_jcol[j+zoff] = jcol[j];
//    }
//
//    //data
//    ierr = ReadVec(PETSC_COMM_SELF,viewer,(char *)"data",&data,&vcnt);CHKERRQ(ierr);
//    a = malloc(vcnt*sizeof(PetscScalar));
//    VecGetArray(data,&a);
//    printf("vcnt %ld\n",vcnt);
//    for (j=0;j<vcnt;j++){
//      local_data[j+zoff] = a[j];
//    }
//    zoff+=vcnt;
// 
//    //weights
//    ierr = ReadVec(PETSC_COMM_SELF,viewer,(char *)"weights",&weights,&vcnt);CHKERRQ(ierr);
//    w = malloc(vcnt*sizeof(PetscScalar));
//    VecGetArray(weights,&w);
//    for (j=0;j<vcnt;j++){
//      local_wts[j+roff2] = w[j];
//    }
//    roff2 += niptr;
//
//    printf("vcnt %ld\n",vcnt);
//
//    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
//  }
//  for (i=0;i<4;i++){
//    printf("rank %d: local_indptr[%ld] %ld\n",rank,i,local_indptr[i]);
//  }
//  for (i=local_nrow-5;i<local_nrow+1;i++){
//    printf("rank %d: local_indptr[%ld] %ld\n",rank,i,local_indptr[i]);
//  }

  MatCreateMPIAIJWithArrays(PETSC_COMM_WORLD,local_nrow,PETSC_DECIDE,global_nrow,global_ncol,local_indptr,local_jcol,local_data,&A);
//  MPI_Barrier(PETSC_COMM_WORLD);
  ShowMatInfo(&A,"A");

  //A is in hand, proceed to K=AT*W*A+L

  /*  Create the W matrix  */
//  Vec global_weights;
//  VecCreate(PETSC_COMM_WORLD,&global_weights);
//  VecSetSizes(global_weights,local_nrow,global_nrow);
//  VecSetType(global_weights,VECMPI);
//  PetscInt *indx = malloc(local_nrow*sizeof(PetscInt));
//  for (i=0;i<local_nrow;i++){
//    indx[i] = local_row0+i;
//  }
//  VecSetValues(global_weights,local_nrow,indx,local_wts,INSERT_VALUES);
//  MPI_Barrier(PETSC_COMM_WORLD);
//  ierr = MatCreate(PETSC_COMM_WORLD,&W);CHKERRQ(ierr);
//  ierr = MatSetSizes(W,local_nrow,local_nrow,global_nrow,global_nrow);CHKERRQ(ierr);
//  ierr = MatSetType(W,MATMPIAIJ);CHKERRQ(ierr);
//  ierr = MatMPIAIJSetPreallocation(W,1,NULL,0,NULL);CHKERRQ(ierr);
//  ierr = MatDiagonalSet(W,global_weights,INSERT_VALUES);CHKERRQ(ierr);
//  if (rank==0){ierr = ShowMatInfo(&W,"W");CHKERRQ(ierr);}
//
//  /*  Start the K matrix  */
//  ierr = MatPtAP(W,A,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&K);CHKERRQ(ierr);
//  PetscInt rowN_local;
//  MatGetOwnershipRange(K,&local_row0,&rowN_local);
//  //ncol_local = rowN_local-local_row0;
//  rowN_local--;
//  printf("%d local nrow %ld to %ld\n",rank,local_row0,rowN_local);

  /*  Create the L matrix  */
//  Vec global_reg;
//  VecCreate(PETSC_COMM_WORLD,&global_reg);
//  VecSetSizes(global_reg,ncol_local,global_ncol);
//  VecSetType(global_reg,VECMPI);
//  sprintf(junkstr,"%s/regularization.h5",dir);
//  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,junkstr,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
//  ierr = ReadVec(PETSC_COMM_SELF,viewer,(char *)"lambda",&lambda,&junk);CHKERRQ(ierr);
//  PetscScalar *lam = malloc(global_ncol*sizeof(PetscScalar));
//  PetscScalar *lamloc = malloc(ncol_local*sizeof(PetscScalar));
//  VecGetArray(lambda,&lam);
//  indx = malloc(ncol_local*sizeof(PetscInt));
//  for (i=0;i<ncol_local;i++){
//    indx[i] = i+local_row0;
//    lamloc[i] = lam[indx[i]];
//  }
//  VecSetValues(global_reg,ncol_local,indx,lamloc,INSERT_VALUES);
//  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
//  VecAssemblyBegin(global_reg);
//  VecAssemblyEnd(global_reg);
//  ierr = MatCreate(PETSC_COMM_WORLD,&L);CHKERRQ(ierr);
//  ierr = MatSetSizes(L,ncol_local,ncol_local,global_ncol,global_ncol);CHKERRQ(ierr);
//  ierr = MatSetType(L,MATMPIAIJ);CHKERRQ(ierr);
//  ierr = MatMPIAIJSetPreallocation(L,1,NULL,0,NULL);CHKERRQ(ierr);
//  ierr = MatDiagonalSet(L,global_reg,INSERT_VALUES);CHKERRQ(ierr);
//  if (rank==0){ierr = ShowMatInfo(&L,"L");CHKERRQ(ierr);}
//  
//  //ierr = MatAXPY(K,(PetscScalar)1.0,L,SUBSET_NONZERO_PATTERN);CHKERRQ(ierr);
//  if (rank==0){ierr = ShowMatInfo(&K,"K");CHKERRQ(ierr);}


  //local indices

  
  //ISCreate(PETSC_COMM_WORLD,indptr);

  

//  /*  Read CSR matrix and weights  */
//  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,fname,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
//  ierr = ReadIndexSet(viewer,(char *)"indptr",&indptr,&nrows);CHKERRQ(ierr);
//  nrows-=1;
//  ierr = ReadIndexSet(viewer,(char *)"indices",&indices,&junk);CHKERRQ(ierr);
//  ierr = ReadVec(viewer,(char *)"data",&data,&junk);CHKERRQ(ierr);
//  ierr = ReadVec(viewer,(char *)"weights",&weights,&junk);CHKERRQ(ierr);
//  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
//
//  /* Read regularization file  */
//  sprintf(fname,"%s/regularization.h5",dname);
//  printf("%s\n",fname);
//  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,fname,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
//  ierr = ReadVec(viewer,(char *)"lambda",&lambda,&junk);CHKERRQ(ierr);
//  ierr = ReadVec(viewer,(char *)"transforms_0",&tforms0,&junk);CHKERRQ(ierr);
//  ierr = ReadVec(viewer,(char *)"transforms_1",&tforms1,&junk);CHKERRQ(ierr);
//  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
//
//  /*  Determine M and N  */
//  ierr = ISGetMinMax(indices,&junk,&ncols);CHKERRQ(ierr);
//  ncols+=1;
//  printf("matrix will be %ld x %ld\n",nrows,ncols);
//
//  /*  Create A matrix from the CSR  inputs  */
//  ierr = ISGetIndices(indptr,&i);CHKERRQ(ierr);
//  ierr = ISGetIndices(indices,&j);CHKERRQ(ierr);
//  ierr = VecGetArray(data,&a);CHKERRQ(ierr);
//  ierr = MatCreateSeqAIJWithArrays(PETSC_COMM_WORLD,nrows,ncols,(PetscInt *)i,(PetscInt *)j,a,&A);CHKERRQ(ierr);
//  ierr = ShowMatInfo(&A,"A");CHKERRQ(ierr);
//
//  /*  Create weights matrix  */
//  ierr = MatCreate(PETSC_COMM_WORLD,&W);CHKERRQ(ierr);
//  ierr = MatSetSizes(W,nrows,nrows,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
//  ierr = MatSetType(W,MATSEQAIJ);CHKERRQ(ierr);
//  ierr = MatSeqAIJSetPreallocation(W,(PetscInt)1,NULL);CHKERRQ(ierr);
//  ierr = MatDiagonalSet(W,weights,INSERT_VALUES);CHKERRQ(ierr);
//  ierr = ShowMatInfo(&W,"W");CHKERRQ(ierr);
//
//  /*  Create regularization matrix  */
//  ierr = MatCreate(PETSC_COMM_WORLD,&L);CHKERRQ(ierr);
//  ierr = MatSetSizes(L,ncols,ncols,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
//  ierr = MatSetType(L,MATSEQAIJ);CHKERRQ(ierr);
//  ierr = MatSeqAIJSetPreallocation(L,(PetscInt)1,NULL);CHKERRQ(ierr);
//  ierr = MatDiagonalSet(L,lambda,INSERT_VALUES);CHKERRQ(ierr);
//  ierr = ShowMatInfo(&L,"L");CHKERRQ(ierr);
//
//  /*  Create the K matrix  */
//  ierr = MatPtAP(W,A,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&K);CHKERRQ(ierr);
//  ierr = MatAXPY(K,(PetscScalar)1.0,L,SUBSET_NONZERO_PATTERN);CHKERRQ(ierr);
//  ierr = ShowMatInfo(&K,"K");CHKERRQ(ierr);
//
//  /*  Create Lm vectors  */
//  ierr = VecDuplicate(tforms0,&Lm0);CHKERRQ(ierr);
//  ierr = VecDuplicate(tforms0,&Lm1);CHKERRQ(ierr);
//  ierr = MatMult(L,tforms0,Lm0);CHKERRQ(ierr);
//  ierr = MatMult(L,tforms1,Lm1);CHKERRQ(ierr);
//
//  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//                Create the linear solver and set various options
//     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
//  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
//  ierr = KSPSetOperators(ksp,K,K);CHKERRQ(ierr);
//  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
//  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//                      Solve the linear system
//     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
//  ierr = VecDuplicate(tforms0,&x0);CHKERRQ(ierr);
//  ierr = VecDuplicate(tforms0,&x1);CHKERRQ(ierr);
//  PetscTime(&t0);
//  ierr = KSPSolve(ksp,Lm0,x0);CHKERRQ(ierr);
//  PetscTime(&t1);
//  ierr = KSPSolve(ksp,Lm1,x1);CHKERRQ(ierr);
//  PetscTime(&t2);
//  printf("1st solve: %0.1f sec\n",t1-t0);
//  printf("2nd solve: %0.1f sec\n",t2-t1);
//  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//                      Check solution and clean up
//     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
//  ierr = VecDuplicate(tforms0,&err0);CHKERRQ(ierr);
//  ierr = VecDuplicate(tforms0,&err1);CHKERRQ(ierr);
//  ierr = VecScale(Lm0,(PetscScalar)-1.0);CHKERRQ(ierr);
//  ierr = VecScale(Lm1,(PetscScalar)-1.0);CHKERRQ(ierr);
//  ierr = MatMultAdd(K,x0,Lm0,err0);CHKERRQ(ierr);                  //err0 = Kx0-Lm0
//  ierr = MatMultAdd(K,x1,Lm1,err1);CHKERRQ(ierr);                  //err1 = Kx1-Lm1
//  ierr = VecNorm(err0,NORM_2,&norm0);CHKERRQ(ierr);         //NORM_2 denotes sqrt(sum_i |x_i|^2)
//  ierr = VecNorm(err1,NORM_2,&norm1);CHKERRQ(ierr);         //NORM_2 denotes sqrt(sum_i |x_i|^2)
//  ierr = VecNorm(Lm0,NORM_2,&tmp0);CHKERRQ(ierr);         //NORM_2 denotes sqrt(sum_i |x_i|^2)
//  ierr = VecNorm(Lm1,NORM_2,&tmp1);CHKERRQ(ierr);         //NORM_2 denotes sqrt(sum_i |x_i|^2)
//  ierr = PetscPrintf(PETSC_COMM_WORLD,"Precision %0.3g\n",(double)sqrt(norm0*norm0+norm1*norm1)/sqrt(tmp0*tmp0+tmp1*tmp1));CHKERRQ(ierr);
//
//  ierr = VecDuplicate(weights,&err0);CHKERRQ(ierr);
//  ierr = VecDuplicate(weights,&err1);CHKERRQ(ierr);
//  ierr = MatMult(A,x0,err0);CHKERRQ(ierr);                  //err0 = Ax0
//  ierr = MatMult(A,x1,err1);CHKERRQ(ierr);                  //err1 = Ax1
//  ierr = VecNorm(err0,NORM_2,&norm0);CHKERRQ(ierr);         //NORM_2 denotes sqrt(sum_i |x_i|^2)
//  ierr = VecNorm(err1,NORM_2,&norm1);CHKERRQ(ierr);         //NORM_2 denotes sqrt(sum_i |x_i|^2)
//  ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error %0.3g\n",(double)sqrt(norm0*norm0+norm1*norm1));CHKERRQ(ierr);
//
//  //calculate the mean and standard deviation
//  ierr = VecAbs(err0);CHKERRQ(ierr);
//  ierr = VecAbs(err1);CHKERRQ(ierr);
//  ierr = VecSum(err0,&s0);CHKERRQ(ierr);
//  ierr = VecSum(err1,&s1);CHKERRQ(ierr);
//  printf("%f %f\n",s0,s1);
//  tmp0 = (s0+s1)/(2*nrows); //mean
//  ierr = VecShift(err0,-1.0*tmp0);CHKERRQ(ierr);
//  ierr = VecShift(err1,-1.0*tmp0);CHKERRQ(ierr);
//  ierr = VecNorm(err0,NORM_2,&s0);CHKERRQ(ierr);
//  ierr = VecNorm(err1,NORM_2,&s1);CHKERRQ(ierr);
//  tmp1 = sqrt((s0*s0+s1*s1)/(2*nrows));
//  printf("mean(|Ax|) +/- std(|Ax|) : %0.1f +/- %0.1f\n",tmp0,tmp1);
//
//  //cleanup
//  VecDestroy(&data);
//  VecDestroy(&weights);
//  VecDestroy(&lambda);
//  VecDestroy(&tforms0);
//  VecDestroy(&tforms1);
//  VecDestroy(&Lm0);
//  VecDestroy(&Lm1);
//  VecDestroy(&x0);
//  VecDestroy(&x1);
//  VecDestroy(&err0);
//  VecDestroy(&err1);
//  ISDestroy(&indices);
//  ISDestroy(&indptr);
  MatDestroy(&A);
//  MatDestroy(&W);
//  MatDestroy(&K);
//  MatDestroy(&L);
//  KSPDestroy(&ksp);
//  PetscFree(a);
//  PetscFree(i);
//  PetscFree(j);

  free(dir); 
  free(indexname);
  for (i=0;i<nfiles;i++){
    free(csrnames[i]);
    free(metadata[i]);
  }
  free(csrnames);
  free(metadata);
  free(local_indptr);
  free(local_jcol);
  free(local_data);
  free(local_wts);

  PetscTime(&tall1);
  printf("rank %d total time: %0.1f sec\n",rank,tall1-tall0);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return ierr;
}

