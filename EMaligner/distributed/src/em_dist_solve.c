/** @file em_dist_solve.c
  * @brief Main method for distributed solve of Kx=Lm
  *
  * Reads A from file, reads regularizations and constraints, computes K, solves for x.
*/
static char help[] = "Testing hdf5 I/O\n\n";

#include <sys/resource.h>
#include <stdio.h>
#include <petsctime.h>
#include "ema.h"
//xxx

/** @brief em_dist_solve main
  *
*/
int main(int argc,char **args)
{
  KSP            ksp;                                  //linear solver context
  PetscMPIInt    rank,size;                            //MPI rank and size
  char           fileinarg[PETSC_MAX_PATH_LEN];          //input file name
  char           sln_output[PETSC_MAX_PATH_LEN];          //input file name
  char           *dir,*sln_input,**csrnames;           //various strings
  int            nfiles;                               //number of files
  PetscInt       **metadata;                           //metadata read from index.txt
  PetscInt       local_firstfile,local_lastfile;       //local file indices
  PetscInt       global_nrow,global_ncol,global_nnz;   //global index info
  PetscInt       local_nrow,local_nnz,local_row0;      //local  index info
  PetscInt       local_rowN;
  PetscInt       *local_indptr,*local_jcol;            //index arrays for local CSR
  PetscScalar    *local_data,*local_weights;           //data for local CSR and weights
  PetscBool      flg,trunc;                            //boolean used in checking command line
  PetscErrorCode ierr;                                 //error code that gets passed around.
  PetscLogDouble tall0,tall1;                          //timers
  int i;
  Mat              A,W,K,L,Kadj,Kper;                  //K and the matrices that build it
  Vec              rhs[2],Lm[2],x[2];                  //vectors associated with the solve(s)
  PetscLogDouble   t0,t1,current_mem;                  //some timers
  PetscReal        norm[2],norm2[2],gig=1073741824.;
  PetscLogStage    stage;
  MatPartitioning part;
  int mpisupp;

  /*  Command line handling and setup  */
  MPI_Init_thread(0,0,MPI_THREAD_MULTIPLE,&mpisupp);

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetString(NULL,NULL,"-input",fileinarg,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-output",sln_output,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-truncate",&trunc);CHKERRQ(ierr);

  PetscTime(&tall0);
  sln_input = strdup(fileinarg);
  dir = strdup(dirname(fileinarg));
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

  PetscLogStageRegister("Distributed Read", &stage);
  PetscLogStagePush(stage);
  /*  count the numberof hdf5 CSR files  */
  ierr = CountFiles(PETSC_COMM_WORLD,sln_input,&nfiles);CHKERRQ(ierr);
  printf("rank %d: nfiles %d\n",rank,nfiles);
  /*  allocate for file names and metadata  */
  csrnames = (char **)malloc(nfiles*sizeof(char *));
  metadata = (PetscInt **)malloc(nfiles*sizeof(PetscInt *));
  for (i=0;i<nfiles;i++){
    csrnames[i] = (char *)malloc(PETSC_MAX_PATH_LEN*sizeof(char));
    metadata[i] = (PetscInt *)malloc(4*sizeof(PetscInt));
  }
  /*  read in the metadata  */
  ierr = ReadMetadata(PETSC_COMM_WORLD,sln_input,nfiles,csrnames,metadata);CHKERRQ(ierr);
  //for (i=0;i<nfiles;i++){
  //  printf("%si %ld %ld %ld %ld\n",csrnames[i],metadata[i][0],metadata[i][1],metadata[i][2],metadata[i][3]);
  //}
  /*  what files will this rank read  */
  ierr = SetFiles(PETSC_COMM_WORLD,nfiles,&local_firstfile,&local_lastfile);CHKERRQ(ierr);
  /*  how many rows and nnz per worker  */
  GetGlobalLocalCounts(nfiles,metadata,local_firstfile,local_lastfile,&global_nrow,&global_ncol,&global_nnz,&local_nrow,&local_nnz,&local_row0);

  if (rank==0){
    printf("input file: %s\n",sln_input);
    printf("%d ranks will handle %d files\n",size,nfiles);
  }

  /*  allocate space for local CSR arrays  */
  local_indptr = (PetscInt *)calloc(local_nrow+1,sizeof(PetscInt));
  local_jcol = (PetscInt *)calloc(local_nnz,sizeof(PetscInt));
  local_data = (PetscScalar *)calloc(local_nnz,sizeof(PetscScalar));
  local_weights = (PetscScalar *)calloc(local_nrow,sizeof(PetscScalar));
  /*  read in local hdf5 files and concatenate into CSR arrays  */
  ierr = ReadLocalCSR(PETSC_COMM_SELF,csrnames,local_firstfile,local_lastfile,local_indptr,local_jcol,local_data,local_weights);CHKERRQ(ierr);
  /*  Create distributed A!  */
  MatCreateMPIAIJWithArrays(PETSC_COMM_WORLD,local_nrow,PETSC_DECIDE,global_nrow,global_ncol,local_indptr,local_jcol,local_data,&A);
  free(local_indptr);
  free(local_jcol);
  free(local_data);
  if (rank==0){
    printf("A matrix created\n");
  }
  PetscLogStagePop();

  PetscLogStageRegister("Create K", &stage);
  PetscLogStagePush(stage);

  /*  Create the W matrix  */
  ierr = CreateW(PETSC_COMM_WORLD,local_weights,local_nrow,local_row0,global_nrow,&W);
  free(local_weights);

  /*  Start the K matrix with K = AT*W*A */
  ierr = MatPtAP(W,A,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&K);CHKERRQ(ierr);
  //MatDestroy(&A);
  MatDestroy(&W);

  //find out how the rows are distributed
  MatGetOwnershipRange(K,&local_row0,&local_rowN);
  MatGetSize(K,&global_nrow,NULL);
  local_nrow = local_rowN-local_row0;

  //read in the regularization
  ierr = CreateL(PETSC_COMM_WORLD,dir,local_nrow,global_nrow,trunc,&L);
  if (rank==0){
    printf("L created\n");
  }
  //K = K+L
  MatSetOption(K, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
  ierr = MatAXPY(K,(PetscScalar)1.0,L,SUBSET_NONZERO_PATTERN);CHKERRQ(ierr);
  MatSetOption(K,MAT_SYMMETRIC,PETSC_TRUE);

  if (rank==0){
    printf("K matrix created\n");
  }
  PetscLogStagePop();

  PetscLogStageRegister("Get RHS", &stage);
  PetscLogStagePush(stage);

  /*  Read in the RHS vector(s)  */
  PetscInt nrhs;
  ierr = CountRHS(PETSC_COMM_WORLD,dir,&nrhs);CHKERRQ(ierr);
  ierr = ReadRHS(PETSC_COMM_WORLD,dir,local_nrow,global_nrow,nrhs,trunc,rhs);CHKERRQ(ierr);

  /*  Create Lm vectors  */
  for (i=0;i<nrhs;i++){
    ierr = VecDuplicate(rhs[i],&Lm[i]);CHKERRQ(ierr);
    ierr = MatMult(L,rhs[i],Lm[i]);CHKERRQ(ierr);
  }
  MatDestroy(&L);
  if (rank==0){
    printf("Lm(s) created\n");
  }

  PetscLogStagePop();

  //MatConvert(K,MATMPIBAIJ,MAT_INPLACE_MATRIX,&K);
//  PetscLogStageRegister("Reorder", &stage);
//  PetscLogStagePush(stage);
//  PC pc;
//  MatSolverPackage stype="xxxxxxxxxxxxxxxxx";
//  KSPGetPC(ksp,&pc);
//  PCFactorSetMatSolverPackage(pc,MATSOLVERPASTIX);
//  PCFactorGetMatSolverPackage(pc,stype);
//  printf("matsolvertype for factorization: %s\n",stype);
//
////  ierr = MatConvert(K,MATMPIADJ,MAT_INITIAL_MATRIX,&Kadj);CHKERRQ(ierr);
////  ierr = MatPartitioningCreate(PETSC_COMM_WORLD,&part);CHKERRQ(ierr);
////  ierr = MatPartitioningSetAdjacency(part,Kadj);CHKERRQ(ierr);
////  ierr = MatPartitioningSetType(part,MATPARTITIONINGPTSCOTCH);CHKERRQ(ierr);
////  IS is,isg,isrows;
////  ierr = MatPartitioningApply(part,&is);CHKERRQ(ierr);
////  MatPartitioningDestroy(&part);
////  MatDestroy(&Kadj);
////  ISPartitioningToNumbering(is,&isg);
////  ISBuildTwoSided(is,NULL,&isrows);
////  MatCreateSubMatrix(K,isrows,isrows,MAT_INITIAL_MATRIX,&Kper);
//  //ISView(isg,PETSC_VIEWER_STDOUT_WORLD);
//  //ISDestroy(&is);
//  PetscLogStagePop();

  PetscLogStageRegister("Solve", &stage);
  PetscLogStagePush(stage);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,K,K);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  char xname[20];
  PetscViewer viewer;
  if (rank==0){
      ierr = CopyDataSetstoSolutionOut(PETSC_COMM_SELF,sln_input,sln_output); CHKERRQ(ierr);
  }
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,sln_output,FILE_MODE_APPEND,&viewer);CHKERRQ(ierr);
  for (i=0;i<nrhs;i++){
    PetscTime(&t0);
    ierr = VecDuplicate(rhs[i],&x[i]);CHKERRQ(ierr);
    ierr = KSPSolve(ksp,Lm[i],x[i]);CHKERRQ(ierr);
    PetscTime(&t1);
    if (rank==0){
      printf("solve %d: %0.1f sec\n",i,t1-t0);
    }
    sprintf(xname,"transforms_%d",i);
    ierr = PetscObjectSetName((PetscObject)x[i],xname);CHKERRQ(ierr);
    ierr = VecView(x[i],viewer);CHKERRQ(ierr);
  }
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  PetscLogStagePop();

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Check solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscReal num,den;
  num=0;
  den=0;
  for (i=0;i<nrhs;i++){
    //ierr = VecDuplicate(rhs[i],&rhs[i]);CHKERRQ(ierr);
    //from here on, rhs is replaced by err
    ierr = VecScale(Lm[i],(PetscScalar)-1.0);CHKERRQ(ierr);
    ierr = MatMultAdd(K,x[i],Lm[i],rhs[i]);CHKERRQ(ierr);     //err0 = Kx0-Lm0
    ierr = VecNorm(rhs[i],NORM_2,&norm[i]);CHKERRQ(ierr);     //NORM_2 denotes sqrt(sum_i |x_i|^2)
    ierr = VecNorm(Lm[i],NORM_2,&norm2[i]);CHKERRQ(ierr);       //NORM_2 denotes sqrt(sum_i |x_i|^2)
    num += norm[i]*norm[i];
    den += norm2[i]*norm2[i];
  }
  if (rank==0){
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Precision %0.3g\n",(double)sqrt(num)/sqrt(den));CHKERRQ(ierr);
  }

  num=0;
  PetscInt mA,nA,c0,cn;
  ierr = MatGetSize(A,&mA,&nA);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&c0,&cn);CHKERRQ(ierr);
  for (i=0;i<nrhs;i++){
    ierr = VecCreate(PETSC_COMM_WORLD,&rhs[i]);CHKERRQ(ierr);
    ierr = VecSetType(rhs[i],VECMPI);CHKERRQ(ierr);
    ierr = VecSetSizes(rhs[i],cn-c0,mA);CHKERRQ(ierr);
    ierr = MatMult(A,x[i],rhs[i]);CHKERRQ(ierr);                  //err0 = Ax0
    ierr = VecNorm(rhs[i],NORM_2,&norm[i]);CHKERRQ(ierr);         //NORM_2 denotes sqrt(sum_i |x_i|^2)
    num += norm[i]*norm[i];
  }
  if (rank==0){
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error %0.1f\n",(double)sqrt(num));CHKERRQ(ierr);
  }

  //calculate the mean and standard deviation
  PetscReal s[2];
  PetscReal tmp0;
  for (i=0;i<nrhs;i++){
    ierr = VecAbs(rhs[i]);CHKERRQ(ierr);
    ierr = VecSum(rhs[i],&s[i]);CHKERRQ(ierr);
    tmp0 += s[i]/(nrhs*mA);
  }
  num=0;
  for (i=0;i<nrhs;i++){
    ierr = VecShift(rhs[i],-1.0*tmp0);CHKERRQ(ierr);
    ierr = VecNorm(rhs[i],NORM_2,&s[i]);CHKERRQ(ierr);
    num += s[i]*s[i]/(nrhs*mA);
  }
  if (rank==0){
    printf("mean(|Ax|) +/- std(|Ax|) : %0.1f +/- %0.1f\n",tmp0,sqrt(num));
  }

  //cleanup
  for (i=0;i<nrhs;i++){
    VecDestroy(&Lm[i]);
    VecDestroy(&x[i]);
    VecDestroy(&rhs[i]);
    //VecDestroy(&rhs[i]);
  }
  MatDestroy(&A);
  //MatDestroy(&W);
  MatDestroy(&K);
  //MatDestroy(&L);
  KSPDestroy(&ksp);
  free(dir);
  free(sln_input);
  for (i=0;i<nfiles;i++){
    free(csrnames[i]);
    free(metadata[i]);
  }
  free(csrnames);
  free(metadata);

  PetscTime(&tall1);
  if (rank==0){
    printf("rank %d total time: %0.1f sec\n",rank,tall1-tall0);
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
  return ierr;
}
