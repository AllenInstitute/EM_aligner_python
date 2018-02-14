/** @file em_dist_solve.c
  * @brief Main method for distributed solve of Kx=Lm
  *
  * Reads A from file, reads regularizations and constraints, computes K, solves for x.
*/
static char help[] = "Testing hdf5 I/O\n\n";

#include <stdio.h>
#include <petsctime.h>
#include "ema.h"

/** @brief em_dist_solve main
  * 
*/
int main(int argc,char **args)
{
  KSP            ksp;                                  //linear solver context
  PetscMPIInt    rank,size;                            //MPI rank and size
  char           filearg[PETSC_MAX_PATH_LEN];          //input file name
  char           *dir,*indexname,**csrnames;      //various strings
  int            nfiles;                               //number of files
  PetscInt       **metadata;                           //metadata read from index.txt
  PetscInt       local_firstfile,local_lastfile;       //local file indices
  PetscInt       global_nrow,global_ncol,global_nnz;   //global index info
  PetscInt       local_nrow,local_nnz,local_row0;      //local  index info
  PetscInt       local_rowN;
  PetscInt       *local_indptr,*local_jcol;            //index arrays for local CSR
  PetscScalar    *local_data,*local_weights;           //data for local CSR and weights
  PetscBool      flg;                                  //boolean used in checking command line
  PetscErrorCode ierr;                                 //error code that gets passed around.
  PetscLogDouble tall0,tall1;                          //timers
  int i;
  Mat              A,W,K,L;                            //K and the matrices that build it
  Vec              rhs[2],Lm[2],x[2];           //vectors associated with the solve(s)
  PetscLogDouble   t0,t1;                              //some timers
  PetscReal        norm[2],norm2[2];

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

  printf("rank %d will handle files %ld through %ld of %d\n",rank,local_firstfile,local_lastfile,nfiles);

  /*  allocate space for local CSR arrays  */
  local_indptr = (PetscInt *)calloc(local_nrow+1,sizeof(PetscInt));
  local_jcol = (PetscInt *)calloc(local_nnz,sizeof(PetscInt));
  local_data = (PetscScalar *)calloc(local_nnz,sizeof(PetscScalar));
  local_weights = (PetscScalar *)calloc(local_nrow,sizeof(PetscScalar));
  /*  read in local hdf5 files and concatenate into CSR arrays  */
  ierr = ReadLocalCSR(PETSC_COMM_SELF,csrnames,local_firstfile,local_lastfile,local_indptr,local_jcol,local_data,local_weights);CHKERRQ(ierr);
  /*  Create distributed A!  */
  MatCreateMPIAIJWithArrays(PETSC_COMM_WORLD,local_nrow,PETSC_DECIDE,global_nrow,global_ncol,local_indptr,local_jcol,local_data,&A);
  //ShowMatInfo(PETSC_COMM_WORLD,&A,"A");
  if (rank==0){
    printf("A matrix created\n");
  }


  /*  Create the W matrix  */
  ierr = CreateW(PETSC_COMM_WORLD,local_weights,local_nrow,local_row0,global_nrow,&W);
  //ierr = ShowMatInfo(PETSC_COMM_WORLD,&W,"W");CHKERRQ(ierr);

  /*  Start the K matrix with K = AT*W*A */
  ierr = MatPtAP(W,A,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&K);CHKERRQ(ierr);
  //find out how the rows are distributed
  MatGetOwnershipRange(K,&local_row0,&local_rowN);
  MatGetSize(K,&global_nrow,NULL);
  local_nrow = local_rowN-local_row0;
  //read in the regularization 
  ierr = CreateL(PETSC_COMM_WORLD,dir,local_nrow,global_nrow,&L);
  //K = K+L
  ierr = MatAXPY(K,(PetscScalar)1.0,L,SUBSET_NONZERO_PATTERN);CHKERRQ(ierr);
  //ierr = ShowMatInfo(PETSC_COMM_WORLD,&K,"K");CHKERRQ(ierr);
  if (rank==0){
    printf("K matrix created\n");
  }

  /*  Read in the RHS vector(s)  */
  PetscInt nrhs;
  ierr = CountRHS(PETSC_COMM_WORLD,dir,&nrhs);CHKERRQ(ierr);
  ierr = ReadRHS(PETSC_COMM_WORLD,dir,local_nrow,global_nrow,nrhs,rhs);CHKERRQ(ierr);

  /*  Create Lm vectors  */
  for (i=0;i<nrhs;i++){
    ierr = VecDuplicate(rhs[i],&Lm[i]);CHKERRQ(ierr);
    ierr = MatMult(L,rhs[i],Lm[i]);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,K,K);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  for (i=0;i<nrhs;i++){
    PetscTime(&t0);
    ierr = VecDuplicate(rhs[i],&x[i]);CHKERRQ(ierr);
    ierr = KSPSolve(ksp,Lm[i],x[i]);CHKERRQ(ierr);
    PetscTime(&t1);
    if (rank==0){
      printf("solve %d: %0.1f sec\n",i,t1-t0);
    }
  }

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
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Precision %0.3g\n",(double)sqrt(num)/sqrt(den));CHKERRQ(ierr);

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
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error %0.1f\n",(double)sqrt(num));CHKERRQ(ierr);

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
  printf("mean(|Ax|) +/- std(|Ax|) : %0.1f +/- %0.1f\n",tmp0,sqrt(num));

  //cleanup
  for (i=0;i<nrhs;i++){
    VecDestroy(&Lm[i]);
    VecDestroy(&x[i]);
    VecDestroy(&rhs[i]);
    //VecDestroy(&rhs[i]);
  }
  MatDestroy(&A);
  MatDestroy(&W);
  MatDestroy(&K);
  MatDestroy(&L);
  KSPDestroy(&ksp);
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
  free(local_weights);

  PetscTime(&tall1);
  printf("rank %d total time: %0.1f sec\n",rank,tall1-tall0);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return ierr;
}

