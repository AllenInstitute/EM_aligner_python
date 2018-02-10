static char help[] = "Testing hdf5 I/O\n\n";

#include <stdio.h>
#include <petscksp.h>
#include <petscviewerhdf5.h>
//#include <stdint.h>
#include <libgen.h>
//#include <stdlib.h>
#include <petsctime.h>
#include <petscsys.h>
//#include <string.h>

PetscErrorCode ReadVec(PetscViewer,char *,Vec *,long *);
PetscErrorCode ReadIndexSet(MPI_Comm,PetscViewer,char *,IS *,long *);
PetscErrorCode ShowMatInfo(Mat *,const char *);
PetscErrorCode ReadVec(PetscViewer viewer,char *varname,Vec *newvec,long *n){
  PetscErrorCode ierr;
  char           name[256];
  sprintf(name,"%s",varname);
  ierr = VecCreate(PETSC_COMM_WORLD,newvec);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)*newvec,name);
  ierr = VecLoad(*newvec,viewer);CHKERRQ(ierr);
  ierr = VecGetSize(*newvec,n);
  printf("read vector %s, size: %ld\n",varname,*n);
  return ierr;
}
PetscErrorCode ReadIndexSet(MPI_Comm COMM,PetscViewer viewer,char *varname,IS *newIS,long *n){
  PetscErrorCode ierr;
  char           name[256];
  sprintf(name,"%s",varname);
  ierr = ISCreate(COMM,newIS);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)*newIS,name);
  ierr = ISLoad(*newIS,viewer);CHKERRQ(ierr);
  ierr = ISGetSize(*newIS,n);
  printf("read vector %s, size: %ld\n",varname,*n);
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

int main(int argc,char **args)
{
  //KSP            ksp;      /* linear solver context */
  PetscViewer    viewer;              /* viewer */
  //PetscInt       nrows,ncols,junk;
  PetscMPIInt    rank,size;
  char           filearg[PETSC_MAX_PATH_LEN];     /* input file name */
  //char           indexname[PETSC_MAX_PATH_LEN];     /* input file name */
  //char           dir[PETSC_MAX_PATH_LEN];     /* input file name */
  char           *dir,*indexname,*tmp,**csrnames,**local_csrnames;
  char           junkstr[200],tmpname[100];
  int            nfiles=0,nfiles_local=0,local_fileind[2];
  PetscInt       **index_tab,rowspernode,itmp;
  PetscInt       local_start,local_end,**wrows;
  PetscBool      flg;
  PetscErrorCode ierr;
  IS             indices,indptr;
  //Vec            data,weights,lambda,tforms0,tforms1,Lm0,Lm1,x0,x1,err0,err1;
  //const PetscInt *i,*j;
  //PetscScalar    *a,s0,s1;
  //Mat            A,W,K,L;
  //PetscReal      norm0,norm1,tmp0,tmp1;
  //PetscLogDouble t0,t1,t2,tall0,tall1;
  PetscLogDouble tall0,tall1;
  FILE *fp;
  int i,j;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  PetscTime(&tall0);
  ierr = PetscOptionsGetString(NULL,NULL,"-f",filearg,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);

  indexname = strdup(filearg);
  tmp = dirname(filearg);//don't free() tmp
  dir = strdup(tmp);

  /*  Determine which worker will read which files  */
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  wrows = malloc(size*sizeof(PetscInt *));
  for (i=0;i<size;i++){
    wrows[i] = malloc(2*sizeof(PetscInt));
  }
  if (rank==0){
    /*  Read and parse index.txt  */
    PetscFOpen(PETSC_COMM_WORLD,indexname,"r",&fp);
    //count lines
    int ch;
    printf("nfiles: %d\n",nfiles);
    while(!feof(fp)){
      ch = fgetc(fp);
      if(ch == '\n'){
        nfiles++;
      }
    }
    //read in file names and rows for each file
    csrnames = malloc(nfiles*sizeof(char *));
    index_tab = malloc(nfiles*sizeof(PetscInt *));
    for (i=0;i<nfiles;i++){
      index_tab[i] = malloc(8*sizeof(PetscInt));
    }
    fseek(fp,0,SEEK_SET);
    for (i=0;i<nfiles;i++){
      fscanf(fp,"%*s %s ",tmpname);
      fscanf(fp,"%*s %*s %ld %*s %ld ",&index_tab[i][0],&index_tab[i][1]);
      fscanf(fp,"%*s %*s %ld %*s %ld ",&index_tab[i][2],&index_tab[i][3]);
      fscanf(fp,"%*s %*s %ld %*s %ld ",&index_tab[i][4],&index_tab[i][5]);
      fscanf(fp,"%*s %*s %ld %*s %ld\n",&index_tab[i][6],&index_tab[i][7]);
      printf("%s\n",tmpname);
      printf("%d\n",index_tab[i][0]);
      sprintf(junkstr,"%s/%s",dir,tmpname);
      csrnames[i] = malloc(PETSC_MAX_PATH_LEN*sizeof(char));
      sprintf(csrnames[i],"%s",junkstr);
    }
    PetscFClose(PETSC_COMM_WORLD,fp);
  }
  //share the number of files to all workers
  MPI_Bcast(&nfiles,1,MPI_INT,0,PETSC_COMM_WORLD);
  //allocate space in the workers
  if (rank!=0){
    index_tab = malloc(nfiles*sizeof(PetscInt *));
    csrnames = malloc(nfiles*sizeof(char *));
    for (i=0;i<nfiles;i++){
      index_tab[i] = malloc(8*sizeof(PetscInt));
      csrnames[i] = malloc(PETSC_MAX_PATH_LEN*sizeof(char));
    }
  }
  for (i=0;i<nfiles;i++){
    MPI_Bcast(csrnames[i],PETSC_MAX_PATH_LEN,MPI_CHAR,0,PETSC_COMM_WORLD);
    MPI_Bcast(index_tab[i],8,MPIU_INT,0,PETSC_COMM_WORLD);
  }
  //what files should I read?
  if (nfiles<=size){
    for (i=0;i<size;i++){
      if (rank==i){
        local_fileind[0]=i;
        if (i>nfiles-1){
          local_fileind[1]=i-1;
        }
        else{
          local_fileind[1]=i;
        }
      }
    }
  }
  else{
    float avg = nfiles/(float)size;
    int rem = nfiles%size;
    int last=0;
    for (i=0;i<size;i++){
      if (rank==i){
        local_fileind[0]=last;
      }
      if (rem-->0){
        last=last+ceil(avg);
      }
      else{
        last=last+floor(avg);
      }
      if (rank==i){
        local_fileind[1]=last-1;
      }
    }
  }
  MPI_Barrier(PETSC_COMM_WORLD);
  printf("rank %d will handle files %d through %d of %d\n",rank,local_fileind[0],local_fileind[1],nfiles);
  MPI_Barrier(PETSC_COMM_WORLD);

  //what's the global length of the indptr vector?
  //printf("rank %d: %ld of %ld\n",rank,loclen,globlen);

  PetscInt ijunk;
  PetscInt globlen,loclen;
  //local indptr
  //need something here to add 0 at start or correct lengths
  loclen=0;
  for (i=local_fileind[0];i<=local_fileind[1];i++){
    loclen+=index_tab[i][1];
  }
  PetscInt *iptr = malloc(loclen*sizeof(PetscInt));
  //local indices
  loclen=0;
  for (i=local_fileind[0];i<=local_fileind[1];i++){
    loclen+=index_tab[i][3];
  }
  PetscInt *jcol = malloc(loclen*sizeof(PetscInt));
  for (i=local_fileind[0];i<=local_fileind[1];i++){
    printf("namename %s\n",csrnames[i]);
    ierr = PetscViewerHDF5Open(PETSC_COMM_SELF,csrnames[i],FILE_MODE_READ,&viewer);CHKERRQ(ierr);
    //need something here to drop zeros for not the first one
    ierr = ReadIndexSet(PETSC_COMM_SELF,viewer,(char *)"indptr",&indptr,&ijunk);CHKERRQ(ierr);
    ISGetIndices(indptr,iptr+index_tab[i][0]-index_tab[local_fileind[0]][0]);
    ierr = ReadIndexSet(PETSC_COMM_SELF,viewer,(char *)"indices",&indices,&ijunk);CHKERRQ(ierr);
    ISGetIndices(indices,jcol+index_tab[i][2]-index_tab[local_fileind[0]][2]);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  //local indices

  //MPI_Barrier(PETSC_COMM_WORLD);
  
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
//  MatDestroy(&A);
//  MatDestroy(&W);
//  MatDestroy(&K);
//  MatDestroy(&L);
//  KSPDestroy(&ksp);
//  PetscFree(a);
//  PetscFree(i);
//  PetscFree(j);

  free(dir); 
  free(indexname);
//  for (i=0;i<nfiles;i++){
//    free(csrnames[i]);
//    free(index_tab[i]);
//  }
//  free(csrnames);
//  free(index_tab);
  //for (i=0;i<size;i++){
  //  free(wrows[i]);
 // }
  //free(wrows);

  PetscTime(&tall1);
  printf("rank %d total time: %0.1f sec\n",rank,tall1-tall0);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return ierr;
}

