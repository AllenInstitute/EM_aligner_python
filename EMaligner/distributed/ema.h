/** @file ema.h
 *  @brief Function prototypes used by EM_aligner's distributed solver
 *  @author Daniel Kapner
*/

#include <petscsys.h>
#include <petscksp.h>
#include <stdlib.h>
#include <petscviewerhdf5.h>
#include <libgen.h>

PetscErrorCode CountFiles(MPI_Comm COMM, char indexname[], int *nfiles);

PetscErrorCode ReadIndex(MPI_Comm COMM, char indexname[], int nfiles, char *csrnames[], PetscInt **metadata);

PetscErrorCode SetFiles(MPI_Comm COMM, int nfiles, PetscInt *firstfile, PetscInt *lastfile);

PetscErrorCode ReadVec(MPI_Comm COMM,PetscViewer viewer,char *varname,Vec *newvec,long *n);

PetscErrorCode ReadVecWithSizes(MPI_Comm COMM,PetscViewer viewer,char *varname,Vec *newvec,long *n,PetscInt nlocal,PetscInt nglobal);

PetscErrorCode ReadIndexSet(MPI_Comm COMM,PetscViewer viewer,char *varname,IS *newIS,long *n);

PetscErrorCode ShowMatInfo(MPI_Comm COMM,Mat *m,const char *mesg);

void GetGlobalLocalCounts(int nfiles, PetscInt **metadata, int local_firstfile, int local_lastfile, PetscInt *global_nrow, PetscInt *global_ncol, PetscInt *global_nnz, PetscInt *local_nrow, PetscInt *local_nnz, PetscInt *local_row0);

PetscErrorCode ReadLocalCSR(MPI_Comm COMM, char *csrnames[], int local_firstfile, int local_lastfile, PetscInt *local_indptr, PetscInt *local_jcol, PetscScalar *local_data, PetscScalar *local_weights);

PetscErrorCode CreateW(MPI_Comm COMM,PetscScalar *local_weights,PetscInt local_nrow,PetscInt local_row0,PetscInt global_nrow,Mat *W);

PetscErrorCode CreateL(MPI_Comm COMM,char *dir,PetscInt local_nrow,PetscInt global_nrow,Mat *L);

PetscErrorCode CountRHS(MPI_Comm COMM,char *dir,PetscInt *nRHS);

PetscErrorCode ReadRHS(MPI_Comm COMM,char *dir,PetscInt local_nrow,PetscInt global_nrow,PetscInt nrhs,Vec rhs[]);

