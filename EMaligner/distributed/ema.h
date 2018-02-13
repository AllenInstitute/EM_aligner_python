/** @file ema.h
 *  @brief Function prototypes used by EM_aligner's distributed solver
 *  @author Daniel Kapner
*/

#include <petscsys.h>
#include <petscksp.h>
#include <stdlib.h>

PetscErrorCode CountFiles(MPI_Comm COMM, char indexname[], int *nfiles);
/** 
 * @brief Rank 0 processor counts lines in index.txt and broadcasts count.
 * @param COMM The MPI communicator, probably PETSC_COMM_WORLD.
 * @param indexname The full path to index.txt, given as command-line argument with -f.
 * @param *nfiles The result of the count, returned to main().
*/

PetscErrorCode ReadIndex(MPI_Comm COMM, char indexname[], int nfiles, char *csrnames[], PetscInt **metadata);
/** 
 * @brief Rank 0 processor reads metadata and broadcasts
 * @param COMM The MPI communicator, probably PETSC_COMM_WORLD.
 * @param indexname The full path to index.txt, given as command-line argument with -f.
 * @param nfiles The number of lines in index.txt, read from previous CountFiles() call.
 * @param *csrnames An array of file names from index.txt, populated by this function.
 * @param *metadata An array of metadata from index.txt, populated by this function.
*/

PetscErrorCode SetFiles(MPI_Comm COMM, int nfiles, PetscInt *firstfile, PetscInt *lastfile);
/** 
 * @brief Split the list of files roughly evenly amongst all the workers.
 * @param COMM The MPI communicator, probably PETSC_COMM_WORLD.
 * @param nfiles The number of lines in index.txt, read from previous CountFiles() call.
 * @param *firstfile The index of the first file for this worker.
 * @param *lastfile The index of the first file for this worker.
*/

PetscErrorCode ReadVec(MPI_Comm COMM,PetscViewer viewer,char *varname,Vec *newvec,long *n);
/** 
 * @brief Read data from a PetscViewer into a Vec (scalar) object.
 * @param COMM The MPI communicator, PETSC_COMM_SELF or PETSC_COMM_WORLD.
 * @param viewer A PetscViewer instance.
 * @param *varname The new vector will have this name. For reading from hdf5 files, this name must match the dataset name in the file.
 * @param *newvec The new vector object.
 * @param *n The number of entries in the new object.
*/

PetscErrorCode ReadIndexSet(MPI_Comm COMM,PetscViewer viewer,char *varname,IS *newIS,long *n);
/** 
 * @brief Read data from a PetscViewer into an IS (Index Set, i.e. integer) object.
 * @param COMM The MPI communicator, PETSC_COMM_SELF or PETSC_COMM_WORLD.
 * @param viewer A PetscViewer instance.
 * @param *varname The new IS will have this name. For reading from hdf5 files, this name must match the dataset name in the file.
 * @param *newIS The new IS object.
 * @param *n The number of entries in the new object.
*/

PetscErrorCode ShowMatInfo(Mat *m,const char *mesg);
/** 
 * @brief Print to stdout MatInfo for a Mat object.
 * @param *m The matrix.
 * @param *mesg Name or some other string to prepend the output.
*/

void GetGlobalLocalCounts(int nfiles, PetscInt **metadata, int local_firstfile, int local_lastfile, PetscInt *global_nrow, PetscInt *global_ncol, PetscInt *global_nnz, PetscInt *local_nrow, PetscInt *local_nnz, PetscInt *local_row0);
/** 
 * @brief Use metadata to determine global and local sizes and indices.
 * @param nfiles The number of CSR.hdf5 files, from a previous call to CountFiles()
 * @param **metadata Metadata from index.txt from a previous call to ReadIndex()
 * @param local_firstfile, local_lastfile Indices in the list of files for the local worker.
 * @param global_nrow,global_ncol,global_nnz Read from the metadata, these describe A, the matrix to be read from the hdf5 files.
 * @param local_nrow,local_nnz,local_row0 Read from the metadata, these are concatentations of the metadata for all the owned files for one worker.
*/
PetscErrorCode ReadLocalCSR(MPI_Comm COMM, char *csrnames[], int local_firstfile, int local_lastfile, PetscInt *local_indptr, PetscInt *local_jcol, PetscScalar *local_data, PetscScalar *local_weights);
/** 
 * @brief Build local CSR block by sequentially reading in local hdf5 files.
 * @param COMM The MPI communicator, PETSC_COMM_SELF.
 * @param *csrnames[] The names of the CSR.hdf5 files
 * @param local_firstfile,local_lastfile Indices of which files are handled by which rank.
 * @param *local_indptr,*local_jcol, *local_data Holds the concatenated CSR arrays for this rank.
 * @param *local_weights Holds the concatenated weights for this rank.
*/
PetscErrorCode CreateW(MPI_Comm COMM,PetscScalar *local_weights,PetscInt local_nrow,PetscInt local_row0,PetscInt global_nrow,Mat *W);
PetscErrorCode CreateL(MPI_Comm COMM,char *dir,PetscInt local_nrow,PetscInt global_nrow,Mat *L);

