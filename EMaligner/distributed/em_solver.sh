#/bin/bash

#PBS -N em_solver
#PBS -q emconnectome
#PBS -l nodes=10 -n
#PBS -l walltime=01:00:00
#PBS -r n
#PBS -k oe
#PBS -o /allen/programs/celltypes/workgroups/em-connectomics/danielk/EM_aligner_python/EMaligner/distributed/log/
#PBS -m a
#PBS -M danielk@alleninstitute.org

#export PETSC_DIR=/allen/programs/celltypes/workgroups/em-connectomics/danielk/petsc3/petsc-3.8.3/
#export DATAFILESPATH=/allen/programs/celltypes/workgroups/em-connectomics/danielk/petsc3/datafiles/
#export PETSC_DIR=/allen/programs/celltypes/workgroups/em-connectomics/danielk/petsc4/petsc-3.8.3/
#export PETSC_DIR=/allen/programs/celltypes/workgroups/em-connectomics/danielk/hpc_compiled//PETSC/petsc-3.8.3
#export PETSC_ARCH=pastix_debug
#export DATAFILESPATH=/allen/programs/celltypes/workgroups/em-connectomics/danielk/petsc5/datafiles/
#export PATH=${PETSC_DIR}install/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
#export LD_LIBRARY_PATH=${PETSC_DIR}install/lib

module load mpi/mpich-3.2-x86_64 
module load cmake/3.7.0  
module load gcc/4.4.7-centos7
export MYHPC=/allen/programs/celltypes/workgroups/em-connectomics/danielk/hpc_compiled/  
#PATH=$PATH:$MYHPC/install/bin
export PETSC_DIR=$MYHPC/PETSC/petsc-3.8.3
export PETSC_ARCH=pastix_debug
export PATH=${PETSC_DIR}install/bin:$PATH
export LD_LIBRARY_PATH=${PETSC_DIR}install/lib

cd /allen/programs/celltypes/workgroups/em-connectomics/danielk/EM_aligner_python/EMaligner/distributed/
#mpiexec ./em_dist_solve -f /allen/programs/celltypes/workgroups/em-connectomics/danielk/reflections_505_CSR/index50.txt -truncate -ksp_type preonly -pc_type lu -pc_factor_mat_solver_package pastix -mat_pastix_verbose 1 -mat_pastix_threadnbr 20 -ksp_view -log_view
mpiexec ./em_dist_solve_${PETSC_ARCH} -f /allen/programs/celltypes/workgroups/em-connectomics/danielk/reflections_505_CSR/index20.txt -truncate -ksp_type preonly -pc_type lu -pc_factor_mat_solver_package pastix -mat_pastix_verbose 1 -mat_pastix_threadnbr 16 -log_view
#mpiexec ./em_dist_solve_${PETSC_ARCH} -f /allen/programs/celltypes/workgroups/em-connectomics/danielk/reflections_505_CSR/index50.txt -truncate -ksp_type preonly -pc_type lu -pc_factor_mat_solver_package superlu_dist -ksp_view -log_view
