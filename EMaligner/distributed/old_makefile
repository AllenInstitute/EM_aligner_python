include ${PETSC_DIR}/install/lib/petsc/conf/variables
include ${PETSC_DIR}/install/lib/petsc/conf/rules

#"The Modular Component Architecture (MCA) is the backbone for much of Open MPI's functionality. It is a series of frameworks, components, and modules that are assembled at run-time to create an MPI implementation."
#https://www.open-mpi.org/faq/?category=tuning#mca-def
#mca_opts=--mca btl_tcp_if_include 10.128.124.0/99 --mca plm_rsh_no_tree_spawn 1
#mca_opts=
#mpi_opts=-N 1 -hostfile ./hosts -prefix ${PETSC_DIR}/install ${mca_opts}
#mpi_opts=-N 1 -hostfile ./hosts ${mca_opts}
#mpi_opts= -ppn 4 -f ./hosts 
mpi_opts= -f ./hosts 
#mpi_opts_one=-N 1 ${mca_opts}
solver_choice=-ksp_type preonly -pc_type lu -pc_factor_mat_solver_package pastix -mat_pastix_verbose 1 -mat_pastix_threadnbr 20
#solver_choice=-ksp_type preonly -pc_type lu -pc_factor_mat_solver_package superlu_dist -mat_superlu_dist_statprint# -mat_superlu_dist_colperm NATURAL
petsc_opts=-ksp_view -log_view

one_file_one_node: one_file_one_node.o  chkopts
	-${CLINKER} -o one_file_one_node one_file_one_node.o  ${PETSC_KSP_LIB}
	${RM} one_file_one_node.o

em_dist_solve: ema.o em_dist_solve.o chkopts
	-${CLINKER} -o em_dist_solve_${PETSC_ARCH} ema.o em_dist_solve.o  ${PETSC_KSP_LIB}
	${RM} em_dist_solve.o ema.o

run_one_file_one_node:
	${MPIEXEC} ${mpi_opts_one} ./one_file_one_node -f /allen/programs/celltypes/workgroups/em-connectomics/danielk/tmp_output_one/1020_1020.h5 ${solver_choice} ${petsc_opts}

run_em_dist_solve:
	${MPIEXEC} ${mpi_opts} ./em_dist_solve -f /allen/programs/celltypes/workgroups/em-connectomics/danielk/tmp_output/index.txt ${solver_choice} ${petsc_opts} 

run_505:
	${MPIEXEC} ${mpi_opts} ./em_dist_solve -f /allen/programs/celltypes/workgroups/em-connectomics/danielk/reflections_505_CSR/index.txt ${solver_choice} ${petsc_opts} 

run_100:
	${MPIEXEC} ${mpi_opts} ./em_dist_solve -f /allen/programs/celltypes/workgroups/em-connectomics/danielk/reflections_505_CSR/index100.txt -truncate ${solver_choice} ${petsc_opts} 

run_50:
	${MPIEXEC} ${mpi_opts} ./em_dist_solve -f /allen/programs/celltypes/workgroups/em-connectomics/danielk/reflections_505_CSR/index50.txt -truncate ${solver_choice} ${petsc_opts} 

run_valgrind:
	${MPIEXEC} ${mpi_opts} valgrind --leak-check=yes --suppressions=./valg.supp ./em_dist_solve -f /allen/programs/celltypes/workgroups/em-connectomics/danielk/tmp_output/index.txt ${solver_choice} ${petsc_opts}
	#mpiexec -N 1 ./one_file_one_node -f ../tmp_output/1020_1021.h5 ${solver_choice} ${petsc_opts}

