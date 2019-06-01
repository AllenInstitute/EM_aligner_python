ifeq ($(MAKECMDGOALS),allen)
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
endif

allen: src/em_dist_solve.o chkopts
	@echo ${CLINKER}
	${CLINKER} -o bin/em_dist_solve src/em_dist_solve.o  ${PETSC_KSP_LIB}
	${RM} src/em_dist_solve.o

cori:
	module load cray-petsc-64; \
        module load cray-hdf5-parallel; \
        cc -o ${PWD}/bin/em_solver_cori ${PWD}/src/em_dist_solve.c 
