# EM aligner distributed

For distributed solves across multiple compute nodes. This solver is built with the PETSc libraries

https://www.mcs.anl.gov/petsc/index.html

## Systems

This code has been run on NERSC's Cori and on the Allen Insitute cluster.

For Cori, Cray modules make simple work of compiling and running this code. See `makefile_cori` and `cori_example_script`.

For the Allen cluster, one way to run this is via Singularity containers. `Singularity.petsc` is a definition file for an image with compiled PETSc. This is a fairly lengthy process and should not change very often. The build of the image is manually triggered and maintained on Singularity Hub: https://singularity-hub.org/collections/2940. The solver code in this repository is then compiled in another container that builds from the PETSc image. This is `Singularity.petsc_solver`. For an example of how to run this compilation step, look in `.travis.yml` in this repository. For one example of how to use the built singularity image, see `integration_tests/test_hdf5.py`. For another example, see `allen_example_script.pbs`.

## Usage

```
em_solver_cori -input <input file> -output <output_file> <ksp options>
```
or
```
singularity run --bind <external>:<internal> em_distributed.simf -input <input_file> -output <output_file> <ksp options>
```

`ksp options` specify how PETSc should handle the system. For example whether to use a direct or iterative solver, what type of preconditioner to use, what external packages to invoke.

* direct solve with Pastix: `-ksp_type preonly -pc_type lu`

There are many PETSc options, and not all of them are necessarily installed in the Singularity image here.

## Documentation

https://em-aligner-python.readthedocs.io/en/latest/index.html

## Author

[Dan Kapner](https://github.com/djkapner) e-mail: danielk@alleninstitute.org
