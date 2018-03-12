.. image:: https://travis-ci.org/AllenInstitute/EM_aligner_python.svg?branch=master
   :target: https://travis-ci.org/AllenInstitute/EM_aligner_python
   :alt: Build Status
.. image:: https://codecov.io/gh/AllenInstitute/EM_aligner_python/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/AllenInstitute/EM_aligner_python
  

EM_aligner_python
#################

Alignment of EM datasets. Builds sparse matrices from tilespecs and pointmatch collections and solves for new transforms using constrained linear least squares.

Level of support
################
We are planning on occasional updating this tool with no fixed schedule. Community involvement is encouraged through both issues and pull requests.

setup
#####
1. clone and cd into the repo directory
2. modify sourceme.sh to point RENDER_JAVA_HOME to a directory where bin/java can be found
3. `source sourceme.sh`

alternately, ignore sourceme.sh and change the client_scripts in the input jsons to point to your own version of the render scripts

montage
#######
align the images of each tissue section to each other. sections are treated independently.
::
    for z in zlist:
        assemble from montage_matches(z)
        solve with scipy.sparse
        ingest
command line from within EMaligner directory:

::
    python -m EMaligner.EMaligner --input_json /path/to/montage_test.json

rough alignment
###############
apply a 3D alignment to downsampled montage results. Each montaged section is treated as 1 tile. The tiles are aligned with a rigid transformation. The first tile is fixed to prevent drastic scaling.

command line:
::
    python -m EMaligner.EMaligner --input_json example_jsons/rough_test.json

3D alignment
#############
all tiles are treated separately,with affine transformations.
## small 3D
run 3D solve locally, will be RAM limited
::
    for z in zlist:
        add to assembly from montage_matches(z), cross_matches(z,z+1),...,cross_matches(z,z+depth)
    solve with scipy.sparse
    ingest
command line:
::
    python -m EMaligner.EMaligner --input_json example_jsons/small3D_test.json

large 3D
########
intended to assemble A locally into multiple files for input to a distributed solver
::
    for z in zlist:
        add to assembly from montage_matches(z), cross_matches(z,z+1),...,cross_matches(z,z+depth)
        if assembly>big:
            write_new_file
            assembly = 0
    solve elsewhere with PETSc
    ingest
command line:
::
    python not_ready_yet.py --input_json not_ready_yet.json
    tar cvzf the_files
    scp the_files.tar.gz user@big_machine:
    user@big_machine$ tar -xvzf the_files.tar.gz
    user@big_machine$ solve the_files
    scp user@big_machine:the_solution ./
    python ingest.py the_solution

