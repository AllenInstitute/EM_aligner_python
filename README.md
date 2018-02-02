# EM_aligner_python

## Level of support
We are planning on occasional updating this tool with no fixed schedule. Community involvement is encouraged through both issues and pull requests.

## montage
```
for z in zlist:
    assemble from montage_matches(z)
    solve with scipy.sparse
ingest
```
`python assemble_matrix.py --input_json montage_test.json`
## rough alignment
need to incorporate rigid transformation
## small 3D
run 3D solve locally, will be RAM limited
```
for z in zlist:
    add to assembly from montage_matches(z), cross_matches(z,z+1),...,cross_matches(z,z+depth)
solve with scipy.sparse
ingest
```
`python assemble_matrix.py --input_json small3D_test.json`
## large 3D
intended to assemble A locally into multiple files for input to a distributed solver
```
for z in zlist:
    add to assembly from montage_matches(z), cross_matches(z,z+1),...,cross_matches(z,z+depth)
    if assembly>big:
        write_new_file
        assembly = 0
solve elsewhere with PETSc
ingest
```
```
python not_ready_yet.py --input_json not_ready_yet.json
tar cvzf the_files
scp the_files.tar.gz user@big_machine:
user@big_machine$ tar -xvzf the_files.tar.gz
user@big_machine$ solve the_files
scp user@big_machine:the_solution ./
python ingest.py the_solution
```
