# EM_aligner_python
## montage: 
```
for z in zlist:
    assemble from montage_matches(z)
    solve with scipy.sparse
ingest
```
`python assemble_matrix.py --input_json montage_test.json`
## small 3D:
intended to run locally, will be RAM limited
```
for z in zlist:
    add to assembly from montage_matches(z), cross_matches(z,z+1),...,cross_matches(z,z+depth)
solve with scipy.sparse
ingest
```
`python not_ready_yet.py --input_json not_ready_yet.json`
## large 3D:
intended to assemble A locally into multiple files for input for a distributed solver
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
scp the_files user@big_machine:
user@big_machine$ tar -xvzf the_files.tar.gz
user@big_machine$ solve the_files
scp user@big_machine:the_solution ./
python ingest.py the_solution
```
