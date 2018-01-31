# match_filtering
## intent
point match generation for 3D EM data sets currently outputs, per tile pair:
```
nmax_montage=200
nmax_cross=500
```
the size of the assembled A matrix for affine transformations, at a depth of 2 sections is:
```
A.data.nbytes = n_z*ntilepairs_per_z*(n_montage+2*n_cross)*6*8
```
for n_z=2.5e3, ntilepairs_per_z=2e4

A.data.nbytes  = 2.7TB

and the CSR format means A would consume at least 2x that much memory. Let's call it 6TB.

If we limit to, for example
`nmax=50`
we can drop the memory usage by 8x

This reduction may not be necessary for large 3D problems on a big enough machine, but, would extend the solvable size for small 3D problems.

This code is meant to:
```
nmax = 50
read in original point match collection
for each tile pair
    extract the best nmax matches via RANSAC
    write a new tile pair to a new match collection
use new match collection for solve
```

