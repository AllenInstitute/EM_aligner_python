#!/usr/bin/env python

import argschema

class db_params(argschema.ArgSchema):
    owner = argschema.fields.String(default='',description='owner') 
    project = argschema.fields.String(default='',description='project') 
    name = argschema.fields.String(default='',description='name')
    host = argschema.fields.String(default='em-131fs',description='render host')
    port = argschema.fields.Int(default=8080,description='render port')
    mongo_host = argschema.fields.String(default='em-131fs',description='mongodb host')
    mongo_port = argschema.fields.Int(default=27017,description='mongodb port')
    db_interface = argschema.fields.String(default='mongo')
    client_scripts = argschema.fields.String(default='/allen/programs/celltypes/workgroups/em-connectomics/gayathrim/nc-em2/Janelia_Pipeline/render_latest/render-ws-java-client/src/main/scripts',description='render bin path')

class output_options(argschema.ArgSchema):
    output_mode = argschema.fields.String(default='hdf5')
    output_dir = argschema.fields.String(default='/allen/programs/celltypes/workgroups/em-connectomics/danielk/solver_exchange/python/')
    chunks_per_file = argschema.fields.Int(default=5,description='how many sections with upward-looking cross section to write per .h5 file')

class matrix_assembly(argschema.ArgSchema):
    depth = argschema.fields.Int(default=2,description='depth in z for matrix assembly point matches')
    cross_pt_weight = argschema.fields.Float(default=1.0,description='weight of cross section point matches')
    montage_pt_weight = argschema.fields.Float(default=1.0,description='weight of montage point matches')

class regularization(argschema.ArgSchema):
    default_lambda = argschema.fields.Float(0.005,description='regularization factor')
    translation_lambda = argschema.fields.Float(0.005,description='regularization factor')

class pointmatch(db_params):
    collection_type = argschema.fields.String(default='pointmatch',description="'stack' or 'pointmatch'")
class stack(db_params):
    collection_type = argschema.fields.String(default='stack',description="'stack' or 'pointmatch'")

class MySchema(argschema.ArgSchema):
    first_section = argschema.fields.Int(default=1000, description = 'first section for matrix assembly')
    last_section = argschema.fields.Int(default=1000, description = 'last section for matrix assembly')
    input_stack = argschema.fields.Nested(stack)
    pointmatch = argschema.fields.Nested(pointmatch)
    output_options = argschema.fields.Nested(output_options)
    matrix_assembly = argschema.fields.Nested(matrix_assembly)
    regularization = argschema.fields.Nested(regularization)
    showtiming = argschema.fields.Int(default=1,description = 'have the routine showhow long each process takes')
    output_dir = argschema.fields.String(default='/allen/programs/celltypes/workgroups/em-connectomics/danielk/solver_exchange/python/',description='where to send logs and results')

