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

class pointmatch(db_params):
    collection_type = argschema.fields.String(default='pointmatch',description="'stack' or 'pointmatch'")
class stack(db_params):
    collection_type = argschema.fields.String(default='stack',description="'stack' or 'pointmatch'")

class MySchema(argschema.ArgSchema):
    width = argschema.fields.Int(default=3840, description = 'width [pixels] of an image')
    height = argschema.fields.Int(default=3840, description = 'height [pixels]of an image')
    ntile_x = argschema.fields.Int(default=10, description = 'number of tiles along x axis')
    ntile_y = argschema.fields.Int(default=10, description = 'number of tiles along y axis')
    ntile_z = argschema.fields.Int(default=10, description = 'number of tiles along z axis (nsections)')
    overlap = argschema.fields.Float(default=240, description = 'fractional overlap of images')
    npts = argschema.fields.Int(default=50, description = 'number of points per tile pair')
    new_stack = argschema.fields.Nested(stack)
    new_pointmatch = argschema.fields.Nested(pointmatch)
    intermediate_name = argschema.fields.String(default='sections_moved',description="for visually checking section movement")
    raw_name = argschema.fields.String(default='raw_stack',description="name of the fake raw stack")

