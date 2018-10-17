#!/usr/bin/env python

from argschema import ArgSchema
from argschema.fields import String, Int, Boolean, Nested, Float
import marshmallow as mm
from marshmallow import post_load, ValidationError

class db_params(ArgSchema):
    owner = String(
        default='',
        required=False,
        description='owner') 
    project = String(
        default='',
        required=False,
        description='project') 
    name = String(
        required=True,
        description='stack name')
    host = String(
        default=None,
        required=False,
        description='render host')
    port = Int(
        default=8080,
        required=False,
        description='render port')
    mongo_host = String(
        default='em-131fs',
        required=False,
        description='mongodb host')
    mongo_port = Int(
        default=27017,
        required=False,
        description='mongodb port')
    mongo_userName = String(
        default='',
        required=False,
        description='mongo user name')
    mongo_authenticationDatabase = String(
        default='',
        required=False,
        description='mongo admin db')
    mongo_password = String(
        default='',
        required=False,
        description='mongo pwd')
    db_interface = String(
        default='mongo')
    client_scripts = String(
        default='/allen/aibs/pipeline/image_processing/volume_assembly/render-jars/production/scripts',
        required=False,
        description='render bin path')

    @post_load
    def validate_data(self, data):
        if data['db_interface'] is 'mongo':
            if data['mongo_host'] is None or data['mongo_userName'] is None or data['mongo_authenticationDatabase'] is None or data['mongo_password'] is None:
                raise ValidationError("Need mongo DB details")
        else:
            if data['host'] is None:
                raise ValidationError("Need render host") 

class hdf5_options(ArgSchema):
    output_dir = String(
        default='/allen/programs/celltypes/workgroups/em-connectomics/danielk/solver_exchange/python/')
    chunks_per_file = Int(
        default=5,
        description='how many sections with upward-looking cross section to write per .h5 file')

class matrix_assembly(ArgSchema):
    depth = Int(
        default=2,
        required=False,
        description='depth in z for matrix assembly point matches')
    cross_pt_weight = Float(
        default=1.0,
        required=False,
        description='weight of cross section point matches')
    montage_pt_weight = Float(
        default=1.0,
        required=False,
        description='weight of montage point matches')
    npts_min = Int(
        default=5,
        missing=5,
        required=False,
        description='disregard any tile pairs with fewer points than this')
    npts_max = Int(
        default=500,
        required=False,
        description='truncate any tile pairs to this size')
    choose_random = Boolean(
        default=False,
        required=False,
        description='choose random pts to meet for npts_max vs. just first npts_max')
    inverse_dz = Boolean(
        default=True,
        required=False,
        description='cross section point match weighting fades with z')


class regularization(ArgSchema):
    default_lambda = Float(
        default=0.005,
        description='regularization factor')
    translation_factor = Float(
        default=0.005,
        description='regularization factor')
    freeze_first_tile = Boolean(
        default=False,
        required=False)


class pointmatch(db_params):
    collection_type = String(
        default='pointmatch',
        description="'stack' or 'pointmatch'")


class stack(db_params):
    collection_type = String(
        default='stack',
        description="'stack' or 'pointmatch'")
    use_rest = Boolean(
        default=False,
        description="passed as arg in import_tilespecs_parallel")


class EMA_Schema(ArgSchema):
    first_section = Int(
        required=True, 
        description = 'first section for matrix assembly')
    last_section = Int(
        required=True,
        description = 'last section for matrix assembly')
    n_parallel_jobs = Int(
        default=4,
        required=False,
        description = 'number of parallel jobs that will run for assembly')
    solve_type = String(
        default='montage',
        required=False,
        description='Solve type options (montage, 3D) Default=montage')
    close_stack = Boolean(
        default=True,
        required=False,
        description='Close the output stack? - default - True')
    overwrite_zlayer = Boolean(
        default=True,
        required=False,
        description='delete section before import tilespecs?')
    profile_data_load = Boolean(
        default=False)
    transformation = String(
        default='AffineModel',
        validate=lambda x: x in ['AffineModel','SimilarityModel','Polynomial2DTransform', 'affine', 'rigid', 'affine_fullsize'])
    fullsize_transform = Boolean(
        default=False,
        description='use fullsize affine transform')
    poly_order = Int(
        default=3,
        required=False,
        description='order of polynomial transform')
    output_mode = String(
        default='hdf5')
    assemble_from_file = String(
        default='',
        description = 'fullpath to solution_input.h5')
    ingest_from_file = String(
        default='',
        description = 'fullpath to solution_output.h5')
    render_output = String(
        default='null',
        description = '/path/to/file, null (devnull), or stdout for where to redirect render output')
    input_stack = Nested(stack)
    output_stack = Nested(stack)
    pointmatch = Nested(pointmatch)
    hdf5_options = Nested(hdf5_options)
    matrix_assembly = Nested(matrix_assembly)
    regularization = Nested(regularization)
    showtiming = Int(
        default=1,
        description = 'have the routine showhow long each process takes')
    log_level = String(
        default="INFO",
        description='logging level')

class EMA_PlotSchema(EMA_Schema):
    z1 = Int(
        default=1000,
        description='first z for plot')
    z2 = Int(
        default=1000,
        description='second z for plot')
    zoff = Int(
        default=0,
        description='z offset betwene pointmatches and tilespecs')
    plot = Boolean(
        default=True,
        description='make a plot, otherwise, just text output')
    savefig = Boolean(
        default=False,
        description='save to a pdf')
    plot_dir = String(
        default='./')
    threshold = Float(
        default=5.0,
        description='threshold for colors in residual plot [pixels]')
    density = Boolean(
        default=True,
        description='whether residual plot is density (for large numbers of points) or just points')

