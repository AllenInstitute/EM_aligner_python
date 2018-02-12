import os
from jinja2 import Environment, FileSystemLoader
import json

def render_json_template(env, template_file, **kwargs):
    template = env.get_template(template_file)
    d = json.loads(template.render(**kwargs))
    return d

test_files_dir = os.path.join(os.path.dirname(__file__), 'test_files')
example_env = Environment(loader=FileSystemLoader(test_files_dir))

render_host = os.environ.get('RENDER_HOST','localhost')
render_port = os.environ.get('RENDER_PORT',8080)
render_mongo_host = os.environ.get('RENDER_MONGO_HOST','localhost')
mongo_port = os.environ.get('MONGO_PORT',27017)
render_owner = os.environ.get('RENDER_OWNER',"test")
render_output_owner = os.environ.get("RENDER_OUTPUT_OWNER","test")
solver_output_dir = os.environ.get("EM_ALIGNER_OUTPUT_DIR",
    "/allen/programs/celltypes/workgroups/em-connectomics/danielk/solver_exchange/python/")

client_script_location = os.environ.get('RENDER_CLIENT_SCRIPTS',
                          ('/var/www/render/render-ws-java-client/'
                          'src/main/scripts/'))

render_params = {
    'host':render_host,
    'port':render_port,
    'owner':'test',
    'project':'test_project',
    'client_scripts':client_script_location
}

montage_parameters = render_json_template(example_env, 'montage_test.json',
                                          render_host = render_host,
                                          render_owner = render_owner,
                                          render_port = render_port,
                                          render_mongo_host = render_mongo_host,
                                          mongo_port = mongo_port,
                                          render_output_owner = render_output_owner,
                                          solver_output_dir = solver_output_dir
                                          )
