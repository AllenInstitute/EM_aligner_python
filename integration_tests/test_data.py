import os
from jinja2 import Environment, FileSystemLoader
import json
import tempfile

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
render_test_owner = os.environ.get('RENDER_TEST_OWNER', 'test')
render_output_owner = os.environ.get("RENDER_OUTPUT_OWNER","test")
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
                                          render_project = 'em_montage_test',
                                          render_host = render_host,
                                          render_owner = render_owner,
                                          render_port = render_port,
                                          render_mongo_host = render_mongo_host,
                                          mongo_port = mongo_port,
                                          render_output_owner = render_output_owner,
                                          render_client_scripts = client_script_location,
                                          solver_output_dir = 'home/$USER'
                                          )

#test_pm_parameters = render_json_template(example_env,
#                                          'point_match_parameters.json',
#                                           render_host = render_host,
#                                           render_port = render_port,
#                                           render_project = montage_project,
#                                           render_owner = render_test_owner,
#                                           render_client_scripts = client_script_location,
#                                           spark_log_dir = tempfile.mkdtemp(),
#                                           point_match_collection = 'test_montage_collection')

#RAW_STACK_INPUT_JSON = render_json_template(example_env, 
#                                            'raw_tiles_for_montage.json',
#                                            test_data_root=test_files_dir)
#MONT_PM_INPUT_JSON = json.load(open(test_files_dir+'montage_pointmatches.json','r'))
#montage_z = 1015
