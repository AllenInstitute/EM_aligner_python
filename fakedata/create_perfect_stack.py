import numpy as np
import renderapi
from fake_schema import *
import copy
import json
import sys
sys.path.insert(0,'../')
from assemble_matrix import make_dbconnection

if __name__=='__main__':
    mod = argschema.ArgSchemaParser(schema_type=MySchema)

    f = open('tilespec_template.json','r')
    tilespec_template = json.load(f)
    f.close()

    newtspecs = []
    for iz in np.arange(mod.args['ntile_z']):
        for ix in np.arange(mod.args['ntile_z']):
            for iy in np.arange(mod.args['ntile_z']):
                newtspec = copy.deepcopy(tilespec_template)
                newtspec['layout']['sectionId'] = str(float(tilespec_template['layout']['sectionId'])+iz)
                newtspec['z'] = float(newtspec['layout']['sectionId'])
                newtspec['tileId'] = 'faketile_%d_%d_%0.1f'%(ix,iy,float(newtspec['layout']['sectionId']))
                newtspec['layout']['imageRow'] = ix
                newtspec['layout']['imageCol'] = iy
                newtspec['layout']['stageX'] = float(ix*(tilespec_template['width']-mod.args['overlap']))
                newtspec['layout']['stageY'] = float(iy*(tilespec_template['height']-mod.args['overlap']))
                newtspec['transforms']['specList'][0]['dataString'] = "1.0 0.0 0.0 1.0 %0.1f %0.1f"%(newtspec['layout']['stageX'],newtspec['layout']['stageY'])
                newtspecs.append(newtspec)

    dbconnection = make_dbconnection(mod.args['new_stack'])
    renderapi.stack.create_stack(mod.args['new_stack']['name'],render=dbconnection)
    renderapi.client.import_tilespecs(mod.args['new_stack']['name'],newtspecs,render=dbconnection)
    renderapi.stack.set_stack_state(mod.args['new_stack']['name'],state='COMPLETE',render=dbconnection)
