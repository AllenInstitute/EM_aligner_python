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

    #make an intermediate stack where the sections are all moved and rotated a little
    dbconnection = make_dbconnection(mod.args['new_stack']) #will be the source stack
    secstack = copy.deepcopy(mod.args['new_stack'])
    secstack['name'] = mod.args['intermediate_name']
    newtspecs = []
    zvals = np.array(renderapi.stack.get_z_values_for_stack(mod.args['new_stack']['name'],render=dbconnection))

    section_trans_range = 500 #pixels
    section_rot_range = 2*np.pi/180. #radians
    section_scale_range = 0.02 #frac
    sec_xt = np.random.rand(zvals.size)*section_trans_range - section_trans_range/2
    sec_yt = np.random.rand(zvals.size)*section_trans_range - section_trans_range/2
    sec_rot = np.random.rand(zvals.size)*section_rot_range - section_rot_range/2
    sec_scale = 1.0 + np.random.rand(zvals.size)*section_scale_range - section_scale_range/2

    for i in np.arange(len(zvals)):
        tspecs = renderapi.tilespec.get_tile_specs_from_z(mod.args['new_stack']['name'],zvals[i],render=dbconnection)
        x0 = 15000+(int(np.random.rand())*2000-1000)
        y0 = 15000+(int(np.random.rand())*2000-1000)
        for j in np.arange(len(tspecs)):
            t = tspecs[j].to_dict()
            newtspecs.append(copy.deepcopy(t))
            xj = t['layout']['stageX']
            yj = t['layout']['stageY']
            newx = np.cos(sec_rot[i])*(xj-x0)-np.sin(sec_rot[i])*(yj-y0)+sec_xt[i]+x0
            newy = np.sin(sec_rot[i])*(xj-x0)+np.cos(sec_rot[i])*(yj-y0)+sec_yt[i]+y0
            m00 = sec_scale[i]*np.cos(sec_rot[i])
            m10 = sec_scale[i]*np.sin(sec_rot[i])
            m01 = -sec_scale[i]*np.sin(sec_rot[i])
            m11 = sec_scale[i]*np.cos(sec_rot[i])
            newtspecs[-1]['transforms']['specList'][0]['dataString'] = "%0.6f %0.6f %0.6f %0.6f %0.1f %0.1f"%(m00,m10,m01,m11,newx,newy)

    dbconnection = make_dbconnection(secstack)
    renderapi.stack.create_stack(secstack['name'],render=dbconnection)
    renderapi.client.import_tilespecs(secstack['name'],newtspecs,render=dbconnection)
    renderapi.stack.set_stack_state(secstack['name'],state='COMPLETE',render=dbconnection)

    #take the new stack and add in per tile transforms
    rawstack = copy.deepcopy(secstack)
    rawstack['name'] = mod.args['raw_name']
    newtspecs2 = []

    tile_trans_range = 100 #pixels
    tile_rot_range = 2*np.pi/180. #radians
    tile_scale_range = 0.02 #frac
    tile_affine_range = 0.04 #a little extra

    for i in np.arange(len(zvals)):
        tspecs = renderapi.tilespec.get_tile_specs_from_z(mod.args['new_stack']['name'],zvals[i],render=dbconnection)

        txt = np.random.rand(len(tspecs))*tile_trans_range - tile_trans_range/2
        tyt = np.random.rand(len(tspecs))*tile_trans_range - tile_trans_range/2
        trot = np.random.rand(len(tspecs))*tile_rot_range - tile_rot_range/2
        tscale = 1.0 + np.random.rand(len(tspecs))*tile_scale_range - tile_scale_range/2
        taffine = np.random.rand(len(tspecs)*4)*tile_affine_range - tile_affine_range/2

        for j in np.arange(len(tspecs)):
            t = tspecs[j].to_dict()
            newtspecs.append(copy.deepcopy(t))

            xj = t['layout']['stageX']
            yj = t['layout']['stageY']

            #section translation
            newx = np.cos(sec_rot[i])*xj-np.sin(sec_rot[i])*yj+sec_xt[i]
            newy = np.sin(sec_rot[i])*xj+np.cos(sec_rot[i])*yj+sec_yt[i]
            #tile translation 
            newx = newx+txt[j]
            newy = newy+tyt[j]
            m00 = tscale[j]*sec_scale[i]*np.cos(sec_rot[i]+trot[j])+taffine[j*4+0]
            m10 = tscale[j]*sec_scale[i]*np.sin(sec_rot[i]+trot[j])+taffine[j*4+1]
            m01 = -tscale[j]*sec_scale[i]*np.sin(sec_rot[i]+trot[j])+taffine[j*4+2]
            m11 = tscale[j]*sec_scale[i]*np.cos(sec_rot[i]+trot[j])+taffine[j*4+3]
            newtspecs[-1]['transforms']['specList'].append({"type":"leaf","className":"mpicbg.trakem2.transform.AffineModel2D","dataString":""})
            newtspecs[-1]['transforms']['specList'][-1]['dataString'] = "%0.6f %0.6f %0.6f %0.6f %0.1f %0.1f"%(m00,m10,m01,m11,newx,newy)

    dbconnection = make_dbconnection(rawstack)
    renderapi.stack.create_stack(rawstack['name'],render=dbconnection)
    renderapi.client.import_tilespecs(rawstack['name'],newtspecs,render=dbconnection)
    renderapi.stack.set_stack_state(rawstack['name'],state='COMPLETE',render=dbconnection)

