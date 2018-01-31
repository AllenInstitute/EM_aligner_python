import numpy as np
import renderapi
from fake_schema import *
import copy
import json
import sys
sys.path.insert(0,'../')
from assemble_matrix import make_dbconnection
from shapely.geometry import Polygon
from descartes.patch import PolygonPatch
                            
def transform(xvals,yvals,tspec):
    x = xvals - tspec['layout']['stageX']
    y = yvals - tspec['layout']['stageY']
    [m00,m10,m01,m11,dx,dy] = np.array(tspec['transforms']['specList'][0]['dataString'].split(' ')).astype('float')
    nx = m00*x+m01*y+dx - tspec['layout']['stageX']
    ny = m10*x+m11*y+dy - tspec['layout']['stageY']
    return nx,ny

if __name__=='__main__':
    mod = argschema.ArgSchemaParser(schema_type=MySchema)

    dbconnection = make_dbconnection(mod.args['new_stack']) #will be the source stack
    zvals = np.array(renderapi.stack.get_z_values_for_stack(mod.args['new_stack']['name'],render=dbconnection))
    rawstack = copy.deepcopy(mod.args['new_stack'])
    rawstack['name'] = mod.args['raw_name']
    dbconnection2 = make_dbconnection(rawstack) #will be the source stack
    
    f = open('pointmatch_template.json')
    pm_template = json.load(f)
    f.close()

    #load the 'perfect' tilespecs
    tspecs = []
    #and the modified 'raw'
    tspecs2 = []
    for i in np.arange(len(zvals)):
        row = renderapi.tilespec.get_tile_specs_from_z(mod.args['new_stack']['name'],zvals[i],render=dbconnection)
        tspecs.append(row)
        row2 = renderapi.tilespec.get_tile_specs_from_z(rawstack['name'],zvals[i],render=dbconnection)
        tspecs2.append(row2)

    polys = []
    for row in tspecs:
        polyrow = []
        for tile in row:
            pts = []
            pts.append((tile.minX,tile.minY))
            pts.append((tile.maxX,tile.minY))
            pts.append((tile.maxX,tile.maxY))
            pts.append((tile.minX,tile.maxY))
            pts.append((tile.minX,tile.minY))
            polyrow.append(Polygon(pts))
        polys.append(polyrow)

    newmatches = []

    for i in np.arange(len(zvals)):
        jmax = np.min([i+3,len(zvals)])
        irow = polys[i]
        for j in np.arange(i,jmax):
            jrow = polys[j]
            for ni in np.arange(len(irow)):
                if i==j:
                    njmax = ni
                else:
                    njmax = len(jrow)
                for nj in np.arange(njmax):
                    intsct = irow[ni].intersection(jrow[nj])
                    if intsct.type=='Polygon':
                        area = intsct.area
                        if ((i==j)&(ni!=nj)&(area>100000.0)): #montage
                            npts = 50
                        elif (i==j)&(ni==nj):
                            npts=0
                        elif (i!=j): #cross
                            if 1000000>area>100000:
                                npts=15
                            elif area>1000000:
                                npts=75
                            else:
                                npts=0
                        else:
                            npts = 0
                        if npts!=0:
                            x0 = min(intsct.boundary.xy[0])
                            w = max(intsct.boundary.xy[0])-x0
                            y0 = min(intsct.boundary.xy[1])
                            h = max(intsct.boundary.xy[1])-y0
                            xvals = np.random.rand(npts)*w+x0
                            yvals = np.random.rand(npts)*h+y0
                            #for p
                            px,py = transform(xvals,yvals,tspecs2[i][ni].to_dict())
                            #for q
                            qx,qy = transform(xvals,yvals,tspecs2[j][nj].to_dict())
                            newpm = copy.deepcopy(pm_template)
                            ti = tspecs[i][ni].to_dict()
                            tj = tspecs[j][nj].to_dict()
                            newpm['pGroupId'] = ti['layout']['sectionId']
                            newpm['pId'] = ti['tileId']
                            newpm['qGroupId'] = tj['layout']['sectionId']
                            newpm['qId'] = tj['tileId']
                            newpm['matches']['p'][0] = px.tolist()
                            newpm['matches']['p'][1] = py.tolist()
                            newpm['matches']['q'][0] = qx.tolist()
                            newpm['matches']['q'][1] = qy.tolist()
                            newpm['matches']['w'] = np.ones(len(px)).astype('float').tolist()
                            newmatches.append(newpm)

    pm = mod.args['new_pointmatch']
    pm['db_interface'] = 'render'
    dbconnection = make_dbconnection(pm) #will be the source stack
    renderapi.pointmatch.import_matches(pm['name'],newmatches,render=dbconnection)
