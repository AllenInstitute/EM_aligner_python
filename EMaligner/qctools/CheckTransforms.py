from pymongo import MongoClient
import numpy as np
import renderapi
from .. EM_aligner_python_schema import *
from .. EMaligner import make_dbconnection,get_matches
import time
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
from shapely.geometry import Polygon
from descartes.patch import PolygonPatch
import mpl_scatter_density
import mpl_scatter_density
import subprocess

def fixpi(arr):
    #make the angular values fall around zero
    ind = np.argwhere(arr>np.pi)
    while len(ind)!=0:
        arr[ind] = arr[ind]-2.0*np.pi
        ind = np.argwhere(arr>np.pi)
    ind = np.argwhere(arr<-np.pi)
    while len(ind)!=0:
        arr[ind] = arr[ind]+2.0*np.pi
        ind = np.argwhere(arr<-np.pi)
    return arr


def make_patch(tile):
    pts = []
    pts.append((tile.minX,tile.minY))
    pts.append((tile.maxX,tile.minY))
    pts.append((tile.maxX,tile.maxY))
    pts.append((tile.minX,tile.maxY))
    pts.append((tile.minX,tile.minY))
    return PolygonPatch(Polygon(pts))

def make_transform_patches(tilespecs):
    #getting tilespecs ready for plotting
    patches = []
    shearlist = []
    xscalelist = []
    yscalelist = []
    rotlist = []
    xmin = 1e9
    xmax = -1e9
    ymin = 1e9
    ymax = -1e9
    border=4000
    for ts in tilespecs:
        patches.append(make_patch(ts)) 
        shearlist.append(ts.tforms[-1].shear)
        xscalelist.append(ts.tforms[-1].scale[0])
        yscalelist.append(ts.tforms[-1].scale[1])
        rotlist.append(ts.tforms[-1].rotation)
        if ts.minX < xmin:
            xmin=ts.minX
        if ts.minY < ymin:
            ymin=ts.minY
        if ts.maxX > xmax:
            xmax=ts.maxX
        if ts.maxY > ymax:
            ymax=ts.maxY

    xscalelist = np.array(xscalelist)
    yscalelist = np.array(yscalelist)
    shearlist = fixpi(np.array(shearlist))
    rotlist = fixpi(np.array(rotlist))
    return [patches,shearlist,rotlist,xscalelist,yscalelist],(xmin-border,xmax+border),(ymin-border,ymax+border)

class CheckTransforms(argschema.ArgSchemaParser):
    default_schema = EMA_Schema

    def run(self,z1,plot=True):
        self.make_plot(z1,self.args['output_stack'],plot=plot)

    def make_transform_plot(fig,i,j,k,xlim,ylim,patches,value,bar=True):
        #plot a map of the transform value
        cmap = plt.cm.plasma_r
        ax = fig.add_subplot(i,j,k)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        LC = PatchCollection(patches,cmap=cmap)
        LC.set_array(value)
        LC.set_edgecolor('none')
        ax.add_collection(LC)
        if bar:
            fig.colorbar(LC)
        ax.set_aspect('equal')
        ax.patch.set_color([0.5,0.5,0.5])
        ax.set_xticks([])
        ax.set_yticks([])
        return ax
    
    
    def make_plot(self,z1,stack,thr=None,plot=True):
        stack['db_interface']='render'
        stack_dbconnection = make_dbconnection(stack)
        
        tspecs = renderapi.tilespec.get_tile_specs_from_z(stack['name'],int(float(z1)),render=stack_dbconnection)
        tids = []
        tforms = []
        for ts in tspecs:
            tids.append(ts.tileId)
            tforms.append(ts.tforms[-1])
    
        fig = plt.figure(1,figsize=(16,4))
        fig.clf()
    
        tpatches,xlim,ylim = make_transform_patches(tspecs)
        self.shear = tpatches[1]
        self.rotation = tpatches[2]
        self.xscale = tpatches[3]
        self.yscale = tpatches[4]

        i=0
        if plot:
            for j in np.arange(1,5):
                make_transform_plot(fig,2,2,j,xlim,ylim,tpatches[0],tpatches[j])
            plt.subplot(2,2,1);plt.title('shear')
            plt.subplot(2,2,2);plt.title('rotation')
            plt.subplot(2,2,3);plt.title('xscale')
            plt.subplot(2,2,4);plt.title('yscale')
    
            fname = 'transforms_%s_%d.pdf'%(stack['name'],z1)
            pdf = PdfPages(fname)
            pdf.savefig(fig) #save the figure as a pdf page
            pdf.close()
            plt.ion()
            plt.show()


if __name__=='__main__':
    t0 = time.time()
    mod = CheckTransforms(schema_type=EMA_Schema)
    mod.run(z1)
    print('total time: %0.1f'%(time.time()-t0))
   
