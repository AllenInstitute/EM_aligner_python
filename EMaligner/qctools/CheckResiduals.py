from pymongo import MongoClient
import numpy as np
import renderapi
from .. EM_aligner_python_schema import *
from .. EMaligner import make_dbconnection,get_matches
import time
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import mpl_scatter_density
import subprocess

def transform(pt,tr):
    #apply an affine transformation
    newpt = np.zeros_like(pt)
    newpt[0,:] = tr.M[0,0]*pt[0,:]+tr.M[0,1]*pt[1,:]+tr.M[0,2]
    newpt[1,:] = tr.M[1,0]*pt[0,:]+tr.M[1,1]*pt[1,:]+tr.M[1,2]
    return newpt

def compute_residuals(tids,tforms,matches,nsub=-1):
    xya = []
    xyd = [] 
    for match in matches:
        #cycle through each item, find the appropriate transforms, and apply the transform
        sind1 = np.argwhere(tids[0]==match['pId'])
        sind2 = np.argwhere(tids[1]==match['qId'])
        if (len(sind1)>0) & (len(sind2)>0):
            sind1 = sind1[0][0]
            sind2 = sind2[0][0]
            p = np.array(match['matches']['p'])
            q = np.array(match['matches']['q'])
            npts = p.shape[1]

            #filter the point matches, just to cut down the numbers
            subind = np.arange(npts)
            if (nsub!=-1)&(nsub<npts):
            #currently only random nsub chosen
                subind = np.random.choice(subind,nsub)
              
            pxy = transform(p[:,subind],tforms[0][sind2])
            qxy = transform(q[:,subind],tforms[1][sind1])

            thind = np.arange(pxy.shape[1])

            #append onto the outputs
            if thind.size>0:
               xya.append(0.5*(pxy[:,thind]+qxy[:,thind]))
               xyd.append(pxy[:,thind]-qxy[:,thind])

    return [np.block(xya),np.block(xyd)]

def make_sd_plot(fig,i,j,k,x,y,c,density=True):
    #function to make the map of the residuals as scatter density plots
    ax = fig.add_subplot(i,j,k,projection='scatter_density')
    if density:
        density = ax.scatter_density(x, y, c=c,cmap=plt.cm.plasma_r)
    else:
        density = ax.scatter(x, y, c=c,cmap=plt.cm.plasma_r,edgecolors=None)
    ax.set_aspect('equal')
    ax.patch.set_color([0.5,0.5,0.5])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis() #match fine stack in ndviz
    fig.colorbar(density)
    return ax,density

def make_plot(z1,z2,stack,collection,thr=None,density=True,plot=True):
    cmap = plt.cm.plasma_r
    stack['db_interface']='render'
    stack_dbconnection = make_dbconnection(stack)
    dbconnection = make_dbconnection(collection)
   
    #get the tilespecs, ids, and transforms
    tids = []
    tforms = []
    for z in [z1,z2]:
        tspecs = renderapi.tilespec.get_tile_specs_from_z(stack['name'],int(float(z)),render=stack_dbconnection)
        tids.append([])
        tforms.append([])
        for ts in tspecs:
            tids[-1].append(ts.tileId)
            tforms[-1].append(ts.tforms[-1])

    tids[0] = np.array(tids[0])
    tids[1] = np.array(tids[1])

    #use mongo to get the point matches
    matches = get_matches(z1,z2,collection,dbconnection)
    xya,xyd = compute_residuals(tids,tforms,matches) 
    rss = np.sqrt(np.power(xyd[0,:],2.0)+np.power(xyd[1,:],2.0))

    mx = 'mean(dx)   +/- sigma(dx):   %0.1f +/- %0.1f pixels'%(xyd[0,:].mean(),xyd[0,:].std())
    mx = mx + '\nmean(|dx|) +/- sigma(|dx|): %0.1f +/- %0.1f pixels'%(np.abs(xyd[0,:]).mean(),np.abs(xyd[0,:]).std())
    my = 'mean(dy)   +/- sigma(dy):   %0.1f +/- %0.1f pixels'%(xyd[1,:].mean(),xyd[1,:].std())
    my = my + '\nmean(|dy|) +/- sigma(|dy|): %0.1f +/- %0.1f pixels'%(np.abs(xyd[1,:]).mean(),np.abs(xyd[1,:]).std())
    mr = 'mean(rss)  +/- sigma(rss):  %0.1f +/- %0.1f pixels'%(rss.mean(),rss.std())

    ident = 'owner: %s'%stack['owner']
    ident += '\nproject: %s'%stack['project']
    ident += '\nstack: %s'%stack['name']
    ident += '\n'+'collection: %s'%collection['name']
    ident += '\n'+'z1,z2: %d,%d'%(z1,z2)
    print('\n%s'%ident)
    print(mx)
    print(my)
    print(mr)
   
    if plot:
        fig = plt.figure(1,figsize=(40,7.5))
        fig.clf()
        #x residuals
        ax1,d = make_sd_plot(fig,1,3,1,xya[0,:],xya[1,:],xyd[0,:],density=density)
        if thr is not None:
            d.set_clim(-thr,thr)
        #y residuals
        ax2,d = make_sd_plot(fig,1,3,2,xya[0,:],xya[1,:],xyd[1,:],density=density)
        if thr is not None:
            d.set_clim(-thr,thr)
        #rss
        ax3,d = make_sd_plot(fig,1,3,3,xya[0,:],xya[1,:],rss,density=density)
        if thr is not None:
            d.set_clim(0,thr)

        ident += '\n$\Delta x$'
        ax1.set_title(ident,fontsize=10)
        ax2.set_title('$\Delta y$',fontsize=18)
        ax3.set_title('$\sqrt{\Delta x^2+\Delta y^2}$',fontsize=18)
   
        ax1.set_xlabel(mx,fontsize=12)
        ax2.set_xlabel(my,fontsize=12)
        ax3.set_xlabel(mr,fontsize=12)
        
        fname = 'residuals_%s_%s_%d_%d.pdf'%(stack['name'],collection['name'],z1,z2)
        pdf = PdfPages(fname)
        pdf.savefig(fig,dpi=100) #save the figure as a pdf page
        pdf.close()
        plt.ion()
        plt.show()
   
        return fname

class CheckResiduals(argschema.ArgSchemaParser):
    default_schema = EMA_Schema

    def run(self,z1,z2,thr=None,density=True,plot=True):
        make_plot(z1,z2,self.args['output_stack'],self.args['pointmatch'],thr=thr,density=density,plot=plot)

if __name__=='__main__':
    t0 = time.time()
    mod = CheckPointMatches(schema_type=EMA_Schema)
    mod.run(z1,z2)
    print('total time: %0.1f'%(time.time()-t0))
   
