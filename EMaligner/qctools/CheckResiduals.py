from pymongo import MongoClient
import numpy as np
import renderapi
import argschema
from .. EM_aligner_python_schema import *
from .. EMaligner import make_dbconnection,get_matches
import time
import matplotlib
matplotlib.use('Agg')
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
              
            pxy = transform(p[:,subind],tforms[0][sind1])
            qxy = transform(q[:,subind],tforms[1][sind2])

            thind = np.arange(pxy.shape[1])

            #append onto the outputs
            if thind.size>0:
               xya.append(0.5*(pxy[:,thind]+qxy[:,thind]))
               xyd.append(pxy[:,thind]-qxy[:,thind])

    return [np.block(xya),np.block(xyd)]

class CheckResiduals(argschema.ArgSchemaParser):
    default_schema = EMA_PlotSchema

    def run(self):
        if self.args['z1'] > self.args['z2']:
            tmp = self.args['z1']
            self.args['z1'] = self.args['z2']
            self.args['z2'] = tmp
        self.compute_values()
        if self.args['plot']:
            self.make_plots()
    
    def compute_values(self):
        self.cmap = plt.cm.plasma_r
        self.args['output_stack']['db_interface']='render'
        z1 = self.args['z1']
        z2 = self.args['z2']
        stack_dbconnection = make_dbconnection(self.args['output_stack'])
        dbconnection = make_dbconnection(self.args['pointmatch'])
       
        #get the tilespecs, ids, and transforms
        tids = []
        tforms = []
        for z in [z1,z2]:
            tspecs = renderapi.tilespec.get_tile_specs_from_z(self.args['output_stack']['name'],int(float(z)),render=stack_dbconnection)
            tids.append([])
            tforms.append([])
            for ts in tspecs:
                tids[-1].append(ts.tileId)
                tforms[-1].append(ts.tforms[-1])
    
        tids[0] = np.array(tids[0])
        tids[1] = np.array(tids[1])
    
        #use mongo to get the point matches
        matches = get_matches(z1,z2,self.args['pointmatch'],dbconnection)
        self.xya,self.xyd = compute_residuals(tids,tforms,matches) 
        self.rss = np.sqrt(np.power(self.xyd[0,:],2.0)+np.power(self.xyd[1,:],2.0))
    
        self.mx = '[min , max] ave +/- sig\n[%0.1f , %0.1f] %0.1f +/- %0.1f'%(self.xyd[0,:].min(),self.xyd[0,:].max(),self.xyd[0,:].mean(),self.xyd[0,:].std())
        self.my = '[min , max] ave +/- sig\n[%0.1f , %0.1f] %0.1f +/- %0.1f'%(self.xyd[1,:].min(),self.xyd[1,:].max(),self.xyd[1,:].mean(),self.xyd[1,:].std())
        self.mr = '[min , max] ave +/- sig\n[%0.1f , %0.1f] %0.1f +/- %0.1f'%(self.rss.min(),self.rss.max(),self.rss.mean(),self.rss.std())
    
        self.ident = 'owner: %s'%self.args['output_stack']['owner']
        self.ident += '\nproject: %s'%self.args['output_stack']['project']
        self.ident += '\nstack: %s'%self.args['output_stack']['name']
        self.ident += '\n'+'collection: %s'%self.args['pointmatch']['name']
        self.ident += '\n'+'z1,z2: %d,%d'%(z1,z2)
        print('\n%s'%self.ident)
        print(self.mx)
        print(self.my)
        print(self.mr)


    def make_sd_plot(self,ax,choice='rss'):
        cmin = -self.args['threshold']
        cmax = self.args['threshold']
        if choice=='x':
            c = self.xyd[0,:]
            xlab = self.mx
        elif choice=='y':
            c = self.xyd[1,:]
            xlab = self.my
        elif choice=='rss':
            c = self.rss
            xlab = self.mr
            cmin=0

        #function to make the map of the residuals as scatter density plots
        if self.args['density']:
            density = ax.scatter_density(self.xya[0,:],self.xya[1,:], c=c,cmap=self.cmap)
        else:
            density = ax.scatter(self.xya[0,:],self.xya[1,:], c=c,cmap=self.cmap,edgecolors=None)
        ax.set_aspect('equal')
        ax.patch.set_color([0.5,0.5,0.5])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.invert_yaxis() #match fine stack in ndviz
        ax.set_xlabel(xlab)
        fig = plt.gcf()
        fig.colorbar(density)
        if self.args['threshold'] is not None:
            density.set_clim(cmin,cmax)

    def make_plots(self):
        fig = plt.figure(1,figsize=(40,7.5))
        fig.clf()

        #x residuals
        ax1 = fig.add_subplot(1,3,1,projection='scatter_density')
        self.make_sd_plot(ax1,choice='x')
        #y residuals
        ax2 = fig.add_subplot(1,3,2,projection='scatter_density')
        self.make_sd_plot(ax2,choice='y')
        #rss residuals
        ax3 = fig.add_subplot(1,3,3,projection='scatter_density')
        self.make_sd_plot(ax3,choice='rss')
    
        ax1.set_title(self.ident+'\n$\Delta x$',fontsize=10)
        ax2.set_title(self.ident+'\n$\Delta y$',fontsize=10)
        ax3.set_title(self.ident+'\n$\sqrt{\Delta x^2+\Delta y^2}$',fontsize=10)
       
        ax1.set_xlabel(self.mx,fontsize=12)
        ax2.set_xlabel(self.my,fontsize=12)
        ax3.set_xlabel(self.mr,fontsize=12)
 
        if self.args['savefig']:
           self.outputname = '%s/residuals_%s_%s_%d_%d.pdf'%(self.args['plot_dir'],self.args['output_stack']['name'],self.args['pointmatch']['name'],self.args['z1'],self.args['z2'])
           pdf = PdfPages(self.outputname)
           pdf.savefig(fig,dpi=100) #save the figure as a pdf page
           pdf.close()
           plt.ion()
           plt.show()
           print('wrote %s'%self.outputname)

if __name__=='__main__':
    t0 = time.time()
    mod = CheckResiduals(schema_type=EMA_PlotSchema)
    mod.run()
    print('total time: %0.1f'%(time.time()-t0))
   
