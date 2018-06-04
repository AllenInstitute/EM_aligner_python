import numpy as np
import renderapi
import argschema
from .. EM_aligner_python_schema import *
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import mpl_scatter_density


def transform_pq(tspecs,matches):
    p = []
    q = []
    p_transf = []
    q_transf = [] 
    tids = []
    for ts in tspecs:    
        tids.append(np.array([x.tileId for x in ts]))
    tsp_ind0=0
    tsp_ind1=1
    if len(tspecs)==1:
        #montage
        tsp_ind1=0

    for match in matches:
        #cycle through each item, find and apply the appropriate transforms
        sind0 = np.argwhere(tids[tsp_ind0]==match['pId'])
        sind1 = np.argwhere(tids[tsp_ind1]==match['qId'])
        if (len(sind0)>0) & (len(sind1)>0):
            sind0 = sind0[0][0]
            sind1 = sind1[0][0]
            ip = np.flipud(np.rot90(np.array(match['matches']['p'])))
            iq = np.flipud(np.rot90(np.array(match['matches']['q'])))

            ip_transf = tspecs[tsp_ind0][sind0].tforms[-1].tform(ip)
            iq_transf = tspecs[tsp_ind1][sind1].tforms[-1].tform(iq)

            p.append(ip)
            q.append(iq)
            p_transf.append(ip_transf)
            q_transf.append(iq_transf)

    return [p,q,p_transf,q_transf]

class CheckResiduals(argschema.ArgSchemaParser):
    default_schema = EMA_PlotSchema

    def run(self):
        self.compute_residuals()
        self.cmap = plt.cm.plasma_r
        if self.args['plot']:
            self.make_plots()
    
    def compute_residuals(self):
        stack_dbconnection = renderapi.connect(**self.args['output_stack'])
        match_dbconnection = renderapi.connect(**self.args['pointmatch'])
       
        #get the tilespecs and pointmatches
        tspecs = [renderapi.tilespec.get_tile_specs_from_z(self.args['output_stack']['name'],float(self.args['z1']),render=stack_dbconnection)]
        if self.args['z1']!=self.args['z2']:
            tspecs.append(renderapi.tilespec.get_tile_specs_from_z(self.args['output_stack']['name'],float(self.args['z2']),render=stack_dbconnection))
        matches = renderapi.pointmatch.get_matches_from_group_to_group(self.args['pointmatch']['name'],
                                                                     str(float(self.args['z1'])),
                                                                     str(float(self.args['z2'])),
                                                                     render=match_dbconnection)

        self.p,self.q,self.p_transf,self.q_transf = transform_pq(tspecs,matches) 
        self.xy_ave = [0.5*(p+q) for p,q in zip(self.p_transf,self.q_transf)]
        self.xy_diff = [(p-q) for p,q in zip(self.p_transf,self.q_transf)]
        self.rss = [np.sqrt(np.power(x[:,0],2.0)+np.power(x[:,1],2.0)) for x in self.xy_diff]

        self.mx=''
        self.my=''
        self.mr=''

        #self.rss = np.sqrt(np.power(self.xy_diff[0,:],2.0)+np.power(self.xy_diff[1,:],2.0))
        #self.mx = '[min , max] ave +/- sig\n[%0.1f , %0.1f] %0.1f +/- %0.1f'%(self.xy_diff[0,:].min(),self.xy_diff[0,:].max(),self.xy_diff[0,:].mean(),self.xy_diff[0,:].std())
        #self.my = '[min , max] ave +/- sig\n[%0.1f , %0.1f] %0.1f +/- %0.1f'%(self.xy_diff[1,:].min(),self.xy_diff[1,:].max(),self.xy_diff[1,:].mean(),self.xy_diff[1,:].std())
        #self.mr = '[min , max] ave +/- sig\n[%0.1f , %0.1f] %0.1f +/- %0.1f'%(self.rss.min(),self.rss.max(),self.rss.mean(),self.rss.std())
    
        self.ident = 'owner: %s'%self.args['output_stack']['owner']
        self.ident += '\nproject: %s'%self.args['output_stack']['project']
        self.ident += '\nstack: %s'%self.args['output_stack']['name']
        self.ident += '\n'+'collection: %s'%self.args['pointmatch']['name']
        self.ident += '\n'+'z1,z2: %d,%d'%(self.args['z1'],self.args['z2'])
        #print('\n%s'%self.ident)
        #print(self.mx)
        #print(self.my)
        #print(self.mr)

    def make_plot(self,ax,coord_choice='xya',color_choice='rss',colorbar=False):
        #plot against original lens-corrected coordinates
        sign=1
        if coord_choice=='xya':
            plot_coords = np.concatenate(self.xy_ave)
        elif coord_choice=='p':
            plot_coords = np.concatenate(self.p)
        elif coord_choice=='q':
            plot_coords = np.concatenate(self.q)
            sign=-1
        cmin = -self.args['threshold']
        cmax = self.args['threshold']
        if color_choice=='x':
            c = sign*np.concatenate(self.xy_diff)[:,0]
            xlab = self.mx
        elif color_choice=='y':
            c = sign*np.concatenate(self.xy_diff)[:,1]
            xlab = self.my
        elif color_choice=='rss':
            c = np.concatenate(self.rss)
            xlab = self.mr
            cmin=0
        #function to make the map of the residuals as scatter density plots
        if self.args['density']:
            density = ax.scatter_density(plot_coords[:,0],plot_coords[:,1], c=c,cmap=self.cmap)
        else:
            density = ax.scatter(plot_coords[:,0],plot_coords[:,1], c=c,cmap=self.cmap,edgecolors=None)

        ax.set_aspect('equal')
        ax.patch.set_color([0.5,0.5,0.5])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.invert_yaxis() #match fine stack in ndviz
        ax.set_xlabel(xlab)
        fig = plt.gcf()
        if colorbar:
            fig.colorbar(density)

        if self.args['threshold'] is not None:
            density.set_clim(cmin,cmax)

    def make_lc_plots(self,fig):
        fig.clf()
        ax1=fig.add_subplot(141,projection='scatter_density')
        self.make_plot(ax1,coord_choice='p',color_choice='x',colorbar=True)
        ax1.set_title('$\Delta x$ as p')
        ax2=fig.add_subplot(142,projection='scatter_density')
        self.make_plot(ax2,coord_choice='q',color_choice='x',colorbar=True)
        ax2.set_title('$\Delta x$ as q')
        ax3=fig.add_subplot(143,projection='scatter_density')
        self.make_plot(ax3,coord_choice='p',color_choice='y',colorbar=True)
        ax3.set_title('$\Delta y$ as p')
        ax4=fig.add_subplot(144,projection='scatter_density')
        self.make_plot(ax4,coord_choice='q',color_choice='y',colorbar=True)
        ax4.set_title('$\Delta y$ as q')


    def make_plots(self):
        fig = plt.figure(1,figsize=(12,7.5))
        fig.clf()

        #x residuals
        ax1 = fig.add_subplot(1,3,1,projection='scatter_density')
        self.make_plot(ax1,coord_choice='xya',color_choice='x')
        #y residuals
        ax2 = fig.add_subplot(1,3,2,projection='scatter_density')
        self.make_plot(ax2,coord_choice='xya',color_choice='y')
        #rss residuals
        ax3 = fig.add_subplot(1,3,3,projection='scatter_density')
        self.make_plot(ax3,coord_choice='xya',color_choice='rss')
    
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
    mod = CheckResiduals(schema_type=EMA_PlotSchema)
    mod.run()
   
