from pymongo import MongoClient
import numpy as np
import renderapi
from .. EM_aligner_python_schema import *
from .. EMaligner import make_dbconnection,get_matches
import time
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import subprocess

def specs(z,stack):
    #force render, so we can read the bbox
    stack['db_interface']='render'
    dbconnection = make_dbconnection(stack)
    xc = []
    yc = []
    tid = []
    tspecs = renderapi.tilespec.get_tile_specs_from_z(stack['name'],float(z),render=dbconnection)
    for k in np.arange(len(tspecs)):
        xc.append(0.5*tspecs[k].bbox[0]+tspecs[k].bbox[2])
        yc.append(0.5*tspecs[k].bbox[1]+tspecs[k].bbox[3])
        tid.append(tspecs[k].tileId)
    return np.array(xc),np.array(yc),np.array(tid)

def make_plot(z1,z2,stack,collection):
    cmap = plt.cm.plasma_r
    x1,y1,id1 = specs(z1,stack)
    x2,y2,id2 = specs(z2,stack)
    dbconnection = make_dbconnection(collection)

    #use mongo to get the point matches
    pm = get_matches(z1,z2,collection,dbconnection)

    print'len(pm): %d'%(len(pm))

    lclist = [] #will hold coordinates of line segments (between tile pairs)
    clist = []  #will hold number of point match pairs, used as color

    #tiny line to make sure zero is in there for consistent color range
    tmp=[]
    tmp.append((0,0))
    tmp.append((0,0.1))
    lclist.append(tmp)
    clist.append(0)
    #tiny line to make sure max is in there for consistent color range
    tmp=[]
    tmp.append((0.1,0.1))
    tmp.append((0,0.1))
    lclist.append(tmp)
    if z1!=z2:
       clist.append(500) #limit was set at 500 for cross-section matches
    else:
       clist.append(200) #limit was set at 200 for within section matches

    if len(pm)!=0: #only plot if there are matches
        for k in np.arange(len(pm)):
            #find the tilespecs
            k1 = np.argwhere(id1==pm[k]['qId']).flatten()
            k2 = np.argwhere(id2==pm[k]['pId']).flatten()
            if (k1.size!=0)&(k2.size!=0):
                k1 = k1[0]
                k2 = k2[0]
                tmp=[]
                tmp.append((x1[k1],y1[k1]))
                tmp.append((x2[k2],y2[k2]))
                lclist.append(tmp)
                clist.append(len(pm[k]['matches']['q'][0]))
        print sum(clist[2:]) 
        #plot the line segments all at once for speed:
        # https://matplotlib.org/examples/pylab_examples/line_collection2.html
        fig = plt.figure(1,figsize=(11.69,8.27))
        fig.clf()
        ax = fig.add_subplot(111)
        ax.set_xlim(0,580000)
        ax.set_ylim(0,580000)

        LC = LineCollection(lclist,cmap=cmap)
        LC.set_array(np.array(clist))
        ax.add_collection(LC)
        fig = plt.gcf()
        ax.set_aspect('equal')
        ax.patch.set_color([0.5,0.5,0.5]) #gray background, easier to see yellow
        ax.set_title('%s %s\n%d tile pairs %d point pairs'%(z1,z2,len(pm),sum(clist[2:])))
        fig.colorbar(LC)
        plt.draw()
        #plt.show()
        fname = '%s_%d_%d.pdf'%(collection['name'],z1,z2)
        pdf = PdfPages(fname)
        pdf.savefig(fig) #save the figure as a pdf page
        pdf.close()
        plt.ion()
        plt.show()
        print('wrote %s\n'%fname)
  
class CheckPointMatches(argschema.ArgSchemaParser):
    default_schema = EMA_Schema

    def run(self,z1,z2):
        make_plot(z1,z2,self.args['input_stack'],self.args['pointmatch'])

if __name__=='__main__':
    t0 = time.time()
    mod = CheckPointMatches(schema_type=EMA_Schema)
    mod.run(z1,z2)
    print('total time: %0.1f'%(time.time()-t0))
   