import numpy as np
import triangle
import scipy.optimize 
from scipy.spatial import Delaunay

example={
    'npts':100000,
    'xinbox':0.0,
    'yinbox':0.0,
    'nvtarget':1800,
    'tile_width':3840,
    'tile_height':3840,
}

class MeshMatches():

    def __init__(self,nvtarget,width,height,coords=None,matches=None):
        if (coords is None)&(matches is None):
            print('must provide Nx2 coordinates or point matches')
            return
        if (matches is not None)&(coords is None):
            self.matches=matches
            coords=self.condense_coords()
        self.coords=coords
        self.nvtarget=nvtarget
        self.width=width
        self.height=height

    def condense_coords(self):
        #condense point match structure into Nx2
        x=[]
        y=[]
        for m in self.matches:
            x+=m['matches']['p'][0]
            x+=m['matches']['q'][0]
            y+=m['matches']['p'][1]
            y+=m['matches']['q'][1]
        return np.transpose(np.vstack((np.array(x),np.array(y))))

    def run(self):
        self.create_PSLG()
        self.find_delaunay_with_max_vertices()
        self.force_vertices_with_npoints(3)
        if self.has_hole:
            self.add_center_point()
        print('asked for %d vertices'%self.nvtarget)
        print('returned  %d vertices'%self.mesh.npoints)

    def add_center_point(self):
        #maybe don't need it
        return

    def create_PSLG(self):
        #define a PSLG for triangle
        #http://dzhelil.info/triangle/definitions.html
        #https://www.cs.cmu.edu/~quake/triangle.defs.html#pslg
        vertices=np.array([
                [0,0],
                [0,self.height],
                [self.width, self.height],
                [self.width, 0]])
        segments=np.array([
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 0]])
        #check if there is a hole (focuses elements for montage sets)
        self.has_hole=False
        hw=0.5*self.width
        hh=0.5*self.height
        r=np.sqrt(np.power(self.coords[:,0]-hw,2.0)+np.power(self.coords[:,1]-hh,2.0))
        if r.min()>0.05*0.5*(self.width+self.height):
            inbx=50
            inby=50
            while np.count_nonzero((np.abs(self.coords[:,0]-hw)<inbx)&(np.abs(self.coords[:,1]-hh)<inby))==0:
                inbx+=50
            inbx-=50
            while np.count_nonzero((np.abs(self.coords[:,0]-hw)<inbx)&(np.abs(self.coords[:,1]-hh)<inby))==0:
                inby+=50
            inby-=50
            vertices=np.append(vertices,np.array([[hw-inbx,hh-inby],[hw-inbx,hh+inby],[hw+inbx,hh+inby],[hw+inbx,hh-inby]])).reshape(8,2)
            segments=np.append(segments,np.array([[4,5],[5,6],[6,7],[7,4]])).reshape(8,2)
            self.has_hole=True
        self.bbox={}
        self.bbox['vertices']=vertices
        self.bbox['segments']=segments
        if self.has_hole:
            self.bbox['holes']=np.array([[hw,hh]])

    def calculate_mesh(self,a,target,get_t=False):
        t=triangle.triangulate(self.bbox,'pqa%0.1f'%a)
        if get_t:
            #scipy.Delaunay has nice find_simplex method, 
            #but, no obvious way to iteratively refine meshes, like triangle
            #numbering is different, so, switch here 
            return Delaunay(t['vertices'])
        return target-len(t['vertices'])

    def find_delaunay_with_max_vertices(self):
        #find bracketing values
        a1=a2=1e6
        t1=self.calculate_mesh(a1,self.nvtarget)
        afac=np.power(10.,-np.sign(t1))
        while np.sign(t1)==np.sign(self.calculate_mesh(a2,self.nvtarget)):
            a2*=afac
        val_at_root=-1
        nvtweak=self.nvtarget
        while val_at_root<0:
            a=scipy.optimize.brentq(self.calculate_mesh,a1,a2,args=(nvtweak,))
            val_at_root=self.calculate_mesh(a,self.nvtarget)
            a1=a*2
            a2=a*0.5
            nvtweak-=1
        self.mesh=self.calculate_mesh(a,None,get_t=True)
        self.area_triangle_par=a
        return 

    def compute_barycentrics(self,coords):
        #https://en.wikipedia.org/wiki/Barycentric_coordinate_system#Conversion_between_barycentric_and_Cartesian_coordinates
        triangle_indices = self.mesh.find_simplex(coords)
        vt=np.vstack((np.transpose(self.mesh.points),np.ones(self.mesh.points.shape[0])))
        mt=np.vstack((np.transpose(coords),np.ones(coords.shape[0])))
        bary=np.zeros((3,coords.shape[0]))
        self.Rinv=[]
        for tri in self.mesh.simplices:
            self.Rinv.append(np.linalg.inv(vt[:,tri]))
        for i in range(self.mesh.nsimplex): 
            ind = np.argwhere(triangle_indices==i).flatten()
            bary[:,ind] = self.Rinv[i].dot(mt[:,ind])
        return np.transpose(bary)

    def count_points_near_vertices(self,t):
        flat_tri=t.simplices.flatten()
        flat_ind=np.repeat(np.arange(t.nsimplex),3)
        v_touches=[] 
        for i in range(t.npoints):
            v_touches.append(flat_ind[np.argwhere(flat_tri==i)])
        pt_count=np.zeros(t.npoints)
        found = t.find_simplex(self.coords,bruteforce=True)
        for i in range(t.npoints):
            for j in v_touches[i]:
                pt_count[i]+=np.count_nonzero(found==j)
        return pt_count

    def force_vertices_with_npoints(self,npts):
        while True:
            t=self.calculate_mesh(self.area_triangle_par,None,get_t=True)
            pt_count = self.count_points_near_vertices(t)
            if pt_count.min()>=npts:
                break
            self.area_triangle_par*=1.05
        self.mesh = t
        return 
    
if __name__=='__main__':
    #fake some coordinates
    coords = np.random.rand(example['npts'],2)
    coords[:,0]*=example['tile_width']
    coords[:,1]*=example['tile_height']
    ind=np.argwhere((np.abs(coords[:,0]-0.5*example['tile_width'])<0.5*example['xinbox']*example['tile_width'])&(np.abs(coords[:,1]-0.5*example['tile_height'])<0.5*example['yinbox']*example['tile_height'])).flatten()
    x = np.delete(coords[:,0],ind)
    y = np.delete(coords[:,1],ind)
    coords=np.transpose(np.vstack((x,y)))

    mod = MeshMatches(example['nvtarget'],example['tile_width'],example['tile_height'],coords=coords)
    mod.run()

