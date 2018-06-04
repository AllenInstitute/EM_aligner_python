import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

def plot_mesh_and_points(mod,ax):
    pc=[]
    for tri in mod.mesh.simplices:
       pc.append(Polygon(mod.mesh.points[tri]))
    pcoll = PatchCollection(pc,edgecolors=[0.,0.,0.,1.],facecolors=[0,1.,1.,0.3],linewidth=1.0)
    ax.scatter(mod.coords[:,0],mod.coords[:,1],marker='.',c='r',s=1.8)
    ax.add_collection(pcoll)
    ax.set_aspect('equal')
