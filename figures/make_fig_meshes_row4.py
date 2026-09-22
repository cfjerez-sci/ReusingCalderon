import sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
EXP="/mnt/user-data/uploads/ReusingCalderon/data/figexport/meshes"
MSH="/mnt/user-data/uploads/ReusingCalderon/data/meshes"
NOMC="#8aa8bf"; PERC="#e0a6a6"
def load(b,n):
    V=np.loadtxt(f"{b}/{n}_verts.csv",delimiter=",")
    F=np.loadtxt(f"{b}/{n}_faces.csv",delimiter=",").astype(int)-1
    return V,F
def taper(t): return np.cos(np.pi*t/2.0)**2
def perturb_corner(V,a,tol=1e-6):
    W=V.copy(); x,y,z=V[:,0],V[:,1],V[:,2]
    on=lambda p,q,r:((np.abs(p)<tol)&(q>-tol)&(r>-tol)&(q<1+tol)&(r<1+tol))
    m=on(x,y,z); W[m,0]+=a*taper(y[m])*taper(z[m])
    m=on(y,x,z); W[m,1]+=a*taper(x[m])*taper(z[m])
    m=on(z,x,y); W[m,2]+=a*taper(x[m])*taper(y[m])
    return W
def panel(ax,Vn,Fn,Vp,Fp,title,elev,azim,zoom,cube=True,lw=0.18,elw=0.15):
    e=set()
    for f in Fn:
        for a,b in ((f[0],f[1]),(f[1],f[2]),(f[2],f[0])): e.add((min(a,b),max(a,b)))
    segs=np.array([[Vn[a],Vn[b]] for a,b in e])
    ax.add_collection3d(Line3DCollection(segs,colors=NOMC,linewidths=lw,alpha=0.55))
    ax.add_collection3d(Poly3DCollection(Vp[Fp],facecolor=PERC,edgecolor="black",linewidths=elw,alpha=0.97))
    allV=np.vstack([Vn,Vp]); mins,maxs=allV.min(0),allV.max(0); ctr=(mins+maxs)/2
    if cube:
        r=(maxs-mins).max()/2*1.02
        for lim,c in zip((ax.set_xlim,ax.set_ylim,ax.set_zlim),ctr): lim(c-r,c+r)
        ax.set_box_aspect((1,1,1),zoom=zoom)
    else:
        ext=np.maximum(maxs-mins,1e-9)
        for lim,c,x in zip((ax.set_xlim,ax.set_ylim,ax.set_zlim),ctr,ext): lim(c-0.52*x,c+0.52*x)
        ax.set_box_aspect(tuple(ext/ext.max()),zoom=zoom)
    ax.view_init(elev=elev,azim=azim); ax.set_axis_off()
    ax.set_title(title,fontsize=11,pad=-4)
Vsn,Fsn=load(EXP,"sphere_nominal"); Vsp,Fsp=load(EXP,"sphere_perturbed50pct")
Ven,Fen=load(EXP,"ellipsoid_nominal"); Vep,Fep=load(EXP,"ellipsoid_perturbed30pct")
Vfn,Ffn=load(MSH,"fichera_h0.15"); Vfp=perturb_corner(Vfn,0.15)
Van=np.load("/tmp/alm_Vn.npy"); Faf=np.load("/tmp/alm_F.npy"); Vap=np.load("/tmp/alm_Vp.npy")
wr=[float(x) for x in sys.argv[2].split(",")] if len(sys.argv)>2 else [1,1,1,1]
fig=plt.figure(figsize=(float(sys.argv[3]) if len(sys.argv)>3 else 10.5,3.05))
gs=fig.add_gridspec(1,4,width_ratios=wr,wspace=-0.04)
ax=fig.add_subplot(gs[0,0],projection="3d"); panel(ax,Vsn,Fsn,Vsp,Fsp,r"Sphere, $50\%$",20,-60,1.38)
ax=fig.add_subplot(gs[0,1],projection="3d"); panel(ax,Ven,Fen,Vep,Fep,r"Ellipsoid, $30\%$",20,-60,1.50)
ax=fig.add_subplot(gs[0,2],projection="3d"); panel(ax,Vfn,Ffn,Vfp,Ffn,r"Fichera corner, $15\%$",20,70,1.05)
ax=fig.add_subplot(gs[0,3],projection="3d"); panel(ax,Van,Faf,Vap,Faf,r"NASA almond, $\lambda/6.3$",30,-62,1.25,cube=False,lw=0.22,elw=0.13)
fig.subplots_adjust(left=0.0,right=1.0,top=0.97,bottom=0.0)
plt.savefig(sys.argv[1],dpi=200,bbox_inches="tight"); print("saved",sys.argv[1])
