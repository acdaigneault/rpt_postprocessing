import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches
from matplotlib import ticker
from scipy.linalg import norm
from scipy import optimize


# Set LaTex font
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

# Reactor dimensions
L = 0.3 # m
R = 0.1 # m

# Detector dimensions
r = 0.0381 # m
l = 0.0762 # m

# Position of the detector
FP = [0.2, 0, 0.075]
MP = [0.2381, 0, 0.075]
print(FP, MP)

# Positions specifications
x_ = "x"
y_ = "y"
z_ = "z"
positions_real = pd.read_csv("reconstruction.csv", usecols=["real_x", "real_y", "real_z"])
positions_found = pd.read_csv("reconstruction.csv", usecols=[x_, y_, z_])

# Path to save figures and format
global save_path
save_path = "/home/audrey/image_presentation/reconstruction/distance_volume_100000"
image_format = ".png"


# Distance
def calculate_distance3d(positions_real, positions_found):
    d = np.sqrt((positions_real[:,0] - positions_found[:,0])**2 +
                (positions_real[:,1] - positions_found[:,1])**2 +
                (positions_real[:,2] - positions_found[:,2]) ** 2)
    return d

def calculate_distance2d(positions_real, positions_found, x, y):

    d = np.sqrt((positions_real[:,x] - positions_found[:,x])**2 +
                (positions_real[:,y] - positions_found[:,y])**2)
    return d


def calculate_distance_center(positions_found):

    d = np.sqrt((positions_found[:,0])**2 +
                (positions_found[:,1])**2)
    return d

# Function for plotting
def plotting(data, X, Y, x, y, color, plan, title=0):
    fig, ax = plt.subplots()

    # Allow to scale the color bar to all data in plot
    cmap = plt.get_cmap(color)
    norm = plt.Normalize(data.min(), data.max())

    if plan == "xy":
        ax.plot(np.insert(x, 1, [0.00325, 0.00505, 0.00605]), np.insert(y, 1, [0.0005, 0.0017, 0.0049]), color="black", linewidth=0.25)
    else:
        ax.plot(x, y, color="black", linewidth=0.25)

    #ax.scatter(x, y, s=25, linewidths=0.25, color="gray", edgecolors="black")

    for i, xx, yy in zip(data, X, Y):
        ax.scatter(xx, yy, c=[cmap(norm(i))], s=30, linewidths=0.25, edgecolors="black")

    #ax.scatter(X, Y, s=10, linewidths=0.25,  color="darkred")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    divider = make_axes_locatable(ax)

    # Show the detector face prior plan
    if plan == "yz":
        #detector_yz = patches.Circle((FP[1], FP[2]), r, linestyle="--", linewidth=1, edgecolor='k', facecolor='none')
        #ax.add_patch(detector_yz)
        ax.set_xlabel("Position en y (m)")
        ax.set_ylabel("Position en z (m)")
        ax.set_xlim(-R, R)
        ax.set_ylim(0.05, L - 0.05)
        fig.set_size_inches(4.25, 4.25)
        cax = divider.append_axes("top", size="6.5%", pad=0.05)

    elif plan == "xz":
        #detector_1d_side = plt.vlines(R, FP[2] - r, FP[2] + r, linestyle="--", linewidth=2, color="black")
        ax.set_xlabel("Position en x (m)")
        ax.set_ylabel("Position en z (m)")
        ax.set_xlim(-R, R)
        ax.set_ylim(0.05, L - 0.05)
        fig.set_size_inches(4.25, 4.25)
        cax = divider.append_axes("top", size="6.5%", pad=0.05)

    elif plan == "xy":
        #detector_1d_top = plt.vlines(R, FP[1] - r, FP[1] + r, linestyle="--", linewidth=2, color="black")
        reactor = patches.Circle((0, 0), R, linestyle="-", linewidth=1, edgecolor='k', facecolor='none')
        ax.add_patch(reactor)
        ax.set_xlabel("Position en x (m)")
        ax.set_ylabel("Position en y (m)")
        ax.set_xlim(-R, R)
        ax.set_ylim(-R, R)
        fig.set_size_inches(4.5, 4.25)
        cax = divider.append_axes("top", size="6.5%", pad=0.05)

    # If there's title
    if title != 0:
        ax.set_title(title)

    cb = fig.colorbar(sm, ax=ax, cax=cax, orientation="horizontal")
    cb.ax.set_title(r'Distance (m)', size=10)
    cb.locator = ticker.MaxNLocator(nbins=7)
    cb.update_ticks()
    cax.xaxis.set_ticks_position("top")


    # Set the equal scale and save figure
    ax.set_aspect("equal", "box")
    #fig.savefig(save_path + plan + image_format, dpi=500)
    plt.close(fig)
    ax.clear()
    #plt.show()

def plotting_yz(data, X, Y, x, y):
    color = "Reds"
    plan = "yz"
    plotting(data, X, Y, x, y, color, plan)


def plotting_xz(data, X, Y, x, y):  #
    color = "Blues"
    plan = "xz"
    plotting(data, X, Y, x, y, color, plan)

def plotting_xy(data, X, Y, x, y):
    color = "Greens"
    plan = "xy"
    plotting(data, X, Y, x, y, color, plan)

pos_real = np.array(positions_real)
pos_found = np.array(positions_found)
distance = calculate_distance3d(pos_real, pos_found)
title = 0
for plan in {"yz", "xz", "xy"}:
    X = pos_found[:, 0]
    Y = pos_found[:, 1]
    Z = pos_found[:, 2]
    x = pos_real[:, 0]
    y = pos_real[:, 1]
    z = pos_real[:, 2]
    if plan == "yz":
        plotting_yz(distance.T, Y, Z, y, z)
    elif plan == "xz":
        plotting_xz(distance, X, Z, x, z)
    else:
        plotting_xy(distance, X, Y, x, y)




def data_for_cylinder(p0, p1, r):
    #vector in direction of axis
    v = p1 - p0
    #find magnitude of vector
    mag = norm(v)
    #unit vector in direction of axis
    v = v / mag
    #make some vector not in the same direction as v
    not_v = np.array([1, 0, 0])
    if (np.abs(v) == not_v).all():
        not_v = np.array([0, 1, 0])
    #make vector perpendicular to v
    n1 = np.cross(v, not_v)
    #normalize n1
    n1 /= norm(n1)
    #make unit vector perpendicular to v and n1
    n2 = np.cross(v, n1)
    #surface ranges over t from 0 to length of axis and 0 to 2*pi
    t = np.linspace(0, mag, 100)
    theta = np.linspace(0, 2 * np.pi, 100)
    rsample = np.linspace(0, r, 2)

    # use meshgrid to make 2d arrays
    t, theta2 = np.meshgrid(t, theta)
    rsample, theta = np.meshgrid(rsample, theta)

    # "Tube"
    X, Y, Z = [p0[i] + v[i] * t + r * np.sin(theta2) * n1[i] + r * np.cos(theta2) * n2[i] for i in [0, 1, 2]]
    # "Bottom"
    X1, Y1, Z1 = [p0[i] + rsample[i] * np.sin(theta) * n1[i] + rsample[i] * np.cos(theta) * n2[i] for i in [0, 1, 2]]
    # "Top"
    X2, Y2, Z2 = [p0[i] + v[i] * mag + rsample[i] * np.sin(theta) * n1[i] + rsample[i] * np.cos(theta) * n2[i] for i in
                  [0, 1, 2]]

    return X, Y, Z, X1, Y1, Z1, X2, Y2, Z2




# Meshes and plots generation
Xr, Yr, Zr, X1, Y1, Z1, X2, Y2, Z2 = data_for_cylinder(np.array([0, 0, 0]), np.array([0, 0, L]), R)


angle = [0, 2*np.pi/3, -2*np.pi/3, 0, 2*np.pi/3, -2*np.pi/3]
face_x = [FP[0], FP[0], FP[0], -FP[0], -FP[0], -FP[0]]
face_xl = [FP[0]+l, FP[0]+l, FP[0]+l, -(FP[0]+l), -(FP[0]+l), -(FP[0]+l)]
pos_z = [FP[2], FP[2], FP[2], 0.225, 0.225, 0.225]


def plot_detector(p0, p1, r, angle, i):
    Xd, Yd, Zd, X1, Y1, Z1, X2, Y2, Z2  = data_for_cylinder(p0, p1, r)
    ax.plot_surface(Xd, Yd, Zd, alpha=0.5, color="royalblue", label="Detector" + str(i))
    ax.plot_surface(X1, Y1, Z1, alpha=0.5, color="royalblue")
    ax.plot_surface(X2, Y2, Z2, alpha=0.5, color="royalblue")

# Plots
fig = plt.figure(0)
ax = plt.axes(projection='3d')
ax.plot_surface(Xr, Yr, Zr, alpha=0.2, color="grey", label="Reactor")

for i in range(0, len(angle)):
    p0 = np.array([face_x[i] * np.cos(angle[i]), face_x[i] * np.sin(angle[i]), pos_z[i]])
    p1 = np.array([face_xl[i] * np.cos(angle[i]), face_xl[i] * np.sin(angle[i]), pos_z[i]])
    plot_detector(p0, p1, 0.95*r, angle[i], i)




"""
ax.plot_surface(Xr, Yr, Zr, alpha=0.2, color="grey", label="Reactor")
ax.plot_surface(Xd, Yd, Zd, alpha=0.5, color="black", label="Detector0")
top = patches.Circle((FP[1], FP[2]), r, color="black", alpha=0.5)
ax.add_patch(top)
art3d.pathpatch_2d_to_3d(top, z=FP[0], zdir="x")
ax.plot_surface(Xd1, Yd1, Zd1, alpha=0.5, color="black", label="Detector1")
ax.plot_surface(Xd2, Yd2, Zd2, alpha=0.5, color="black", label="Detector2")
ax.plot_surface(Xd3, Yd3, Zd3, alpha=0.5, color="black", label="Detector3")
ax.plot_surface(Xd4, Yd4, Zd4, alpha=0.5, color="black", label="Detector4")
ax.plot_surface(Xd5, Yd5, Zd5, alpha=0.5, color="black", label="Detector5")
"""



def moving_average(X, Y, Z):
    x = np.zeros(len(X))
    y = np.zeros(len(X))
    z = np.zeros(len(X))

    # Keep first and last positions
    x[0] = X[0]
    y[0] = Y[0]
    z[0] = Z[0]
    x[-1] = X[-1]
    y[-1] = Y[-1]
    z[-1] = Z[-1]

    # Moving average with 3 points
    for i in {1, -2}:
        x[i] = (X[i-1] + X[i] + X[i+1])/3
        y[i] = (Y[i - 1] + Y[i] + Y[i + 1]) / 3
        z[i] = (Z[i - 1] + Z[i] + Z[i + 1]) / 3

    # Moving average with 5 points
    for i in range(2, len(X)-2):
        x[i] = (X[i-2] + X[i-1] + X[i] + X[i+1] + X[i+2])/5
        y[i] = (Y[i-2] + Y[i - 1] + Y[i] + Y[i + 1] + Y[i+2]) / 5
        z[i] = (Z[i-2] + Z[i - 1] + Z[i] + Z[i + 1] + Z[i+2]) / 5

    return x, y, z


#curve_x, curve_y, curve_z = moving_average(np.array(positions_found["x"]),  np.array(positions_found["y"]),  np.array(positions_found["z"]))


#ax.scatter3D(np.array(positions_real["real_x"]),  np.array(positions_real["real_y"]), np.array(positions_real["real_z"]),             color="black", s=5)
ax.scatter3D(np.array(positions_found[x_]),  np.array(positions_found[y_]),  np.array(positions_found[z_]), color="blue", s=5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.plot(np.array(positions_real["real_x"]),  np.array(positions_real["real_y"]),  np.array(positions_real["real_z"]), color="black")

world_limits = ax.get_w_lims()
ax.set_box_aspect((world_limits[1]-world_limits[0],world_limits[3]-world_limits[2],world_limits[5]-world_limits[4]))
plt.close(fig)
ax.clear()
#plt.show()



positions_found0 = pd.read_csv("reconstruction.csv", usecols=["x0", "y0", "z0"])
pos_found0 = np.array(positions_found0)
positions_found1 = pd.read_csv("reconstruction.csv", usecols=["x1", "y1", "z1"])
pos_found1 = np.array(positions_found1)
positions_found2 = pd.read_csv("reconstruction.csv", usecols=["x2", "y2", "z2"])
pos_found2 = np.array(positions_found2)

d = calculate_distance3d(pos_real, pos_found)
d0 = calculate_distance3d(pos_real, pos_found0)
d1 = calculate_distance3d(pos_real, pos_found1)
d2 = calculate_distance3d(pos_real, pos_found2)

D = calculate_distance2d(pos_real, pos_found, 0, 1)
D0 = calculate_distance2d(pos_real, pos_found0, 0, 1)
D1 = calculate_distance2d(pos_real, pos_found1, 0, 1)
D2 = calculate_distance2d(pos_real, pos_found2, 0, 1)

DD = calculate_distance_center(pos_found)
DD0 = calculate_distance_center(pos_found0)
DD1 = calculate_distance_center(pos_found1)
DD2 = calculate_distance_center(pos_found2)
DDreal = calculate_distance_center(pos_real)

fig, ax = plt.subplots()







"""
# erreur vs distance centre
#plt.plot(DD,d,".", label="10 000 itérations de MC")
plt.plot(DD0,d0,".", label="100 000 itérations de MC", color="tab:gray")


ax.set_xlabel("Distance par rapport à l'axe central (m)")
ax.set_ylabel("Distance entre position reconstruite et théorique (m)")
fig.set_size_inches(5.45,4.5)
ax.legend(loc="upper left", edgecolor="k", fancybox=0)
ax.set_ylim(0, 0.004)
"""

"""
# erreur vs distance z
#plt.plot(pos_real[:,2],pos_found[:,2]-pos_real[:,2],".", label="10 000 itérations de MC")
#plt.plot(pos_real[:,2],pos_found1[:,2]-pos_real[:,2],".", label="10 000 itérations de MC")
plt.plot(pos_real[:,2],pos_found2[:,2]-pos_real[:,2],".", label="10 000 itérations de MC")
#plt.plot(pos_real[:,2],pos_found0[:,2]-pos_real[:,2],".", label="100 000 itérations de MC", color="tab:gray"
ax.set_xlabel("Hauteur de la position théorique (m)")
ax.set_ylabel("Distance entre position reconstruite et théorique en z (m)")
fig.set_size_inches(5.75,4.5)
ax.legend(loc="upper left", edgecolor="k", fancybox=0)
ax.set_ylim(-0.004, 0.004)
save = "/home/audrey/image_presentation/reconstruction/distance_z_10000.png"
#fig.savefig(save, dpi=500)

"""
"""
# erreur vs distance xy
#plt.plot(DDreal,D,".", label="10 000 itérations de MC")
#plt.plot(DDreal,D1,".", label="10 000 itérations de MC 2")
plt.plot(DDreal,D2,".", label="10 000 itérations de MC 2")
#plt.plot(DDreal,D0,".", label="100 000 itérations de MC", color="tab:gray")
ax.set_xlabel("Distance de la position théorique par rapport à l'axe central (m)")
ax.set_ylabel("Distance entre position reconstruite et théorique en xy (m)")
fig.set_size_inches(5.75,4.5)
ax.legend(loc="upper left", edgecolor="k", fancybox=0)
ax.set_ylim(0, 0.0025)
save = "/home/audrey/image_presentation/reconstruction/distance_xy_100000.png"
#fig.savefig(save, dpi=500)

"""

volumes = pd.read_csv("volume.csv")
v = np.array(volumes["v"])
v0 = np.array(volumes["v0"])
v1 = np.array(volumes["v1"])
v2 = np.array(volumes["v2"])
rc = np.array(volumes["rc"])
wc = np.array(volumes["wc"])
rc0 = np.array(volumes["rc0"])
wc0 = np.array(volumes["wc0"])
rc1 = np.array(volumes["rc1"])
wc1 = np.array(volumes["wc1"])
rc2 = np.array(volumes["rc2"])
wc2 = np.array(volumes["wc2"])

def go_bool(vect):
    new_vect = np.ones(len(vect), dtype=int)
    boolean = []
    for i in range(0, len(vect)):
        if np.abs(vect[i] - 1) < 1e-6:
            boolean.append(True)
        else:
            boolean.append(False)

    new_vect = np.array(boolean)
    return new_vect

rc = go_bool(rc)
rc0 = go_bool(rc0)
rc1 = go_bool(rc1)
rc2 = go_bool(rc2)
wc = go_bool(wc)
wc0 = go_bool(wc0)
wc1 = go_bool(wc1)
wc2 = go_bool(wc2)
wc1[9] = False


def func(x, a, b):
    y = a * np.exp(b * x)
    return y

def regression(x, y):
    alpha, beta = optimize.curve_fit(func, xdata = x, ydata = y)[0]
    return [alpha, beta]

[alpha, beta] = regression(np.delete(v0,9), y = np.delete(d0,9))
"""
# error vs cell volume
#plt.plot(v,d,".", label="10 000 itérations de MC")
#plt.plot(v1,d1,".", label="10 000 itérations de MC")
plt.plot(np.delete(v0,9),np.delete(d0, 9),".", label="100 000 itérations de MC", color="tab:gray")
plt.plot(np.delete(v0,9), alpha*np.exp(beta*np.delete(v0,9)), 'r')


ax.set_xlabel("Volume de la cellule (m³)")
ax.set_ylabel("Distance entre position reconstruite et théorique (m)")
fig.set_size_inches(5.45,4.5)
ax.legend(loc="upper left", edgecolor="k", fancybox=0)
ax.set_ylim(0, 0.004)
save = "/home/audrey/image_presentation/reconstruction/distance_volume_10000.png"
#fig.savefig(save, dpi=500)
"""



# erreur vs distance xy rcwr
"""
x = DDreal[rc]
y = D[rc]

[alpha, beta] = regression(DDreal[rc], D[rc])


plt.plot(DDreal[rc],D[rc],".", color="green", label="Bonne cellule")
plt.plot(x, alpha*np.exp(beta*x), color="green")
[alpha, beta] = regression(DDreal[wc], D[wc])
plt.plot(DDreal[wc],D[wc],".", color="red", label="Mauvaise cellule")
plt.plot(x, alpha*np.exp(beta*x), color="red")
#plt.plot(DDreal[rc0],D0[rc0],".", color="forestgreen", label="Bonne cellule")
#plt.plot(DDreal[wc0],D0[wc0],".", color="lightcoral", label="Mauvaise cellule")
ax.set_xlabel("Distance de la position théorique par rapport à l'axe central (m)")
ax.set_ylabel("Distance entre position reconstruite et théorique en xy (m)")
fig.set_size_inches(5.75,4.5)
ax.legend(loc="upper left", edgecolor="k", fancybox=0)
ax.set_ylim(0, 0.002)
save = "/home/audrey/image_presentation/reconstruction/distance_xy_10000.png"
fig.savefig(save, dpi=500)
"""

# error vs volume according to right/wrong cell
[alpha, beta] = regression(np.delete(v0,9)[rc0],np.delete(d0,9)[rc0])
plt.plot(np.delete(v0,9)[rc0], alpha*np.exp(beta*np.delete(v0,9)[rc0]), color="green")
plt.plot(np.delete(v0,9)[rc0],np.delete(d0,9)[rc0],".", color="green", label="Bonne cellule")


[alpha, beta] = regression(np.delete(v0[wc0],9)[wc0],np.delete(d0[wc0],9))
plt.plot(np.delete(v0[wc0],9), alpha*np.exp(beta*np.delete(v0,9[wc0])), color="red")
plt.plot(np.delete(v0[wc0],9),np.delete(d0[wc0],9),".", color="red", label="Mauvaise cellule")

#plt.plot(v1[rc1],d1[rc1],".", color="forestgreen", label="Bonne cellule")
#plt.plot(v1[wc1],d1[wc1],".", color="lightcoral", label="Mauvaise cellule")
#plt.plot(v2[rc2],d2[rc2],".", color="forestgreen", label="Bonne cellule")
#plt.plot(v2[wc2],d2[wc2],".", color="lightcoral", label="Mauvaise cellule")

#plt.plot(v0[rc0],d0[rc0],".", color="forestgreen", label="Bonne cellule")
#plt.plot(v0[wc0],d0[wc0],".", color="lightcoral", label="Mauvaise cellule")

ax.set_xlabel("Volume de la cellule (m³)")
ax.set_ylabel("Distance entre position reconstruite et théorique (m)")
fig.set_size_inches(5.45,4.5)
ax.legend(loc="upper left", edgecolor="k", fancybox=0)
ax.set_ylim(0, 0.004)
save = "/home/audrey/image_presentation/reconstruction/distance_volume_celltype_100000.png"
#fig.savefig(save, dpi=500)

"""
# erreur vs distance z and cells
Z_real = pos_real[:,2]
ZZ = pos_found[:,2]-pos_real[:,2]
ZZ0 = pos_found0[:,2]-pos_real[:,2]
ZZ1 = pos_found1[:,2]-pos_real[:,2]
ZZ2 = pos_found2[:,2]-pos_real[:,2]
plt.plot(Z_real[rc1],ZZ1[rc1],".", color="green", label="Bonne cellule")
plt.plot(Z_real[wc1],ZZ1[wc1],".", color="red", label="Mauvaise cellule")
#plt.plot(Z_real[rc2],ZZ2[rc2],".", color="lightgreen")
#plt.plot(Z_real[wc2],ZZ2[wc2],".", color="lightcoral")
#plt.plot(Z_real[rc0],ZZ0[rc0],".", color="forestgreen", label="Bonne cellule")
#plt.plot(Z_real[wc0],ZZ0[wc0],".", color="lightcoral", label="Mauvaise cellule")
#plt.plot(pos_real[:,2],pos_found0[:,2]-pos_real[:,2],".", label="100 000 itérations de MC", color="tab:gray")

ax.set_xlabel("Hauteur de la position théorique (m)")
ax.set_ylabel("Distance entre position reconstruite et théorique en z (m)")
fig.set_size_inches(5.75,4.5)
ax.legend(loc="upper left", edgecolor="k", fancybox=0)
ax.set_ylim(-0.004, 0.004)
save = "/home/audrey/image_presentation/reconstruction/distance_z_10000.png"
fig.savefig(save, dpi=500)"""


#ax.set_ylim(0, 0.004)
#ig.savefig(save_path + image_format, dpi=500)

print(max(D),max(D0))

plt.show()

n = 0
for i in range(0, len(pos_real)):
    #print(pos_found[i, :], pos_found0[i, :])
    if (np.abs(pos_found[i, 0] - pos_found0[i, 0]) < 1e-6 and
            np.abs(pos_found[i, 1] - pos_found0[i, 1]) < 1e-6 and
            np.abs(pos_found[i, 2] - pos_found0[i, 2]) < 1e-6):
        n += 1

print(f"{n} cellules communes")

