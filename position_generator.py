import numpy as np
import scipy.optimize as scop
import pandas as pd
import matplotlib.pyplot as plt


### Dimensions and parameters ###

# Reactor dimensions
R = 0.1
L = 0.5


# Detector dimensions
r = 0.0381
l = 0.0762

# Position of the detector (face
FP = [0.2, 0, L/2]
MP = [FP[0] + l/2, FP[1], FP[2]]


# Number of points per line (even number)
nb = 25

# Distance between particle and face position of the detector
k = 2  # Factor to be applied to radius of reactor (200%  R)
d = k*R

# Plot initialization
fig = plt.figure()
fig1 = plt.figure()

ax = fig.add_subplot(1,1,1, projection='3d')
ax1 = fig1.add_subplot(1,1,1, projection='3d')

# Function to generate reactor, detector & sphere for constant angle
def data_for_cylinder(center_x,center_y,radius,height_z):
    z = np.linspace(0, height_z, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid=np.meshgrid(theta, z)
    x_grid = radius*np.cos(theta_grid) + center_x
    y_grid = radius*np.sin(theta_grid) + center_y
    return x_grid,y_grid,z_grid

def data_for_sphere(center_x,center_y,center_z,radius):
    u, v = np.mgrid[0:10 * np.pi:100j, 0:np.pi:50j]
    x_grid = radius * np.cos(u) * np.sin(v) + center_x
    y_grid = radius * np.sin(u) * np.sin(v) + center_y
    z_grid = radius *np.cos(v) + center_z
    return x_grid,y_grid,z_grid



### Set0 : Constant distance with z = constant ###

# Calculate alpha
def sphere_cylinder_zcte(val,FPdR):
    x = val[0]
    y = val[1]

    FP = FPdR[:3]
    d = FPdR[3]
    R = FPdR[4]

    return [(x - FP[0])**2 + (y - FP[1])**2 - d**2,
            x**2 + y**2 - R**2]

FPdR = FP.copy()
FPdR.extend([d, R])

[xi, yi] = scop.fsolve(sphere_cylinder_zcte,x0=[d, d], args=FPdR)
alpha_max = np.arctan(np.abs(yi-FP[1])/np.abs(xi-FP[0]))
[xi, yi] = scop.fsolve(sphere_cylinder_zcte,x0=[-d, -d], args=FPdR)
alpha_min = -np.arctan(np.abs(yi-FP[1])/np.abs(xi-FP[0]))

alpha0 = np.linspace(alpha_min, alpha_max, nb)

z0 = FP[2]*np.ones(nb)  # z = z_faceposition
x0 = FP[0] - d*np.cos(alpha0)
y0 = d*np.sin(alpha0) + FP[1]

# Plot the line
ax.plot(x0, y0, z0,  ".", color="xkcd:rouge", label="distance & z = cte")

### Set 1 : Constant distance with y = constant ###

# Calculate alpha
alpha_max = np.arccos((FP[0]-R)/d) # Max alpha angle
z_max_check = d*np.sin(alpha_max) + FP[2]
if z_max_check > L:  # if alpha reaches higher than top of the reactor
    alpha_max = np.arcsin((L-FP[2])/d)

alpha_min = -np.arccos((FP[0]-R)/d)
z_min_check = d*np.sin(alpha_min) + FP[2]
if z_min_check < 0:  # if alpha reaches lower than bottom of the reactor
    alpha_min = np.arcsin((-FP[2])/d)

alpha1 = np.linspace(alpha_min, alpha_max, nb)

# Calculate half position because of the +/- sqrt for y
y1 = FP[1]*np.ones(nb)  # y = y_faceposition (= 0)
z1 = d*np.sin(alpha1) + FP[2]
x1 = FP[0] - d*np.cos(alpha1)

# Plot the line
ax.plot(x1, y1, z1,  ".", color="xkcd:royal blue", label="distance & y = cte")

### Set 2 : Constant distance with 90 degree cross-section of the cylinder ###

# Calculate alpha
def sphere_cylinder_yisz(val,FPdR):
    x = val[0]
    y = val[1]
    z = val[2]

    FP = FPdR[:3]
    d = FPdR[3]
    R = FPdR[4]

    return [(x - FP[0])**2 + (y - FP[1])**2 + (z - FP[2])**2 - d**2,
            x**2 + y**2 - R**2,
            (z - FP[2]) - (y - FP[1])]

[xi, yi, zi] = scop.fsolve(sphere_cylinder_yisz,x0=[FP[0]+d, FP[1]+d, FP[2]+d], args=FPdR)

if zi < 0:
    zi = 0
elif zi > L:
    zi = L

yi = zi + FP[2] + FP[1]
di = np.sqrt(2)*(zi - FP[2])
alpha_max = np.arcsin(di/d)

[xi, yi, zi] = scop.fsolve(sphere_cylinder_yisz,x0=[FP[0]-d, FP[1]-d, FP[2]-d], args=FPdR)

if zi < 0:
    zi = 0
elif zi > L:
    zi = L

yi = zi + FP[2] + FP[1]
di = np.sqrt(2)*(zi - FP[2])
alpha_min = np.arcsin(di/d)

alpha2 = np.linspace(alpha_min, alpha_max, nb)

# Generate y and z
z2 = d*np.sin(alpha2)/np.sqrt(2) + FP[2]
y2 = z2 - FP[2] + FP[1]

# Solve for x position
a = 1
b = -2*R
c = R**2 + y2**2 + (z2 - FP[2])**2 - d**2
delta = b**2 - 4*a*c
x2 = (-b - np.sqrt(delta))/(2*a)
x2 = x2 + (FP[0] - R)

# Plot the line
ax.plot(x2, y2, z2,  ".", color="xkcd:grassy green", label="distance = cte & y = z")
ax.legend()

# Mesh generation
Xc, Yc, Zc = data_for_cylinder(0,0,R,L)
Zd, Yd, Xd = data_for_cylinder(FP[2],FP[1],r,l)
Xs, Ys, Zs = data_for_sphere(FP[0],FP[1],FP[2],d)

reactor = ax.plot_surface(Xc, Yc, Zc, alpha=0.2, color="grey", label="Reactor")
detector = ax.plot_surface(Xd+FP[0], Yd, Zd, alpha=0.5, color="xkcd:black", label="Detector")
sphere = ax.plot_surface(Xs, Ys, Zs, alpha=0.05, color="xkcd:powder blue", label="Constant distance")

### Plotting ###

# Equal axis for 3d plots
ax1.set_xlim(-0.15, 0.4)
ax1.set_ylim(-0.2, 0.2)
ax1.set_zlim(0, 0.5)
world_limits = ax.get_w_lims()
ax.set_box_aspect((world_limits[1]-world_limits[0],world_limits[3]-world_limits[2],world_limits[5]-world_limits[4]))

ax.set_title("Positions des particules ayant une distance constante")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")


### Set 3 : Constant angles  ###

theta = np.pi/3
phi = np.pi/3

# Max distance for the angle (from FP to opposite side of reactor)
dmax = 2*R*np.cos(theta)/np.cos(phi)

xmax = dmax*np.cos(phi)*np.cos(theta) - (FP[0] - (FP[0] - R))  # Because detector x orientation has opposite sign
ymax = dmax*np.cos(phi)*np.sin(theta) + FP[1]
zmax = dmax*np.sin(phi) + FP[2]

dmax = np.linalg.norm([xmax, ymax, zmax])

if zmax > L:
    zmax = L
elif zmax < 0:
    zmax = 0

# New xmax... if zmax was outside of the cylinder
dmax = (zmax - FP[2])/np.sin(phi)
xmax = dmax*np.cos(phi)*np.cos(theta) - (FP[0] - (FP[0] - R))
ymax = dmax * np.cos(phi) * np.sin(theta) + FP[1]

ptmax = np.array([xmax, ymax, zmax])

# Minimum values
def func(t,pt):
    P = pt[:3]
    Q = pt[3:6]
    R = pt[6]

    return (P[0] + t*(Q[0] - P[0]))**2 + (P[1] + t*(Q[1] - P[1]))**2 - R**2

pt = FP.copy()
pt.extend([xmax, ymax, zmax, R])

t = scop.fsolve(func,x0=[-1, 1],args=pt)

xmin = FP[0] + t[0]*(xmax - FP[0])
ymin = FP[1] + t[0]*(ymax - FP[1])
zmin = FP[2] + t[0]*(zmax - FP[2])

ptmin = np.array([xmin, ymin, zmin])

# Get the unit vector
vectormax = ptmax - np.array(FP) # Vector of the detector fp to point at max distance
vectormin = ptmin - np.array(FP)

fmax = np.linalg.norm(vectormax) # Norm = max factor of the unit vector for the distance
fmin = np.linalg.norm(vectormin)
unit_vector = vectormax/fmax

# Generate position at contant angles
f_vector = np.linspace(fmin, fmax, nb)
positions = np.ones((nb,3))
distance3 = np.ones(nb)
for i in range(0,nb):
    positions[i] = f_vector[i] * unit_vector + np.array(FP) # factor * unit vector + face position detector
    distance3[i] = np.linalg.norm(f_vector[i] * unit_vector)

x3 = positions[:,0]
y3 = positions[:,1]
z3 = positions[:,2]
ax1.plot(x3, y3, z3,  ".", color="xkcd:royal purple", label="angle = cte")


### Set 4 : Case S1, y & z = cte  ###

y4 = FP[1]*np.ones(nb)
z4 = FP[2]*np.ones(nb)
x4 = np.linspace(-R, R, nb)

distance4 = x4

ax1.plot(x4, y4, z4,  ".", color="xkcd:pumpkin", label="angle & y, z = cte")
ax1.legend()

reactor = ax1.plot_surface(Xc, Yc, Zc, alpha=0.2, color="grey", label="Reactor")
detector = ax1.plot_surface(Xd+FP[0], Yd, Zd, alpha=0.5, color="xkcd:black", label="Detector")


### Plotting ###

# Equal axis for 3d plots
ax1.set_xlim(-0.15, 0.4)
ax1.set_ylim(-0.2, 0.2)
ax1.set_zlim(0, 0.5)
world_limits = ax1.get_w_lims()
ax1.set_box_aspect((world_limits[1]-world_limits[0],world_limits[3]-world_limits[2],world_limits[5]-world_limits[4]))

ax1.set_title("Positions des particules ayant un angle constant")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("z")

plt.show()


### Export positions ###
data = pd.DataFrame(columns=["x_position", "y_position", "z_position", "angle_distance"])
data["x_position"] = np.concatenate((x0, x1, x2, x3, x4), axis=None)
data["y_position"] = np.concatenate((y0, y1, y2, y3, y4), axis=None)
data["z_position"] = np.concatenate((z0, z1, z2, z3, z4), axis=None)
data["angle_distance"] = np.concatenate((alpha0, alpha1, alpha2, distance3, distance4), axis=None)


path_export_data = "/mnt/DATA/rpt_postprocessing/positions/"
#path_export_data = "C:/Users/Acdai/OneDrive - polymtl.ca/Polytechnique/Session E2021/GCH8392 - Projet individuel de génie chimique/Data/positions/"
#data.to_csv(path_export_data + "positions_counts1.csv", index=False)
