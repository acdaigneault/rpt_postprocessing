import numpy as np
import scipy.optimize as scop
import pandas as pd
import matplotlib.pyplot as plt


### Dimensions and parameters ###

###
L = 0.2
r = 0.0381
l = 0.0762
FP = [0.15, 0, 0.1]
MP = [0.17, FP[1], FP[2]]

# Reactor dimensions
R = 0.1
#L = 0.5


# Detector dimensions
#r = L/(2*10)
#l = 2*r

# Position of the detector (face
#FP = [R, 0, L/2]
#MP = [FP[0]+l/2, FP[1], FP[2]]


# Number of points per line (even number)
nb = 10

# Distance between particle and face position of the detector
k = 1.75  # Factor to be applied to radius of reactor (75% + R)
d = k*R


# Plot initialization
fig = plt.figure()

ax = fig.add_subplot(1, 3, 1, projection='3d')
ax1 = fig.add_subplot(1, 3, 2, projection='3d')
ax2 = fig.add_subplot(1, 3, 3, projection='3d')


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
ax.plot(x0, y0, z0,  ".", color="red", label="distance & z = constant")

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
ax.plot(x1, y1, z1,  ".", color="blue", label="distance & y = constant")

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

print(xi, yi, zi)

[xi, yi, zi] = scop.fsolve(sphere_cylinder_yisz,x0=[FP[0]-d, FP[1]-d, FP[2]-d], args=FPdR)


if zi < 0:
    zi = 0
elif zi > L:
    zi = L

yi = zi + FP[2] + FP[1]
di = np.sqrt(2)*(zi - FP[2])
alpha_min = np.arcsin(di/d)

print(xi, yi, zi)


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
ax.plot(x2, y2, z2,  ".", color="green", label="distance constant")

ax.legend()


### Generating reactor, detector and sphere representing constant distance
# from face position of the detector ###

# Cylinder mesh function
def data_for_cylinder(center_x,center_y,radius,height_z):
    z = np.linspace(0, height_z, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid=np.meshgrid(theta, z)
    x_grid = radius*np.cos(theta_grid) + center_x
    y_grid = radius*np.sin(theta_grid) + center_y
    return x_grid,y_grid,z_grid

# Sphere mesh function
def data_for_sphere(center_x,center_y,center_z,radius):
    u, v = np.mgrid[0:10 * np.pi:100j, 0:np.pi:50j]
    x_grid = radius * np.cos(u) * np.sin(v) + center_x
    y_grid = radius * np.sin(u) * np.sin(v) + center_y
    z_grid = radius *np.cos(v) + center_z
    return x_grid,y_grid,z_grid

# Mesh generation
Xc, Yc, Zc = data_for_cylinder(0,0,R,L)
Zd, Yd, Xd = data_for_cylinder(FP[2],FP[1],r,l)
Xs, Ys, Zs = data_for_sphere(FP[0],FP[1],FP[2],d)

reactor = ax.plot_surface(Xc, Yc, Zc, alpha=0.3, color="grey", label="Reactor")
detector = ax.plot_surface(Xd+FP[0], Yd, Zd, alpha=0.3, color="blue", label="Detector")
sphere = ax.plot_surface(Xs, Ys, Zs, alpha=0.05, color="coral", label="Constant distance")


### Plotting ###

# Equal axis for 3d plots
world_limits = ax.get_w_lims()
ax.set_box_aspect((world_limits[1]-world_limits[0],world_limits[3]-world_limits[2],world_limits[5]-world_limits[4]))

ax.set_title("Positions with constant distance")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")


### Set 9 : Constant angles  ###

theta = -np.pi/3
phi = -np.pi/4

bigR = FP[0] # Radius of center of reactor to face position of detector

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


dmax = (zmax - FP[2])/np.sin(phi)
xmax = dmax*np.cos(phi)*np.cos(theta) - (FP[0] - (FP[0] - R))
ymax = dmax * np.cos(phi) * np.sin(theta) + FP[1]

ax1.plot(xmax, ymax, zmax, '.', color="red")

print('phi : ', phi, np.arcsin((zmax - FP[2])/(dmax)))

"""
dmax0 = 0
while dmax0 != dmax:

    if xmax > (bigR+R)*np.cos(theta)*np.cos(theta) - FP[0]:
        xmax = (bigR+R)*np.cos(theta)**2 - FP[0]
        dmax = (xmax + FP[0])/np.cos(phi)*np.cos(theta)

    if ymax > (bigR+R)*np.cos(theta)*np.sin(theta) + FP[1]:
        ymax = (bigR+R)*np.cos(theta)*np.sin(theta) + FP[1]
        dmax = (ymax - FP[1]) / np.cos(phi) * np.sin(theta)

    zmax = dmax * np.sin(phi) + FP[2]

    dmax0 = dmax

"""

a = (xmax-FP[0])/(ymax-FP[1])
xmin = np.sqrt()

# Get the unit vector
ptmax = np.array([xmax, ymax, zmax])

vector = ptmax - np.array(FP) # Vector of the detector fp to point at max distance

f_max = np.linalg.norm(vector) # Norm = max factor of the unit vector for the distance
unit_vector = vector/f_max

# Generate position at contant angles
f_vector = np.linspace(0, f_max, nb)
positions = np.ones((nb,3))
distance9 = np.ones(nb)
for i in range(0,nb):
    positions[i] = f_vector[i] * unit_vector + np.array(FP) # factor * unit vector + face position detector
    distance9[i] = np.linalg.norm(f_vector[i] * unit_vector)

x9 = positions[:,0]
y9 = positions[:,1]
z9 = positions[:,2]
ax1.plot(x9, y9, z9,  ".", color="black", label="angles = constant but random")


ax1.legend()


reactor = ax1.plot_surface(Xc, Yc, Zc, alpha=0.3, color="grey", label="Reactor")
detector = ax1.plot_surface(Xd+FP[0], Yd, Zd, alpha=0.3, color="blue", label="Detector")


### Plotting ###

# Equal axis for 3d plots
world_limits = ax1.get_w_lims()
ax1.set_box_aspect((world_limits[1]-world_limits[0],world_limits[3]-world_limits[2],world_limits[5]-world_limits[4]))

ax1.set_title("Positions with constant angle")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("z")

### Set 5 : Case S1, y & z = cte  ###

y5 = FP[1]*np.ones(nb)
z5 = FP[2]*np.ones(nb)
x5 = np.linspace(-R, R, nb)

distance5 = x5

ax2.plot(x5, y5, z5,  ".", color="black")


reactor = ax2.plot_surface(Xc, Yc, Zc, alpha=0.3, color="grey", label="Reactor")
detector = ax2.plot_surface(Xd+FP[0], Yd, Zd, alpha=0.3, color="blue", label="Detector")

# Equal axis for 3d plots
world_limits = ax2.get_w_lims()
ax2.set_box_aspect((world_limits[1]-world_limits[0],world_limits[3]-world_limits[2],world_limits[5]-world_limits[4]))


ax2.set_title("Positions moving in one direction")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("z")
plt.show()


### Export positions ###
#data = pd.DataFrame(columns=["x_position", "y_position", "z_position", "angle_distance"])
#data["x_position"] = np.concatenate((x0, x1, x2, x3, x4, x9, x5, x6, x7, x8), axis=None)
#data["y_position"] = np.concatenate((y0, y1, y2, y3, y4, y9, y5, y6, y7, y8), axis=None)
#data["z_position"] = np.concatenate((z0, z1, z2, z3, z4, z9, z5, z6, z7, z8), axis=None)
#data["angle_distance"] = np.concatenate((alpha0, alpha1, alpha2, distance3, distance4, distance9, distance5, distance6, distance7, distance8), axis=None)


#path_export_data = "/mnt/DATA/rpt_postprocessing/positions"
#path_export_data = "C:/Users/Acdai/OneDrive - polymtl.ca/Polytechnique/Session E2021/GCH8392 - Projet individuel de génie chimique/Data/positions/"
#data.to_csv(path_export_data + "positions.csv", index=False)
