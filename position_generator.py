import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


### Dimensions and parameters ###

# Reactor dimensions
R = 0.1
L = 0.2


# Detector dimensions
#r = L/(2*10)
#l = 2*r

r = 0.0381
l = 0.0762


# Position of the detector (face
#FP = [R, 0, L/2]
#MP = [FP[0]+l/2, FP[1], FP[2]]

FP = [0.15, 0, 0.1]
MP = [0.17, FP[1], FP[2]]

# Number of points per line (even number)
nb = 100

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
zi = FP[2]
xi = R*(1-0.5*k**2)
yi = np.sqrt(d**2 - (xi - R)**2 - (zi - FP[2])**2)

alpha = np.arctan(xi/yi)
alpha0 = np.linspace(-alpha, alpha, nb)


z0 = FP[2]*np.ones(nb)  # z = z_faceposition
x0 = FP[0] - d*np.cos(alpha0)
y0 = d*np.sin(alpha0) + FP[1]

# Plot the line
distancectezcte = ax.plot(x0, y0, z0,  ".", color="red", label="distance & z = constant")

### Set 1 : Constant distance with y = constant ###

# Calculate alpha (trivial for y = cte)
alpha = np.pi/2
alpha1 = np.linspace(-alpha, alpha, nb)

# Calculate half position because of the +/- sqrt for y
y1 = FP[1]*np.ones(nb)  # y = y_faceposition (= 0)
z1 = d*np.sin(alpha1) + FP[2]
x1 = FP[0] - d*np.cos(alpha1)

# Plot the line
distancecteycte = ax.plot(x1, y1, z1,  ".", color="blue", label="distance & y = constant")


### Set 2 : Constant distance with 90 degree cross-section of the cylinder ###

# Calulate alpha
zi = FP[2] - d/2
yi = zi
di = np.sqrt(zi**2 + yi**2)
alpha = np.arccos(d/di)
alpha2 = np.linspace(-alpha, alpha, nb)


# Generate y and z
z2 = d*np.sin(alpha2)/np.sqrt(2) + FP[2]
y2 = z2 - FP[2]

# Solve for x position
a = 1
b = -2*R
c = R**2 + y2**2 + (z2 - FP[2])**2 - d**2
delta = b**2 - 4*a*c
x2 = (-b - np.sqrt(delta))/(2*a)

# Plot the line
distancecte = ax.plot(x2, y2, z2,  ".", color="green", label="distance constant")

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
Yd, Zd, Xd = data_for_cylinder(FP[0],FP[2],r,l)
Xs, Ys, Zs = data_for_sphere(FP[0],FP[1],FP[2],d)

reactor = ax.plot_surface(Xc, Yc, Zc, alpha=0.3, color="grey", label="Reactor")
detector = ax.plot_surface(Xd+R, Yd-R, Zd, alpha=0.3, color="blue", label="Detector")
sphere = ax.plot_surface(Xs, Ys, Zs, alpha=0.05, color="coral", label="Constant distance")


### Plotting ###

# Equal axis for 3d plots
world_limits = ax.get_w_lims()
ax.set_box_aspect((world_limits[1]-world_limits[0],world_limits[3]-world_limits[2],world_limits[5]-world_limits[4]))

ax.set_title("Positions with constant distance")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")


### Set 3 : Constant angles & z ###

theta = np.pi/4
phi = 0

# Max distance for the angle
dmax = 2*R*np.cos(theta)/np.cos(phi)

# Get the unit vector
ptmax = np.array([dmax*np.cos(phi)*np.cos(theta) - FP[0], # Because detector x orientation has opposite sign
         dmax*np.cos(phi)*np.sin(theta) + FP[1],
         dmax*np.sin(phi) + FP[2]])

vector = ptmax - np.array(FP) # Vector of the detector fp to point at max distance

f_max = np.linalg.norm(vector) # Norm = max factor of the unit vector for the distance
unit_vector = vector/f_max

# Generate position at contant angles
f_vector = np.linspace(0, f_max, nb)
positions = np.ones((nb,3))
distance3 = np.ones(nb)
for i in range(0,nb):
    positions[i] = f_vector[i] * unit_vector + np.array(FP) # factor * unit vector + face position detector
    distance3[i] = np.linalg.norm(f_vector[i] * unit_vector)

x3 = positions[:,0]
y3 = positions[:,1]
z3 = positions[:,2]
angleszcte = ax1.plot(x3, y3, z3,  ".", color="red", label="angles & z = constant")


### Set 4 : Constant angles  ###

theta = np.pi/4
phi = theta

# Max distance for the angle
dmax = 2*R*np.cos(theta)/np.cos(phi)

# Get the unit vector
ptmax = np.array([dmax*np.cos(phi)*np.cos(theta) - FP[0], # Because detector x orientation has opposite sign
         dmax*np.cos(phi)*np.sin(theta) + FP[1],
         dmax*np.sin(phi) + FP[2]])

vector = ptmax - np.array(FP) # Vector of the detector fp to point at max distance

f_max = np.linalg.norm(vector) # Norm = max factor of the unit vector for the distance
unit_vector = vector/f_max

# Generate position at contant angles
f_vector = np.linspace(0, f_max, nb)
positions = np.ones((nb,3))
distance4 = np.ones(nb)
for i in range(0,nb):
    positions[i] = f_vector[i] * unit_vector + np.array(FP) # factor * unit vector + face position detector
    distance4[i] = np.linalg.norm(f_vector[i] * unit_vector)

x4 = positions[:,0]
y4 = positions[:,1]
z4 = positions[:,2]
anglescte = ax1.plot(x4, y4, z4,  ".", color="blue", label="angles = constant")


### Set 9 : Constant angles  ###

theta = -np.pi/3
phi = -np.pi/4

# Max distance for the angle
dmax = 2*R*np.cos(theta)/np.cos(phi)

# Get the unit vector
ptmax = np.array([dmax*np.cos(phi)*np.cos(theta) - FP[0], # Because detector x orientation has opposite sign
         dmax*np.cos(phi)*np.sin(theta) + FP[1],
         dmax*np.sin(phi) + FP[2]])

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
anglescte = ax1.plot(x9, y9, z9,  ".", color="black", label="angles = constant but random")


ax1.legend()

# Mesh generation
Xc, Yc, Zc = data_for_cylinder(0,0,R,L)
Yd, Zd, Xd = data_for_cylinder(FP[0],FP[2],r,l)

reactor = ax1.plot_surface(Xc, Yc, Zc, alpha=0.3, color="grey", label="Reactor")
detector = ax1.plot_surface(Xd+R, Yd-R, Zd, alpha=0.3, color="blue", label="Detector")


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
x5 = np.linspace(FP[0], FP[0] - 2*R, nb)

distance5 = x5

ax2.plot(x5, y5, z5,  ".", color="black")

### Set 6 : Case S3, y & z = cte  ###

y6 = FP[1]*np.ones(nb)
z6 = FP[2]*np.ones(nb) + R
x6 = np.linspace(FP[0], FP[0] - 2*R, nb)

distance6 = x6

ax2.plot(x6, y6, z6,  ".", color="black")

### Set 7 : x & z = cte  ###

y7 = np.linspace(-R, R, nb)
z7 = FP[2]*np.ones(nb)
x7 = 0*np.ones(nb)

distance7 = y7

ax2.plot(x7, y7, z7,  ".", color="black")

### Set 8 : x & y = cte  ###

x8 = 0*np.ones(nb)
y8 = FP[1]*np.ones(nb)
z8 = np.linspace(0, L, nb)

distance8 = z8

ax2.plot(x8, y8, z8,  ".", color="black")


# Mesh generation
Xc, Yc, Zc = data_for_cylinder(0,0,R,L)
Yd, Zd, Xd = data_for_cylinder(FP[0],FP[2],r,l)

reactor = ax2.plot_surface(Xc, Yc, Zc, alpha=0.3, color="grey", label="Reactor")
detector = ax2.plot_surface(Xd+R, Yd-R, Zd, alpha=0.3, color="blue", label="Detector")

# Equal axis for 3d plots
world_limits = ax2.get_w_lims()
ax2.set_box_aspect((world_limits[1]-world_limits[0],world_limits[3]-world_limits[2],world_limits[5]-world_limits[4]))


ax2.set_title("Positions moving in one direction")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("z")
plt.show()


### Export positions ###
data = pd.DataFrame(columns=["x_position", "y_position", "z_position", "angle_distance"])
data["x_position"] = np.concatenate((x0, x1, x2, x3, x4, x9, x5, x6, x7, x8), axis=None)
data["y_position"] = np.concatenate((y0, y1, y2, y3, y4, y9, y5, y6, y7, y8), axis=None)
data["z_position"] = np.concatenate((z0, z1, z2, z3, z4, z9, z5, z6, z7, z8), axis=None)
data["angle_distance"] = np.concatenate((alpha0, alpha1, alpha2, distance3, distance4, distance9, distance5, distance6, distance7, distance8), axis=None)


path_export_data = "C:/Users/Acdai/OneDrive - polymtl.ca/Polytechnique/Session E2021/GCH8392 - Projet individuel de génie chimique/Data/positions/"
data.to_csv(path_export_data + "set1.csv", index=False)
