import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches

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

save_path = "/home/audrey/image_presentation/reconstruction/chart_10000"
image_format = ".png"

"""
cells = ["Right cell", "Wrong cell"]
test = np.array([[11, 54, 7, 15], [33, 8, 17, 40]])
data_10000 = np.array([[1, 96, 0, 0], [1, 204, 1, 0]])
data_100000 = np.array([[0, 43, 0, 0], [0, 57, 1, 0]])
size = 2

DATA = data_10000

def plot_pie_chart(data, size):
    # normalizing data to 2 pi
    norm = data / np.sum(data) * 2 * np.pi

    # obtaining ordinates of bar edges
    left = np.cumsum(np.append(0,
                               norm.flatten()[:-1])).reshape(data.shape)

    # Creating color scale
    cmap = plt.get_cmap("RdYlGn")
    outer_colors = cmap(np.array([256, 0]))
    inner_colors = cmap(np.array([234, 212, 190, 168,
                                  23, 44, 66, 88]))

    # Creating plot
    fig, ax = plt.subplots(figsize=(10, 7),
                           subplot_kw=dict(polar=True))

    ax.bar(x=left[:, 0],
           width=norm.sum(axis=1),
           bottom=1 - size,
           height=size,
           color=outer_colors,
           edgecolor='k',
           linewidth=0.5
        ,
           align="edge")

    ax.bar(x=left.flatten(),
           width=norm.flatten(),
           bottom=1 - 2 * size,
           height=size,
           color=inner_colors,
           edgecolor='k',
           linewidth=0.5,
           align="edge")

    #ax.set(title="10 000 itérations de Monte-Carlo pour le calcul des nombres de photons aux noeuds")
    ax.set_axis_off()

    # The slices will be ordered and plotted counter-clockwise.
    pourcentages = [0]*10
    tot = 303
    pourcentages[0] = sum(data[0,:]/tot)
    pourcentages[5] = sum(data[1,:])/tot
    for i in [0, 1, 2, 3]:
        pourcentages[i+1] = data[0, i] / tot
        pourcentages[i+6] = data[1, i] / tot

    pourcentages = np.array(pourcentages) *100
    pc = np.around(pourcentages,1)


    labels = [r"Bonne cellule (" + str(pc[0]) +  "\%)",
              r"Statut 0 (" + str(pc[1]) +  "\%)",
              r"Statut 1 (" + str(pc[2]) +  "\%)",
              r"Statut 2 (" + str(pc[3]) +  "\%)",
              r"Statut 3 (" + str(pc[4]) +  "\%)",
              r"Mauvaise cellule (" + str(pc[5]) +  "\%)",
              r"Statut 4 (" + str(pc[6]) +  "\%)",
              r"Statut 5 (" + str(pc[7]) +  "\%)",
              r"Statut 6 (" + str(pc[8]) +  "\%)",
              r"Statut 7 (" + str(pc[9]) +  "\%)"]

    colors = np.concatenate((outer_colors, inner_colors), axis=0)
    patches = []
    for i in [0, 2, 3, 4, 5, 1, 6, 7, 8, 9]:
        patches.append(mpatches.Patch(color=colors[i], linewidth=0.75))
        patches[-1].set_edgecolor("k")


    fig.legend(handles=patches, labels=labels, loc="lower center", ncol=2, edgecolor="k", fancybox=0)
    fig.set_size_inches(5, 6.25)

    # show plot
    fig.savefig(save_path + image_format, dpi=500)
    plt.close(fig)
    ax.clear()
    plt.show()

plot_pie_chart(DATA, size)

"""
fig, ax = plt.subplots(figsize=(10, 10))
angle = np.linspace(0, 2*np.pi, 9)

x = 0.1*np.cos(angle)
y = 0.1*np.sin(angle)

x = np.append(x, 2*0.1/3*np.cos(angle))
y = np.append(y, 2*0.1/3*np.sin(angle))

x = np.append(x, -0.1)
y = np.append(y, 0)

ax.plot(x, y)


angle2 = np.linspace(np.pi/4, (2+1/4)*np.pi, 5)
xc = 0.1/3*np.cos(angle2)
yc = 0.1/3*np.sin(angle2)

ax.plot(xc, yc)

xl = [0.1*np.sin(1*np.pi/4), xc[-1]]
yl = [0.1*np.cos(1*np.pi/4), yc[-1]]
ax.plot(xl, yl)

xl = [0.1*np.sin(3*np.pi/4), xc[-2]]
yl = [0.1*np.cos(3*np.pi/4), yc[-2]]
ax.plot(xl, yl)

xl = [0.1*np.sin(5*np.pi/4), xc[-3]]
yl = [0.1*np.cos(5*np.pi/4), yc[-3]]
ax.plot(xl, yl)

xl = [0.1*np.sin(7*np.pi/4), xc[-4]]
yl = [0.1*np.cos(7*np.pi/4), yc[-4]]
ax.plot(xl, yl)

xl = [x[2], x[6]]
yl = [y[2], y[6]]
ax.plot(xl, yl)
ax.set_axis_off()
#fig.savefig( "/home/audrey/image_presentation/reconstruction/coarse_meshvect2.eps", format="eps")


size = 1

fig1, ax1 = plt.subplots(figsize=(20/3, 10))
x = [-0.1, 0.1, 0.1, -0.1, -0.1]
z = [0, 0, 0.3, 0.3, 0]
ax1.plot(x, z, color="k", linewidth=size)

x_line = [-0.1, 0.1]
ax1.plot(x_line, [0.05, 0.05], color="k", linewidth=size)
ax1.plot(x_line, [0.1, 0.1], color="k", linewidth=size)
ax1.plot(x_line, [0.15, 0.15], color="k", linewidth=size)
ax1.plot(x_line, [0.20, 0.20], color="k", linewidth=size)
ax1.plot(x_line, [0.25, 0.25], color="k", linewidth=size)

ax1.plot([2*-0.1/3, 2*-0.1/3], [0, 0.3], color="k", linewidth=size)
ax1.plot([2*0.1/3, 2*0.1/3], [0, 0.3], color="k", linewidth=size)

ax1.plot([(-0.1/3)*np.sin(np.pi/3), (-0.1/3)*np.sin(np.pi/3)], [0, 0.3], color="k", linewidth=size)
ax1.plot([(0.1/3)*np.sin(np.pi/3), (0.1/3)*np.sin(np.pi/3)], [0, 0.3], color="k", linewidth=size)
ax1.plot([0, 0], [0, 0.3], color="k")





ax1.set_axis_off()

fig1.savefig( "/home/audrey/image_presentation/reconstruction/coarse_mesh_sidexy.png", dpi=500)

"""

data = pd.read_csv("/mnt/DATA/rpt_postprocessing/grid_counts.csv")
value = np.array([])
value1 = np.array([])
value2 = np.array([])

for i in range(0, data.shape[0]):
    if (data["particle_positions_x"][i] == 0):
        value = np.append(value, data["counts_it1000"][i])
        value1 = np.append(value1, data["counts_it10001"][i])
        value2 = np.append(value2, data["counts_it10002"][i])

"""
plt.show()