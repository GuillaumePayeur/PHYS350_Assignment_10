# Guillaume Payeur
import numpy as np
import matplotlib.pyplot as plt

# Enabling LaTeX
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({'text.latex.preamble':[r'\usepackage{physics}']})

# Function that returns the vector potential for all locations in grid (r) due
# to a current ring drawn by the points (points) and carrying current (I)
def A(r,I,points):
    A_x = np.zeros((r.shape[1],r.shape[2]))
    A_y = np.zeros((r.shape[1],r.shape[2]))
    A_z = np.zeros((r.shape[1],r.shape[2]))
    for i in range(len(points)):
        l_i = points[i]-points[i-1]
        segment_center = (points[i]+points[i-1])/2
        segment_center = np.expand_dims(segment_center,axis=1)
        segment_center = np.expand_dims(segment_center,axis=2)
        r_i = np.sqrt(np.sum((r-segment_center)**2,axis=0))
        A_x += l_i[0]/r_i
        A_y += l_i[1]/r_i
        A_z += l_i[2]/r_i
    return np.array([A_x,A_y,A_z])

# Building a square loop of size 2 on the xy plane
square = np.array([[1,1,0],[-1,1,0],[-1,-1,0],[1,-1,0]])

# Building a circular loop of radius 1 on the xy plane
npoints = 500
circle = []
for i in range(npoints):
    theta = i*2*np.pi/npoints
    x_i, y_i = np.cos(theta), np.sin(theta)
    circle.append([x_i,y_i,0])
circle = np.array(circle)

# Calculating the potential for all angles on a sphere of radius r
npoints = 100
r = 10
I = 1
theta = np.linspace(0,np.pi,npoints)
phi = np.linspace(0,2*np.pi,2*npoints)

theta_grid,phi_grid = np.meshgrid(theta,phi)
x_grid = r*np.cos(phi_grid)*np.sin(theta_grid)
y_grid = r*np.sin(phi_grid)*np.sin(theta_grid)
z_grid = r*np.cos(theta_grid)
r_grid = np.array([x_grid,y_grid,z_grid])

A_grid = A(r_grid,I,square)

# Function that returns the dipole contribution to the vector potential for all
# locations in grid (r) due to a current ring carrying current (I) and having
# area vector (a)
def A_dip(r,I,a):
    norm_r = np.sqrt(np.sum(r**2,axis=0))
    m = I*a
    m_x = m[0]*np.ones((r.shape[1],r.shape[2]))
    m_y = m[1]*np.ones((r.shape[1],r.shape[2]))
    m_z = m[2]*np.ones((r.shape[1],r.shape[2]))
    m = np.array([m_x,m_y,m_z])
    mxr = np.cross(m,r,axisa=0,axisb=0)
    mxr = np.transpose(mxr, axes=(2,0,1))
    return mxr/(norm_r**3)

# Calculating the dipole component of the potential for all angles on a sphere
# of radius r
a_circle = [0,0,np.pi]
a_square = [0,0,4]
A_dip_grid = A_dip(r_grid,I,a_square)


# Creating the plot
max = max(np.max(A_grid),np.max(A_dip_grid))

fig, axes = plt.subplots(nrows=2, ncols=3)
axes[0,0].set_title('$A_x$')
axes[0,1].set_title('$A_y$')
axes[0,2].set_title('$A_z$')

axes[0,0].set_ylabel('$\\theta$')
im = axes[0,0].imshow(A_grid[0], extent=(0,np.pi,0,2*np.pi),
                                 vmin=-max,vmax=max,
                                 origin='lower')
axes[0,0].set_xlabel('$\phi$')
im = axes[0,1].imshow(A_grid[1], extent=(0,np.pi,0,2*np.pi),
                                 vmin=-max,vmax=max,
                                 origin='lower')
axes[0,1].set_xlabel('$\phi$')
im = axes[0,2].imshow(A_grid[2], extent=(0,np.pi,0,2*np.pi),
                                 vmin=-max,vmax=max, origin='lower')
axes[0,2].set_xlabel('$\phi$')

axes[1,0].set_title('$(A_x)_{\\text{dip}}$')
axes[1,1].set_title('$(A_y)_{\\text{dip}}$')
axes[1,2].set_title('$(A_z)_{\\text{dip}}$')

axes[1,0].set_ylabel('$\\theta$')
im = axes[1,0].imshow(A_dip_grid[0], extent=(0,np.pi,0,2*np.pi),
                                     vmin=-max,vmax=max,
                                     origin='lower')
axes[1,0].set_xlabel('$\phi$')
im = axes[1,1].imshow(A_dip_grid[1], extent=(0,np.pi,0,2*np.pi),
                                     vmin=-max,vmax=max, origin='lower')
axes[1,1].set_xlabel('$\phi$')
im = axes[1,2].imshow(A_dip_grid[2], extent=(0,np.pi,0,2*np.pi),
                                     vmin=-max,vmax=max,
                                     origin='lower')
axes[1,2].set_xlabel('$\phi$')

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.89, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
fig.tight_layout()
plt.show()
