from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def xyz(r, theta, phi, x0=0, y0=0, z0=0):
    x = x0 + r * np.cos(phi) * np.sin(theta)
    y = y0 + r * np.sin(phi) * np.sin(theta)
    z = z0 + r * np.cos(theta)
    return x,y,z


def generate_spheres(n_spheres=40, n_points=10000):
    cloud = []
    for i in range(n_spheres):
        r_max = np.random.rand()
        x0 = np.random.rand() * 10
        y0 = np.random.rand() * 10
        z0 = np.random.rand() * 10
        r  = np.random.rand(n_points) * r1
        theta = np.random.rand(n_points) * np.pi
        phi = np.random.rand(n_points) * np.pi *2

        x, y, z = xyz(r, theta, phi, x0, y0, z0)
        cloud.append(np.array([x,y,z]))
        
    return cloud
        
    
    
if __name__ == '__main__':
    n_obj = 40
    n_points = 10000
    cloud = generate_spheres(n_obj, n_points)
    
    tab = Table(data=cloud)
    write_table_hdf5(tab, 'cloud.h5', path='spheres', overwrite=True)
    
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')

    for sphere in cloud:
        ax.scatter(*sphere)
