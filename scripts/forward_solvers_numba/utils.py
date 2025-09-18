import numpy as np
import math
import scipy.interpolate    

def generate_GCOORD(lx, ly, nx, ny):

    x_coords = np.linspace(-lx / 2, lx / 2, nx)
    y_coords = np.linspace(-ly / 2, ly / 2, ny)
    
    xv, yv = np.meshgrid(x_coords, y_coords)
    GCOORD = np.vstack([xv.ravel(), yv.ravel()]).T
    
    return GCOORD

def assemble_EL_connectivity(nel, nnodel, nex, nx):
    EL2NOD = np.zeros((nel,nnodel), dtype=int)

    for iel in range(0,nel):
        row = iel//nex   
        ind = iel + row
        EL2NOD[iel,:] = np.array([ind, ind+1, ind+nx+1, ind+nx])
        
    return EL2NOD

def interpolate_pts(known_pts, known_vals, interp_pts):

    interp_vals = scipy.interpolate.griddata(known_pts, known_vals, interp_pts, method='linear', fill_value=1.)

    for i in range(len(interp_pts)):
        curr_pt = interp_pts[i]
        dist = math.sqrt(curr_pt[0]**2 + curr_pt[1]**2)
        if dist >= 1:
            interp_vals[i] = 1.

    return interp_vals

def central_crop(img, crop_size):
    h, w = img.shape
    startx = w // 2 - (crop_size // 2)
    starty = h // 2 - (crop_size // 2)
    return img[startx:startx+crop_size, starty:starty+crop_size]