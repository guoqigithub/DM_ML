import pygad as pg
import pandas as pd
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import os
import h5py
import caesar
import math
import sys
from baryons_simba import plotting_methods
from pygadgetreader import readsnap, readheader
import time
import glob
import emcee
import camels_library as CL

from scipy import spatial
from scipy.ndimage import gaussian_filter
from scipy.integrate import quad
from scipy import interpolate
from scipy.special import ndtr
from scipy.stats import norm, gaussian_kde

import astropy.constants as const
from astropy.io import ascii
from astropy.io import fits
from astropy.cosmology import WMAP9
from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import Planck15 as cosmo
from astropy.coordinates import SkyCoord
from astropy import units
from astropy import units as u
from astropy.table import unique
from astropy import table
from astropy.table import Table
from astropy.constants import m_p, M_sun

import yt
from yt.units import Mpc
from yt.units import kpc
from yt.data_objects.time_series import DatasetSeries

from mpi4py import MPI
from multiprocessing import Pool
from multiprocessing import Process, current_process
from config.pygad_init import SIMBA_pygad_init
SIMBA_pygad_init()


m_H = const.m_p.value               #kg
X_H = 0.7611

#compute DM
def LH_DM(name):
    for snap in range(21,24):
        if not os.path.exists("./DM/LH_DM/snapshot_%d/DM_LH_%d_SPH_re.npy"%(snap,name)):
            if not os.path.exists("./DM/LH_DM/snapshot_%d"%(snap)):
                os.mkdir("./DM/LH_DM/snapshot_%d"%(snap))
            snapshot = "../LH_Snapshot/LH_%d/snap_%03d.hdf5"%(name,snap)
            N = 256 
            s = pg.Snapshot(snapshot)
            h0 = s.cosmology.h_0
            n_e_ = CL.electron_density(snapshot)     #h^2cm^-3 units
            # in electrons/cm*3 units
            e_density = n_e_*h0**2
            boxsize = 2.5e4
            m = pg.binning.SPH_to_3Dgrid(s.gas, e_density, extent=pg.UnitArr([[0,boxsize],[0,boxsize],[0,boxsize]],'ckpc h_0**-1'), Npx=[N,N,N])

            NE_2D = np.array(((m.grid).sum(axis=2)))
            NE_map = NE_2D.reshape(N**2)

            np.save("./DM/LH_DM/snapshot_%d/DM_LH_%d_SPH_re.npy"%(snap,name),NE_map)

#compute DM

def LineBin(start, end, bins=256): # work for multiple rays at the same time
    if len(start.shape)==1:
        start = np.expand_dims(start,0)
        end = np.expand_dims(end,0)
    ray_num = len(start)
    bin_pos = np.zeros((ray_num, bins, 3))
    bin_num = np.arange(bins)
    na = (-2*bin_num+2*bins+1) / (2*bins)
    nb = (2*bin_num-1) / (2*bins)
    for i in range(3):
        bin_pos[:,:,i] = np.dot(np.expand_dims(start[:,i],1), np.expand_dims(na,0)) + \
        np.dot(np.expand_dims(end[:,i],1), np.expand_dims(nb,0))
    return bin_pos

def ray_gen(ray_num, ray_len, box_len=25000, h=0.6711):
    start_arr = np.array([])
    end_arr = np.array([])
    while len(start_arr)<ray_num:
        start, direc= box_len * np.random.rand(2,3)
        length = np.sqrt(sum((direc - start)**2))
        end = start + (direc - start)*ray_len.to(units.kpc/h).value / length
        if sum(end<box_len) + sum(end>0) == 6:
            start_arr = np.append(start_arr, start).reshape((-1,3))
            end_arr = np.append(end_arr, end).reshape((-1,3))
    return start_arr, end_arr

def dm_calc_1snap_writenpy(snap_add, start, end, redshift, bins=256, h=0.6711):
    
    snap = h5py.File(snap_add)
    
    if len(start.shape)==1:
        start = np.expand_dims(start,0)
        end = np.expand_dims(end,0)
    ray_num = len(start)
    ray_len = np.sqrt(np.sum((end-start)**2, axis=1))
    bin_pos = LineBin(start, end, bins=bins)
    
    sfr = np.array(snap['PartType0']['StarFormationRate'])
    sfg = (sfr!=0)
    
    cell_coord = np.array(snap['PartType0']['Coordinates'])
    tree = sp.KDTree(cell_coord)
    dd, ii = tree.query(bin_pos, k=1)
    
    e_ab = np.array(snap['PartType0']['ElectronAbundance'])
    rho = np.array(snap['PartType0']['Density']) * (10**10 * M_sun/h) * (units.kpc/h)**(-3)
    X_H = 0.76
    n_H = X_H/m_p * rho
    n_e = e_ab * n_H # in comoving coordinate
    n_e_prop = n_e * (1+redshift)**3 # in proper coordinates

    mass = np.array(snap['PartType0']['Masses'])
    density = np.array(snap['PartType0']['Density'])
    volumn = mass/density
    cell_size = volumn**(1/3)  # assume cube cell
    cell_radius = (3/(4*np.pi))**(1/3) * volumn**(1/3)  # assume spherical cell 
    
    passed = (dd<cell_radius[ii]) & (~sfg[ii])
    ray_len_expand = np.expand_dims(ray_len, 1).repeat(bins, axis=1)
    ray_len_expand_prop = ray_len_expand /(1+redshift) # bin length in proper coordinates
    DM = np.sum(n_e_prop[ii]*passed/(1+redshift) * (ray_len_expand_prop/bins)*(units.kpc/h), axis=1) 
    
    return np.array(DM.to((units.pc)*(units.cm)**(-3)).value)

#MCMC fitting
def f1(x, alpha, beta, C):
    return x**(-beta) * np.exp(-((x**(-alpha) - C)**2) / (2 * alpha**2 * sigma**2))

def f(x,alpha, beta, C):
    y = np.exp(x)/mean
    return y**(-beta) * np.exp(-((y**(-alpha) - C)**2) / (2 * alpha**2 * sigma**2))*y

def log_likelihood(theta, x, y):
    alpha, beta, C = theta
    integral_f_x, _ = quad(lambda x: f1(x, alpha, beta, C), 0, np.inf)
    if integral_f_x > 0  and integral_f_x != np.inf:
        A = 1/integral_f_x
        model = A*f(x, alpha, beta, C)
        return -0.5 * np.sum((y - model)**2)
    return -np.inf

def log_prior(theta):
    alpha, beta, C = theta
    if 0<alpha and 0<beta:
        integral_f_x, _ = quad(lambda x: f1(x, alpha, beta, C), 0, np.inf)
        if integral_f_x > 0 and integral_f_x != np.inf:
            A = 1 / integral_f_x
            integral_x_f_x, _ = quad(lambda x: A * np.exp(x)/mean * f(x ,alpha, beta, C), mini, maxi)
            if np.abs(integral_x_f_x - 1) > 0.5:
                return -np.inf
            return -(integral_x_f_x - 1)**2*1200
        return -np.inf
    return -np.inf  

def log_posterior(theta, x, y):
    log_prior_value = log_prior(theta)
    if not np.isfinite(log_prior_value):
        return -np.inf
    return log_prior_value + log_likelihood(theta, x, y)

def MCMC_fiiiting(name):
    test1 = np.load("./DM/LH_snap33_exSFR/DM_LH_%d_SPH.npy"%(name))*2.5e7/0.6711/256   
    test1 = test1[test1!=0]
    global sigma,mean
    sigma = np.std(test1)
    mean = np.mean(test1)

    test = np.log(test1)
    kde = gaussian_kde(test)

    global mini,maxi
    mini = np.min(test)
    maxi = np.max(test)

    x = np.linspace(np.max([np.min(test),np.mean(test)-3*np.std(test)]),np.min([np.max(test),np.mean(test)+5*np.std(test)]), 2000)
    y = kde.evaluate(x)

    ndim = 3
    nwalkers = 100  
    nburn = 1000  
    nsteps = 5000 

    p0 = np.random.rand(nwalkers, ndim)+3
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(x, y))
    sampler.run_mcmc(p0, nburn + nsteps, progress=True)

    samples = sampler.chain[:, nburn:, :].reshape(-1, ndim)

    mcmc_params = np.median(samples, axis=0)

    return mcmc_params

#diff z
Omega_arr=np.loadtxt("SIMBA_para.txt",usecols=(1))[0:1000]

def redshift_space(Omega_m,d):
    cosmo = FlatLambdaCDM(H0=67.11,Om0=Omega_m, Ob0=0.048)
    redshift_space = np.zeros(len(d))
    for i in range(len(d)):
        redshift_space[i]=d[i].to(cu.redshift, cu.redshift_distance(cosmology=cosmo, kind="comoving")).value 
    return redshift_space
    
def LH_DM(name):   
    z = np.array([1.05,0.95,0.86,0.77,0.69,0.61,0.54,0.47,0.40,0.34,0.27,0.21,0.15,0.10,0.05,0.00])
    
    CDF_tar = np.arange(0,1,0.001)
    DM_z = np.zeros([16,len(CDF_tar)])
    
    for snap in range(34):
        DM = np.load("./DM/LH_DM/snapshot_%d/DM_LH_%d_SPH.npy"%(snap,name))*2.5e7/0.6711/256
        DM = DM[DM!=0]
        
        H,X1 = np.histogram(DM,bins = 100,normed = True)
        dx = X1[1] - X1[0]
        F1 = np.cumsum(H)*dx
        F1_ = np.zeros(len(F1)+1)
        F1_[1:] = F1
        inversefunction = interpolate.interp1d(F1_, X1)
        
        DM_z[snap-18,:] = inversefunction(CDF_tar)
    
    f = interpolate.interp1d(z,DM_z,axis=0, kind='cubic', bounds_error=False)

    z_tar = 1* cu.redshift
    cosmo = FlatLambdaCDM(H0=67.11,Om0=Omega_arr[name], Ob0=0.048)
    d = z_tar.to(u.Mpc, cu.redshift_distance(cosmo, kind="comoving")).value
    redshift = redshift_space(Omega_arr[name],np.arange(25,(np.int(d/25)+1)*25,25)*u.Mpc)
    DM_map_ = f(redshift)
    
    func = []
    for i in range(len(DM_map_)):
        func.append(interpolate.interp1d(CDF_tar, DM_map_[i,:], kind='cubic', bounds_error=False))
    
    L_arr = np.random.random(1000000)*(d-25)
    result = pd.DataFrame({'Name' : [],'L' : [],'DM' : []})
    redshift_tar = redshift_space(Omega_arr[name],L_arr*u.Mpc)

    for L_seq in range(len(L_arr)):
        L = L_arr[L_seq]
        DM = 0
        for i in range(np.int64(L/25)):
            DM += func[i](np.random.rand(1))*(1+redshift[i])
        DM += func[i+1](np.random.rand(1))*(1+redshift_tar[L_seq])
        catalog = pd.DataFrame({'Name' : [name],'L' : [L],'DM' : [DM[0]]})
        result=pd.concat([result,catalog])
    result['redshift'] = redshift_tar
    result.to_csv("./DM/LH_DM/diff_z/LH_%d_re_test.csv"%(name))
    