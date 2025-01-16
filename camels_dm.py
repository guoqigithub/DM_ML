import pygad as pg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import caesar
import math
import sys
import itertools
import emcee

import tenet
from tenet.util.simParams import simParams
from tenet.cosmo.spectrum import generate_rays_voronoi_fullbox,_integrate_quantity_along_traced_rays
from scipy import spatial
from scipy import interpolate
from scipy.special import ndtr
from scipy.stats import norm, gaussian_kde
import astropy.units as u
import astropy.cosmology.units as cu
from astropy.cosmology import WMAP9
from astropy.cosmology import FlatLambdaCDM
from multiprocessing import Pool
import astropy.constants as const
import camels_library as CL
from config.pygad_init import SIMBA_pygad_init
SIMBA_pygad_init()


def DM_SIMBA(name_X,name_Y,snap):
    snapshot = "./SIMBA/1P/1P_%d_%d/snap_%03d.hdf5"%(name_X,name_Y,snap)
    N = 256
    s = pg.Snapshot(snapshot)
    h0 = s.cosmology.h_0
    n_e_ = CL.electron_density(snapshot)
    e_density = n_e_*h0**2
    boxsize = 2.5e4
    m = pg.binning.SPH_to_3Dgrid(s.gas, e_density, extent=pg.UnitArr([[0,boxsize],[0,boxsize],[0,boxsize]],'ckpc h_0**-1'), Npx=[N,N,N])

    DM_2D = np.array(((m.grid).sum(axis=2)))
    DM_map = DM_2D.reshape(N**2)*2.5e7/0.6711/256

    np.save("./SIMBA/1P/1P_%d_%d/DM_snap%03d.npy"%(name_X,name_Y,snap),DM_map)

def DM_TNG(name_X,name_Y,z):
    sim = tenet.sim('./TNG/1P/1P_1_0/', redshift=redshift[i])
    rays = h5py.File("./TNG/1P/1P_1_0/data.files/rays/voronoi_fullbox_n%dd2_%03d.hdf5"%(nRaysPerDim,snapshot_num[i]))
    rays_off, rays_len, rays_dl, rays_inds, cell_inds, ray_pos, ray_dir, total_dl = generate_rays_voronoi_fullbox(sim,nRaysPerDim=nRaysPerDim)
    nRaysPerDim = 512
    generate_rays_voronoi_fullbox(sim,nRaysPerDim)

    # projAxis = list(ray_dir).index(1)
    # cell_values = n_e[cell_inds]
    # rays_dl = sim.units.codeLengthToPc(rays_dl)
    # result = _integrate_quantity_along_traced_rays(rays_off, rays_len, rays_dl, rays_inds, cell_values)

def redshift_space(Omega_m):
    cosmo = FlatLambdaCDM(H0=67.11,Om0=Omega_m, Ob0=0.048)
    d = np.arange(25,3825,25)*u.Mpc
    redshift_space = np.zeros(len(d))
    for i in range(len(d)):
        redshift_space[i]=d[i].to(cu.redshift, cu.redshift_distance(cosmology=cosmo, kind="comoving")).value 
    return redshift_space
    
def LH_DM(name='1P_1_0'):   
    z = np.array([1.05,0.95,0.86,0.77,0.69,0.61,0.54,0.47,0.40,0.34,0.27,0.21,0.15,0.10,0.05,0.00])
    CDF_tar = np.arange(0,1,0.001)
    DM_z = np.zeros([16,len(CDF_tar)])
    for snap in range(18,34):
        DM = np.load("./TNG/1P/1P_1_0/DM_n256_snap%03d.npy"%(snap))
        DM = DM[DM!=0]
        H,X1 = np.histogram(DM,bins = 200,normed = True)
        dx = X1[1] - X1[0]
        F1 = np.cumsum(H)*dx
        F1_ = np.zeros(len(F1)+1)
        F1_[1:] = F1
        inversefunction = interpolate.interp1d(F1_, X1)
        DM_z[snap-18,:] = inversefunction(CDF_tar)
    f = interpolate.interp1d(z,DM_z,axis=0, kind='cubic', bounds_error=False)
    redshift = redshift_space(Omega_m=0.3)
    DM_map_ = f(redshift)
    func = []
    for i in range(len(DM_map_)):
        func.append(interpolate.interp1d(CDF_tar, DM_map_[i,:], kind='cubic', bounds_error=False))
    L_arr=np.arange(25,3825,25)
    result = pd.DataFrame({'Name' : [],'L' : [],'redshift' : [],'DM' : []})
    for L in L_arr:
        DM_map=np.zeros(90000)
        for j in range(len(DM_map)):
            i=0
            while i<L/25:
                DM_map[j]+=func[i](np.random.rand(1)*0.999)*(1+redshift[i])
                i+=1
        catalog = pd.DataFrame({'Name' : [name],'L' : [L],'redshift' : [redshift[np.int64(L/25)-1]],'DM' : [(DM_map[DM_map>0]).tolist()]})
        result=pd.concat([result,catalog])    
    result.to_csv("./TNG/1P/diff_z/1P_1_0_z1.csv")


def f(x,alpha, beta, C):
    y = np.exp(x)/mean
    return y**(-beta) * np.exp(-((y**(-alpha) - C)**2) / (2 * alpha**2 * sigma**2))*y

def log_likelihood(theta, x, y):
    alpha, beta, C = theta
    integral_f_x, _ = quad(lambda x: f(x, alpha, beta, C), mini, maxi)
    if integral_f_x > 0  and integral_f_x != np.inf:
        A = 1/integral_f_x
        model = A*f(x, alpha, beta, C)
        return -0.5 * np.sum((y - model)**2)
    return -np.inf

def log_prior(theta):
    alpha, beta, C = theta
    if 0<alpha and 0<beta:
        integral_f_x, _ = quad(lambda x: f(x, alpha, beta, C), mini, maxi)
        if integral_f_x > 0 and integral_f_x != np.inf:
            A = 1 / integral_f_x
            integral_x_f_x, _ = quad(lambda x: A * np.exp(x)/mean * f(x ,alpha, beta, C), mini, maxi)
            if np.abs(integral_x_f_x - 1) > 0.95:
                return -np.inf
            return -(integral_x_f_x - 1)**2*1200
        return -np.inf
    return -np.inf  

def log_posterior(theta, x, y):
    log_prior_value = log_prior(theta)
    if not np.isfinite(log_prior_value):
        return -np.inf
    return log_prior_value + log_likelihood(theta, x, y)

def MCMC_fitting(name):
    data = pd.read_csv("./TNG/LH/diff_z/DM_LH_%d.csv"%(name))
    alpha = []
    beta = []
    C0 = []
    for i in range(len(data)):
        test1 = np.array(eval(data['DM'][i]))
        if len(test1)>0:
            test1 = test1[test1!=0]
            global sigma,mean
            mean = np.mean(test1)
            sigma = np.std(test1/mean)
        
            test = np.log(test1)
            kde = gaussian_kde(test)
        
            global mini,maxi
            mini = np.max([np.min(test),np.mean(test)-4*np.std(test)])
            maxi = np.min([np.max(test),np.mean(test)+5*np.std(test)])
        
            x = np.linspace(np.max([np.min(test),np.mean(test)-4*np.std(test)]),np.min([np.max(test),np.mean(test)+5*np.std(test)]), 200)
            y = kde.evaluate(x)
        
            ndim = 3
            nwalkers = 100  
            nburn = 1000  
            nsteps = 1000 
        
            p0 = np.random.rand(nwalkers, ndim)+3
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(x, y))
            sampler.run_mcmc(p0, nburn + nsteps, progress=True)
        
            samples = sampler.chain[:, nburn:, :].reshape(-1, ndim)
        
            mcmc_params = np.median(samples, axis=0)
    
            alpha.append(mcmc_params[0])
            beta.append(mcmc_params[1])
            C0.append(mcmc_params[2])
        else:
            alpha.append(np.nan)
            beta.append(np.nan)
            C0.append(np.nan)

    data['alpha'] = alpha
    data['beta'] = beta
    data['C0'] = C0
    data.to_csv("./TNG/LH/MCMC_fitting/DM_fitting_LH_%d_v4.csv"%(name))