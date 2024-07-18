#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import time
import sys
import json
from datetime import datetime
sys.path.append("/global/homes/s/seanjx/gigalens/src")

from gigalens.jax.inference import ModellingSequence
from gigalens.jax.model import ForwardProbModel, BackwardProbModel
from gigalens.jax.simulator import LensSimulator
from gigalens.simulator import SimulatorConfig
from gigalens.model import PhysicalModel

import tensorflow_probability.substrates.jax as tfp

import jax
from jax import random
from jax import numpy as jnp

import numpy as np
import optax
from matplotlib import pyplot as plt

from astropy.io import fits
from astropy.visualization import simple_norm

import myfunctions
from myfunctions import printToFile

from corner import corner

tfd = tfp.distributions

# In[2]:


#create output directory
now = "246_"+str(datetime.now())

path = "output/"+now+"/"
os.makedirs(path)

# In[3]:


#load observation data, do masking
f=fits.open('psf246.fits') 
psf=jnp.array(f[0].data)

observed_img = np.load("cutout246.npy")

f=fits.open('final_96_drz.fits')
background_rms=0.008
exp_time=f[0].header["EXPTIME"]
deltaPix = f[0].header["D002SCAL"]
numPix = np.shape(observed_img)[0]

err_map = np.sqrt(background_rms**2 + observed_img/exp_time)
threshold_lens=1.
error_masked=err_map
#error_masked[45:65,45:65]=np.where(observed_img[45:65,45:65]>threshold_lens, 120000, error_masked[45:65,45:65]) #mask the lens
#error_masked[5:15,75:85]=np.where(observed_img[5:15,75:85]>0.3, 120000, error_masked[5:15,75:85])

plt.figure(figsize=(12,4))
norm = simple_norm(psf, 'sqrt', percent=99.)
plt.subplot(131)
plt.imshow(psf, norm=norm, origin='lower', cmap='viridis')


norm = simple_norm(observed_img, 'sqrt', percent=99.)
plt.subplot(132)
plt.imshow(observed_img, norm=norm, origin='lower', cmap='viridis')
plt.colorbar()

plt.subplot(133)
plt.imshow(err_map, vmin=0,vmax=0.08, origin='lower')
plt.colorbar()

# In[4]:


prior, phys_model, phys_model_Forward = myfunctions.readJson("prior.json")

sim_config = SimulatorConfig(delta_pix=deltaPix, num_pix=numPix, supersample=1, kernel=psf)
lens_sim = LensSimulator(phys_model, sim_config, bs=1)

prob_model = BackwardProbModel(prior, observed_img, error_masked, background_rms=background_rms, exp_time=exp_time)
model_seq = ModellingSequence(phys_model, prob_model, sim_config)

# In[43]:


priorObjects = myfunctions.getPriors(phys_model)
numParams = myfunctions.countParameters(phys_model)
print(numParams,priorObjects)

# In[6]:


start = time.perf_counter()

n_samples_bs = 2000
schedule_fn = optax.polynomial_schedule(init_value=-1e-2, end_value=-1e-4, 
                                      power=0.5, transition_steps=1000)
opt = optax.chain(
  optax.scale_by_adam(),
  optax.scale_by_schedule(schedule_fn),
)

map_estimate, chi = model_seq.MAP(opt, n_samples=n_samples_bs,num_steps=1000,seed=0)
end = time.perf_counter()
MAPtime = end - start
print(MAPtime)

# In[7]:


plt.style.use("default")
np.save(path+"/map.npy",map_estimate)
plt.plot(np.array(chi))
plt.savefig(path+"/chi-squared.png")

# In[8]:


start = time.perf_counter()

try:
    lps = prob_model.log_prob(LensSimulator(phys_model, sim_config, bs=n_samples_bs), map_estimate)[0]
    best = map_estimate[jnp.nanargmax(lps)][jnp.newaxis,:]
except:
    map_estimate=np.load(path+"/map.npy")
    lps = prob_model.log_prob(LensSimulator(phys_model, sim_config, bs=n_samples_bs), map_estimate)[0]
    best = map_estimate[jnp.nanargmax(lps)][jnp.newaxis,:]

end = time.perf_counter()
logProbTime = end-start
print(logProbTime)

# In[9]:


np.save(path+"/best.npy",best)

# In[10]:


params = prob_model.bij.forward(best.tolist()[0])
simulated = lens_sim.lstsq_simulate(params,jnp.array(observed_img),err_map)[0]

plt.figure(figsize=(12, 4))

plt.subplot(131)
norm = simple_norm(observed_img, 'sqrt', percent=99.)
plt.imshow(observed_img, norm=norm, origin='lower', cmap='viridis')
plt.colorbar()

plt.subplot(132)
plt.imshow(simulated, norm=norm, origin='lower', cmap='viridis')
plt.colorbar()
plt.axis('off')

plt.subplot(133)
resid = jnp.array(observed_img) - simulated
plt.imshow(resid/err_map, cmap='coolwarm', origin='lower',interpolation='none', vmin=-5.5, vmax=5.5)


MAPchi = np.mean((resid/err_map)**2)
plt.axis('off')
plt.colorbar()

print(MAPchi)
plt.savefig(path+"/MAPoutput.png")

# In[56]:


start = time.perf_counter()
steps=3000

try:
    schedule_fn = optax.polynomial_schedule(init_value=1e-6, end_value=-5e-4, power=2, transition_steps=steps)
    opt = optax.chain(optax.scale_by_adam(),optax.scale_by_schedule(schedule_fn),)
    qz, loss_hist = model_seq.SVI(best, opt, n_vi=500, num_steps=steps)
    
    mean=qz.mean()
    cov=qz.covariance()
    scale = np.linalg.cholesky(cov)
    
    printToFile("Normal SVI success",path+"output.txt")
except: 
    
    printToFile("Normal SVI failed, skipping...",path+"output.txt")
    schedule_fn = optax.polynomial_schedule(init_value=0, end_value=-1e-15, power=2, transition_steps=500)
    opt = optax.chain(optax.scale_by_adam(),optax.scale_by_schedule(schedule_fn),)
    qz, loss_hist = model_seq.SVI(best, opt, n_vi=500, num_steps=500)
    
    mean=qz.mean()
    cov=qz.covariance()
    scale = np.linalg.cholesky(cov)

end = time.perf_counter()
SVItime = end-start

plt.plot(loss_hist)
#np.save(path+"/SVI.npy",qz)
print(SVItime)
plt.savefig(path+"/SVIloss.png")

# In[64]:


start = time.perf_counter()
print(datetime.now())
num_samples = 750
samples = model_seq.HMC(qz, num_burnin_steps=250, num_results=num_samples)
end = time.perf_counter()

HMCtime = end-start
#np.save(path+"/HMC.npy",samples)
print(HMCtime)

# In[65]:


rhat= tfp.mcmc.potential_scale_reduction(jnp.transpose(samples.all_states, (1,2,0,3)), independent_chain_ndims=2)
print(rhat)

# In[66]:


smp = jnp.transpose(samples.all_states, (1,2,0,3)).reshape((-1, myfunctions.countParameters(phys_model)))
smp_physical = prob_model.bij.forward(list(smp.T))

stack = []
for ii, priorSet in enumerate(priorObjects):
    for iii, priorObject in enumerate(priorSet):
        if priorObject == "EPL":
            stack.extend([
                            smp_physical[ii][iii]["center_x"],
                            smp_physical[ii][iii]["center_y"],
                            smp_physical[ii][iii]["e1"],
                            smp_physical[ii][iii]["e2"],
                            smp_physical[ii][iii]["gamma"],
                            smp_physical[ii][iii]["theta_E"],
            ])
        elif priorObject == "SHEAR":
            stack.extend([
                            smp_physical[ii][iii]["gamma1"],
                            smp_physical[ii][iii]["gamma2"],
            ])
        elif priorObject == "SERSIC_ELLIPSE":
            stack.extend([
                            smp_physical[ii][iii]["R_sersic"],
                            smp_physical[ii][iii]["center_x"],
                            smp_physical[ii][iii]["center_y"],
                            smp_physical[ii][iii]["e1"],
                            smp_physical[ii][iii]["e2"],
                            smp_physical[ii][iii]["n_sersic"],
            ])
        elif priorObject == "SHAPELETS":
            stack.extend([
                            smp_physical[ii][iii]["beta"],
                            smp_physical[ii][iii]["center_x"],
                            smp_physical[ii][iii]["center_y"],
            ])
            
get_samples = np.column_stack(stack)
physical_samples = get_samples

reversed_physical_samples = np.zeros([num_samples*48,numParams])
reversed_physical_samples[:,0:6]=np.flip(physical_samples[:,0:6])
reversed_physical_samples[:,6:8]=np.flip(physical_samples[:,6:8])
reversed_physical_samples[:,8:numParams] = physical_samples[:,8:numParams]

best_HMC=prob_model.pack_bij.forward(np.median(physical_samples,axis=0).tolist())
with open(path+"/bestHMC.json","w") as file:
    json.dump(best_HMC,file,indent=4)

# In[67]:


plt.style.use('default')

labels=[r'$\theta_E$', 
        r'$\gamma$', 
        r'$\epsilon_1$', 
        r'$\epsilon_2$', 
        r'$x$', r'$y$', 
        r'$\gamma_{1,ext}$', 
        r'$\gamma_{2,ext}$']

corner(reversed_physical_samples[:,0:8], show_titles=True, title_fmt='.3f', labels=labels)
print("peepeepoopoo")
plt.savefig(path+"corner.png")

# In[33]:


corner(reversed_physical_samples, show_titles=True, title_fmt='.3f', );
plt.savefig(path+"bigcorner.png")

# In[119]:


plt.style.use('default')
simulated, coeffs = lens_sim.lstsq_simulate(best_HMC,jnp.array(observed_img),err_map)

from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Plots import lens_plot
import lenstronomy.Util.simulation_util as sim_util
from lenstronomy.Data.imaging_data import ImageData
kwargs_data = sim_util.data_configure_simple(numPix*2, deltaPix)
data = ImageData(**kwargs_data)
_coords = data
lens_model_list = ['EPL','SHEAR']
lensModel = LensModel(lens_model_list=lens_model_list)
kwargs_main_lens = {
                        'theta_E': best_HMC[0][0]['theta_E'],
                        'gamma': best_HMC[0][0]['gamma'],
                        'e1': best_HMC[0][0]['e1'],
                        'e2': best_HMC[0][0]['e2'],
                        'center_x': best_HMC[0][0]['center_x'],
                        'center_y': best_HMC[0][0]['center_y'],
                    } #Main lens
kwargs_shear = {
                        'gamma1': best_HMC[0][1]['gamma1'],
                        'gamma2': best_HMC[0][1]['gamma2'],
             }  #External shear
kwargs_lens = [kwargs_main_lens, kwargs_shear]


extent = (-numPix/2*deltaPix, numPix/2*deltaPix, -numPix/2*deltaPix, numPix/2*deltaPix)
scale_length = 1 #arcsec

plt.figure(figsize=(12, 4))

plt.subplot(131)
norm = simple_norm(observed_img, 'sqrt', percent=99.)
plt.imshow(observed_img, norm=norm, origin='lower', cmap='viridis')

ax = plt.subplot(132)
lens_plot.caustics_plot(ax, _coords, lensModel, kwargs_lens, fast_caustic=True, color_crit='red', color_caustic='green',)
plt.imshow(simulated, norm=norm, extent=extent,origin='lower', cmap='viridis')

plt.subplot(133)
resid = jnp.array(observed_img) - simulated
plt.imshow(resid/err_map, cmap='coolwarm', origin='lower',interpolation='none', vmin=-5.5, vmax=5.5)
HMCchi = np.mean((resid/err_map)**2)
print(HMCchi)
#plt.colorbar()

plt.savefig(path+"/HMCoutput.png")

print(kwargs_lens)
def convertEllipticity(e1, e2):
    phi = jnp.arctan2(e2, e1) / 2
    c = jnp.minimum(jnp.sqrt(e1 ** 2 + e2 ** 2), 0.9999)
    q = (1 - c) / (1 + c)
    return float(q), float(phi)

print(convertEllipticity(best_HMC[0][0]['e1'],best_HMC[0][0]['e2']))

# In[118]:


mass = best_HMC[0]
lens_light = best_HMC[1]
source_light = best_HMC[2]
print(coeffs)
for ii, thing in enumerate(lens_light):
    thing["Ie"] = float(coeffs[0][ii]*1/deltaPix**2)

counter = ii+1
for ii, thing in enumerate(source_light):
    if priorObjects[2][ii] == "SHAPELETS":
        #del thing["Ie"]
        n_max = phys_model.getProfiles()[2][ii].n_max
        n_layers = int((n_max + 1) * (n_max + 2) / 2)
        decimal_places = len(str(n_layers))
        amps = []
        for iii in range(n_layers):
            amps.append(f"amp{str(iii).zfill(decimal_places)}")
            thing[amps[iii]]=np.array([coeffs[0][counter]*1/deltaPix**2])
            counter = counter + 1
    else:
        thing["Ie"] = float(coeffs[0][counter]*1/deltaPix**2)
        counter = counter + 1

print(lens_light)
print(source_light)
lens_sim_deconstructed = LensSimulator(phys_model_Forward, sim_config, bs=1)

sourcesimulated = lens_sim_deconstructed.simulate([[], [], source_light])
lenssimulated = lens_sim_deconstructed.simulate([[], lens_light, []])

# individualSourceSimulations = []
# for ii, source in enumerate(source_light):
#     individualSourceSimulations.append(lens_sim_deconstructed.simulate([mass, [], [source_light[ii]]]))    

colors = ["c.","m.","y.","w."]
plt.figure(figsize=(12,4))


plt.subplot(131)
plt.imshow(lenssimulated, norm=norm, extent=extent, origin='lower', cmap='viridis')

ax = plt.subplot(132)
lens_plot.caustics_plot(ax, _coords, lensModel, kwargs_lens, fast_caustic=True, color_crit='red', color_caustic='green',)

for ii in range(0,len(source_light)):
    plt.plot(source_light[ii]["center_x"],source_light[ii]["center_y"],colors[ii])
plt.imshow(sourcesimulated, norm=norm, extent=extent, origin='lower', cmap='viridis')

ax=plt.subplot(133)
for ii in range(0,len(source_light)):
    plt.plot(source_light[ii]["center_x"],source_light[ii]["center_y"],colors[ii])
plt.imshow(sourcesimulated, norm=norm, extent=extent, origin='lower', cmap='viridis')

# ax=plt.subplot(144)
# #lens_plot.caustics_plot(ax, _coords, lensModel, kwargs_lens, fast_caustic=True, color_crit='red', color_caustic='green',)
# plt.plot(source_light[1]["center_x"],source_light[1]["center_y"],colors[1])
# plt.imshow(individualSourceSimulations[1], norm=norm, extent=extent, origin='lower', cmap='viridis')

plt.savefig(path+"/deconstructed.png")

# In[28]:


printToFile(now+"\n", path+"output.txt")
printToFile(str(jax.devices())+"\n", path+"output.txt")
printToFile(str(print(myfunctions.countParameters(phys_model),priorObjects)),path+"output.txt")

# In[29]:


printToFile("MAP took "+str(MAPtime)+" seconds \n",path+"output.txt")
printToFile("log_prob took "+str(logProbTime)+" seconds \n",path+"output.txt")
printToFile('MAP chi-squared: '+str(MAPchi)+ "\n",path+"output.txt")

# In[30]:


printToFile("SVI took "+str(SVItime)+" seconds \n",path+"output.txt")
printToFile("HMC took "+str(HMCtime)+" seconds \n",path+"output.txt")
printToFile('HMC chi-squared: '+str(HMCchi)+ "\n",path+"output.txt")
printToFile('Rhat: '+str(rhat),path+"output.txt")

# In[31]:


for i in range(0,len(best_HMC[0])):
    printToFile("lens: "+ str([[ii,float(best_HMC[0][i][ii])] for ii in best_HMC[0][i]])+ "\n",path+"output.txt")

for i in range(0,len(best_HMC[1])):
    printToFile("lens light: "+ str([[ii,float(best_HMC[1][i][ii])] for ii in best_HMC[1][i]])+ "\n",path+"output.txt")

for i in range(0,len(best_HMC[2])):
    printToFile("source light: "+ str([[ii,float(best_HMC[2][i][ii])] for ii in best_HMC[2][i]])+ "\n",path+"output.txt")
