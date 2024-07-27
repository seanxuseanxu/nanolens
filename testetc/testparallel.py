import os
import jax
print(os.environ)
os.environ['SLURM_STEP_NODELIST'] = os.environ['SLURM_NODELIST']
jax.distributed.initialize()
print(jax.devices())
print(jax.local_devices())