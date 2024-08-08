#!/usr/bin/env python
# coding: utf-8

import jax
jax.distributed.initialize()#local_device_ids=range(16))  # On GPU, see above for the necessary arguments.
print(f"Process {jax.process_index()} global devices : {jax.devices()}")
print(f"Process {jax.process_index()} local devices : {jax.local_devices()}")

# The psum is performed over all mapped devices across the pod slice
key = jax.random.PRNGKey(jax.process_index())
xs = jax.random.uniform(key, (jax.local_device_count(),))
print(xs)
asdf = jax.pmap(lambda x: jax.lax.psum(x, 'i'), axis_name='i')(xs)
print(asdf)