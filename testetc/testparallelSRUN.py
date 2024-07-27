#!/usr/bin/env python
# coding: utf-8

import jax
import jax.numpy as jnp

jax.distributed.initialize()

print(jax.devices())
print(jax.local_devices())

array = jnp.ones([1,100])*3

out = jax.pmap(lambda x: x ** 2)(array) 
print(out)