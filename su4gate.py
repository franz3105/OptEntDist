#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 15:05:26 2022

@author: zsolt
"""

import jax.numpy as jnp
from jax import jit
import jax
from numpy import random
jax.config.update('jax_enable_x64', True)


@jit
def gate(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15):
    U = jnp.array([[jnp.exp((1j * (6 * a1 - 6 * a11 + 6 * a13 + 2 * jnp.sqrt(3) * a14 + jnp.sqrt(
        6) * a15 - 6 * a3 - 6 * a5 - 6 * a7 - 6 * a9)) / 6.) * (-(jnp.cos(a8) * jnp.sin(a12) * jnp.sin(a2)) - jnp.exp(
        1j * (2 * a11 + 2 * a3 + a5 + a7 + a9)) * jnp.cos(a12) * jnp.cos(a2) * jnp.sin(a10) * jnp.sin(a4) - jnp.exp(
        2 * 1j * (a3 + a5 + a7)) * jnp.cos(a2) * jnp.cos(a4) * jnp.cos(a6) * jnp.sin(a12) * jnp.sin(a8) +
                                                                jnp.exp(2 * 1j * (a11 + a9)) * jnp.cos(a10) * jnp.cos(
                a12) * (jnp.exp(2 * 1j * (a3 + a5 + a7)) * jnp.cos(a2) * jnp.cos(a4) * jnp.cos(a6) * jnp.cos(
                a8) - jnp.sin(
                a2) * jnp.sin(a8))),
                    jnp.exp((1j * (6 * a1 - 6 * a11 - 6 * a13 + 2 * jnp.sqrt(3) * a14 + jnp.sqrt(
                        6) * a15 - 6 * a3 - 6 * a5 - 6 * a7 - 6 * a9)) / 6.) * (-(
                            jnp.exp(1j * (2 * a11 + 2 * a3 + a5 + a7 + a9)) * jnp.cos(a2) * jnp.sin(a10) * jnp.sin(
                        a12) * jnp.sin(a4)) + jnp.cos(a12) * (jnp.cos(a8) * jnp.sin(a2) +
                                                              jnp.exp(2 * 1j * (a3 + a5 + a7)) * jnp.cos(a2) * jnp.cos(
                                a4) * jnp.cos(a6) * jnp.sin(a8)) + jnp.exp(2 * 1j * (a11 + a9)) * jnp.cos(
                        a10) * jnp.sin(
                        a12) * (jnp.exp(2 * 1j * (a3 + a5 + a7)) * jnp.cos(a2) * jnp.cos(a4) * jnp.cos(a6) * jnp.cos(
                        a8) - jnp.sin(a2) * jnp.sin(a8))),
                    jnp.exp((-2j * a14) / jnp.sqrt(3) + (1j * a15) / jnp.sqrt(6)) * (
                            jnp.exp(1j * (a1 + a3)) * jnp.cos(a2) * (
                            jnp.exp(1j * (a5 + a7 + a9)) * jnp.cos(a4) * jnp.cos(a6) * jnp.cos(a8) * jnp.sin(
                        a10) + jnp.cos(a10) * jnp.sin(a4)) -
                            jnp.exp(1j * (
                                    a1 - a3 - a5 - a7 + a9)) * jnp.sin(
                        a10) * jnp.sin(a2) * jnp.sin(a8)),
                    jnp.exp((1j * (2 * a1 - jnp.sqrt(6) * a15 + 2 * (a3 + a5))) / 2.) * jnp.cos(a2) * jnp.cos(
                        a4) * jnp.sin(
                        a6)],
                   [(-(jnp.cos(a2) * jnp.cos(a8) * jnp.sin(a12)) - jnp.exp(
                       2 * 1j * (a11 + a3 + a5 + a7 + a9)) * jnp.cos(
                       a10) * jnp.cos(a12) * jnp.cos(a4) * jnp.cos(a6) * jnp.cos(a8) * jnp.sin(a2) + jnp.exp(
                       1j * (2 * a11 + 2 * a3 + a5 + a7 + a9)) * jnp.cos(a12) * jnp.sin(a10) * jnp.sin(a2) * jnp.sin(
                       a4) -
                     jnp.exp(2 * 1j * (a11 + a9)) * jnp.cos(a10) * jnp.cos(a12) * jnp.cos(a2) * jnp.sin(a8) + jnp.exp(
                               2 * 1j * (a3 + a5 + a7)) * jnp.cos(a4) * jnp.cos(a6) * jnp.sin(a12) * jnp.sin(
                               a2) * jnp.sin(
                               a8)) / jnp.exp((1j * (6 * a1 + 6 * a11 - 6 * a13 - 2 * jnp.sqrt(3) * a14 - jnp.sqrt(
                       6) * a15 + 6 * a3 + 6 * a5 + 6 * a7 + 6 * a9)) / 6.),
                    (jnp.cos(a12) * jnp.cos(a2) * jnp.cos(a8) - jnp.exp(2 * 1j * (a11 + a3 + a5 + a7 + a9)) * jnp.cos(
                        a10) * jnp.cos(a4) * jnp.cos(a6) * jnp.cos(a8) * jnp.sin(a12) * jnp.sin(a2) + jnp.exp(
                        1j * (2 * a11 + 2 * a3 + a5 + a7 + a9)) * jnp.sin(a10) * jnp.sin(a12) * jnp.sin(a2) * jnp.sin(
                        a4) -
                     jnp.exp(2 * 1j * (a11 + a9)) * jnp.cos(a10) * jnp.cos(a2) * jnp.sin(a12) * jnp.sin(a8) - jnp.exp(
                                2 * 1j * (a3 + a5 + a7)) * jnp.cos(a12) * jnp.cos(a4) * jnp.cos(a6) * jnp.sin(
                                a2) * jnp.sin(
                                a8)) / jnp.exp((1j * (6 * a1 + 6 * a11 + 6 * a13 - 2 * jnp.sqrt(3) * a14 - jnp.sqrt(
                        6) * a15 + 6 * a3 + 6 * a5 + 6 * a7 + 6 * a9)) / 6.),
                    (-(jnp.exp(1j * (2 * a3 + 2 * a5 + 2 * a7 + a9)) * jnp.cos(a4) * jnp.cos(a6) * jnp.cos(
                        a8) * jnp.sin(
                        a10) * jnp.sin(a2)) - jnp.exp(1j * (2 * a3 + a5 + a7)) * jnp.cos(a10) * jnp.sin(a2) * jnp.sin(
                        a4) - jnp.exp(1j * a9) * jnp.cos(a2) * jnp.sin(a10) * jnp.sin(a8)) /
                    jnp.exp(
                        (1j * (6 * a1 + 4 * jnp.sqrt(3) * a14 - jnp.sqrt(6) * a15 + 6 * a3 + 6 * a5 + 6 * a7)) / 6.),
                    -((jnp.cos(a4) * jnp.sin(a2) * jnp.sin(a6)) / jnp.exp(
                        (1j * (2 * a1 + jnp.sqrt(6) * a15 - 2 * (a3 + a5))) / 2.))],
                   [(-(jnp.exp(1j * (2 * a11 + a9)) * jnp.cos(a12) * (
                           jnp.cos(a4) * jnp.sin(a10) + jnp.exp(1j * (a5 + a7 + a9)) * jnp.cos(a10) * jnp.cos(
                       a6) * jnp.cos(a8) * jnp.sin(a4))) + jnp.exp(1j * (a5 + a7)) * jnp.cos(a6) * jnp.sin(
                       a12) * jnp.sin(
                       a4) * jnp.sin(a8)) /
                    jnp.exp((1j * (6 * a11 - 6 * a13 - 2 * jnp.sqrt(3) * a14 - jnp.sqrt(6) * a15 + 6 * a9)) / 6.),
                    (-(jnp.exp(1j * (2 * a11 + a9)) * jnp.cos(a4) * jnp.sin(a10) * jnp.sin(a12)) - jnp.exp(
                        1j * (a5 + a7)) * jnp.cos(a6) * jnp.sin(a4) * (
                             jnp.exp(2 * 1j * (a11 + a9)) * jnp.cos(a10) * jnp.cos(a8) * jnp.sin(a12) + jnp.cos(
                         a12) * jnp.sin(a8))) /
                    jnp.exp((1j * (6 * a11 + 6 * a13 - 2 * jnp.sqrt(3) * a14 - jnp.sqrt(6) * a15 + 6 * a9)) / 6.),
                    jnp.exp((-2j * a14) / jnp.sqrt(3) + (1j * a15) / jnp.sqrt(6)) * (
                            jnp.cos(a10) * jnp.cos(a4) - jnp.exp(1j * (a5 + a7 + a9)) * jnp.cos(a6) * jnp.cos(
                        a8) * jnp.sin(a10) * jnp.sin(a4)),
                    -(jnp.exp(-1j * jnp.sqrt(1.5) * a15 + 1j * a5) * jnp.sin(a4) * jnp.sin(a6))],
                   [(jnp.sin(a6) * (
                           -(jnp.exp(2 * 1j * (a11 + a9)) * jnp.cos(a10) * jnp.cos(a12) * jnp.cos(a8)) + jnp.sin(
                       a12) * jnp.sin(a8))) / jnp.exp(
                       (1j * (6 * a11 - 6 * a13 - 2 * jnp.sqrt(3) * a14 - jnp.sqrt(6) * a15 - 6 * a7 + 6 * a9)) / 6.),
                    -((jnp.sin(a6) * (
                            jnp.exp(2 * 1j * (a11 + a9)) * jnp.cos(a10) * jnp.cos(a8) * jnp.sin(a12) + jnp.cos(
                        a12) * jnp.sin(a8))) / jnp.exp(
                        (1j * (6 * a11 + 6 * a13 - 2 * jnp.sqrt(3) * a14 - jnp.sqrt(6) * a15 - 6 * a7 + 6 * a9)) / 6.)),
                    -((jnp.cos(a8) * jnp.sin(a10) * jnp.sin(a6)) / jnp.exp(
                        (1j * (4 * jnp.sqrt(3) * a14 - jnp.sqrt(6) * a15 - 6 * (a7 + a9))) / 6.)),
                    jnp.cos(a6) / jnp.exp(1j * jnp.sqrt(1.5) * a15)]], dtype=jnp.complex128)
    return U


# CNOT --you need an extra e^{I 3pi/4} phase
# jnp.set_printoptions(precision=3, suppress=True)
# print(jnp.exp(1j * 3 * jnp.pi / 4) * gate(0, 0, jnp.pi / 4, jnp.pi / 2, jnp.pi / 4, jnp.pi / 2, jnp.pi / 4, 0, 0,
#                                          jnp.pi / 2, 0,
#                                          0, 0, 0, 0))

# hp_rect = jnp.pi * jnp.array(
#    [1, 1 / 2, 1, 1 / 2, 1, 1 / 2, 1, 1 / 2, 1, 1 / 2, 1, 1 / 2, 1, 1 / jnp.sqrt(3), 1 / jnp.sqrt(6)])

# a general SU(4) element
"""a1 = random.uniform(0, hp_rect[0], 1)
a2 = random.uniform(0, hp_rect[1], 1)
a3 = random.uniform(0, hp_rect[2], 1)
a4 = random.uniform(0, hp_rect[3], 1)
a5 = random.uniform(0, hp_rect[4], 1)
a6 = random.uniform(0, hp_rect[5], 1)
a7 = random.uniform(0, hp_rect[6], 1)
a8 = random.uniform(0, hp_rect[7], 1)
a9 = random.uniform(0, hp_rect[8], 1)
a10 = random.uniform(0, hp_rect[9], 1)
a11 = random.uniform(0, hp_rect[10], 1)
a12 = random.uniform(0, hp_rect[11], 1)
a13 = random.uniform(0, hp_rect[12], 1)
a14 = random.uniform(0, hp_rect[13], 1)
a15 = random.uniform(0, hp_rect[14], 1)

# random.uniform creates a list---> [0] needed to refer to the random number

print(
    gate(a1[0], a2[0], a3[0], a4[0], a5[0], a6[0], a7[0], a8[0], a9[0], a10[0], a11[0], a12[0], a13[0], a14[0], a15[0]))"""
