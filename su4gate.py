#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 15:05:26 2022

@author: zsolt
"""

import jax.numpy as jnp
from jax import jit
import jax

jax.config.update('jax_enable_x64', True)

@jit
def SU4gate(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15):
    """
    analytical expression of SU(4) unitary_gate imported from Mathematica
    :return: unitary U
    """
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