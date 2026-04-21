#------------------------------------------------------------------------------------------------------------
#------------------------------------------------- Imports --------------------------------------------------
#------------------------------------------------------------------------------------------------------------

import numpy as np
from constants import *
import os, shutil
import math as mt
import jax
jax.config.update("jax_enable_x64", True) #IMPORTANT since julia is default higher precision!
import jax.numpy as jnp
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy.fft as jfft
import time
from diffrax import diffeqsolve, ODETerm, Tsit5, SaveAt, PIDController, Dopri5, Dopri8, Kvaerno5 # type: ignore
from jax.scipy.sparse.linalg import cg
from scipy import optimize
from scipy.optimize import minimize_scalar
#import pytest #NOTE uncomment this if running pytests, otherwise keep commented while running on HPC
from functools import partial
#from concurrent.futures import ThreadPoolExecutor
from diffrax import RecursiveCheckpointAdjoint
import camb
from scipy.optimize import minimize
import argparse
from itertools import combinations