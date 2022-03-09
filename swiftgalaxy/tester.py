import socket
import unyt as u
from os import path
from _swiftgalaxy import SWIFTGalaxy
from _halo_finders import Velociraptor
import numpy as np

snapnum = 23
if 'cosma' in socket.gethostname():
    base_dir = '/cosma7/data/dp004/dc-chai1/HAWK/'\
        '106e3_104b2_norm_0p3_new_cooling_L006N188/'
elif ('autarch' in socket.gethostname()) \
     or ('farseer' in socket.gethostname()):
    base_dir = '/home/koman/'\
        '106e3_104b2_norm_0p3_new_cooling_L006N188/'

velociraptor_filebase = path.join(base_dir, 'halo_{:04d}'.format(snapnum))
snapshot_filename = path.join(base_dir, 'colibre_{:04d}.hdf5'.format(snapnum))

target_halo_index = 3

SG = SWIFTGalaxy(
    snapshot_filename,
    Velociraptor(
        velociraptor_filebase,
        halo_index=target_halo_index,
        extra_mask='bound_only',
    ),
    auto_recentre=True
)
SG.gas.coordinates, SG.gas.velocities


def f(SG):
    print('v_r=', SG.gas.spherical_velocities.v_r[0].to(u.km / u.s))
    print('v_t=', SG.gas.spherical_velocities.v_theta[0].to(u.km / u.s))
    print('v_p=', SG.gas.spherical_velocities.v_phi[0].to(u.km / u.s))
    SG._void_derived_representations()
    print('v_R=', SG.gas.cylindrical_velocities.v_rho[0].to(u.km / u.s))
    print('v_p=', SG.gas.cylindrical_velocities.v_phi[0].to(u.km / u.s))
    print('v_z=', SG.gas.cylindrical_velocities.v_z[0].to(u.km / u.s))
    SG._void_derived_representations()
    return


SG.gas.coordinates[0] = [0, 0, 1]
SG.gas.velocities[0] = [1, 0, 0]
f(SG)
