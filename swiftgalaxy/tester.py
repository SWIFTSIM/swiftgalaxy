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

SG2 = SWIFTGalaxy(
    snapshot_filename,
    Velociraptor(
        velociraptor_filebase,
        halo_index=target_halo_index,
        extra_mask='bound_only',
    ),
    auto_recentre=True
)

rotmat = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
trans = np.array([0.1, 0, 0]) * u.Mpc
boost = np.array([100 * 1000, 0, 0]) * u.m / u.s  # transform4 matrices are not unit-aware enough!
# store an assumed unit based on metadata and assert that data units match?
SG2.gas.coordinates
print(SG2.gas.velocities[0].to(u.km / u.s))
SG.translate(trans)
SG.boost(boost)
# SG.rotate(rotmat=rotmat)
SG2.translate(trans)
SG2.boost(boost)
# SG2.rotate(rotmat=rotmat)
print(SG.gas.velocities[0].to(u.km / u.s))
print(SG2.gas.velocities[0].to(u.km / u.s))
assert np.isclose(SG.gas.coordinates.to_value(u.Mpc), SG2.gas.coordinates.to_value(u.Mpc)).all()
assert np.isclose(SG.gas.velocities.to_value(u.km / u.s), SG2.gas.velocities.to_value(u.km / u.s)).all()
