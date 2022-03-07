import socket
from os import path
from _swiftgalaxy import SWIFTGalaxy
from _halofinder import Velociraptor

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

print(SG.halo_finder.ids)
