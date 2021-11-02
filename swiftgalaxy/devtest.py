from os import path
from ._swiftgalaxy import SWIFTGalaxy

snapnum = 23
base_dir = '/home/koman/'\
    '106e3_104b2_norm_0p3_new_cooling_L006N188/'
velociraptor_filebase = path.join(base_dir, 'halo_{:04d}'.format(snapnum))
snapshot_filename = path.join(base_dir, 'colibre_{:04d}.hdf5'.format(snapnum))

target_halo_index = 3

SG = SWIFTGalaxy(
    snapshot_filename,
    velociraptor_filebase,
    int(target_halo_index)
)
SGm = SWIFTGalaxy(
    snapshot_filename,
    velociraptor_filebase,
    int(target_halo_index),
    extra_mask='bound_only'
)

print(SG.gas.coordinates.shape)
print(SGm.gas.coordinates.shape)
