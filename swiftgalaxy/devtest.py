import numpy as np
from os import path
from ._swiftgalaxy import SWIFTGalaxy
import unyt as u
from velociraptor import load as load_catalogue
from velociraptor.particles import load_groups

snapnum = 23
base_dir = '/cosma/home/durham/dc-oman1/ColibreTestData/'\
    '106e3_104b2_norm_0p3_new_cooling_L006N188/'
velociraptor_filebase = path.join(base_dir, 'halo_{:04d}'.format(snapnum))
snapshot_filename = path.join(base_dir, 'colibre_{:04d}.hdf5'.format(snapnum))

catalogue = load_catalogue(
    path.join(base_dir, f'{velociraptor_filebase}.properties')
)
target_mask = np.logical_and.reduce((
    catalogue.ids.hosthaloid == -1,
    catalogue.velocities.vmax > 50 * u.km / u.s,
    catalogue.velocities.vmax < 100 * u.km / u.s
))
target_halo_ids = catalogue.ids.id[target_mask]
target_halo_id = target_halo_ids[0]
groups = load_groups(
    path.join(base_dir, f'{velociraptor_filebase}.catalog_groups'),
    catalogue=catalogue
)
particles, unbound_particles = groups.extract_halo(halo_id=int(target_halo_id))

SG = SWIFTGalaxy(
    snapshot_filename,
    velociraptor_filebase,
    int(target_halo_id),
    extra_mask='bound_only'
)
print(SG.gas.coordinates)
SG.rotate(angle_axis=(90 * u.deg, 'z'))
print(SG.gas.coordinates)
