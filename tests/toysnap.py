import os
import h5py
import numpy as np
import unyt as u
from swiftsimio.objects import cosmo_array
from swiftsimio import Writer
from swiftgalaxy import MaskCollection
from swiftgalaxy.halo_finders import _HaloFinder

toysnap_filename = 'toysnap.hdf5'


class ToyHF(_HaloFinder):

    def __init__(self):
        super().__init__()
        return

    def _load(self):
        return

    def _get_spatial_mask(self, SG):
        import swiftsimio
        spatial_mask = [None, None, None]
        swift_mask = swiftsimio.mask(toysnap_filename, spatial_only=True)
        swift_mask.constrain_spatial(spatial_mask)
        return swift_mask

    def _get_extra_mask(self, SG):
        extra_mask = MaskCollection(
            gas=np.s_[-10000:],
            dark_matter=np.s_[-10000:],
            stars=...,
            black_holes=...
        )
        return extra_mask

    def _centre(self):
        return cosmo_array([5, 5, 5], u.Mpc)

    def _vcentre(self):
        return cosmo_array([0, 0, 0], u.km / u.s)


def create_toysnap():
    """
    Creates a sample dataset of a toy galaxy.
    """

    boxsize = 10
    sd = Writer("galactic", np.ones(3, dtype=float) * boxsize * u.Mpc)

    # Insert a uniform gas background plus a galaxy disc
    n_g_all = 32 ** 3
    n_g = 10000
    n_g_b = n_g_all - n_g
    phi = np.random.rand(n_g, 1) * 2 * np.pi
    R = np.random.rand(n_g, 1)
    sd.gas.coordinates = np.vstack((
        np.random.rand(n_g_b, 3) * 10,
        np.hstack((
            # 10 kpc disc radius, centred in box
            5 + R * np.cos(phi) * .01,
            5 + R * np.sin(phi) * .01,
            5 + (np.random.rand(n_g, 1) * 2 - 1) * .001  # 1 kpc disc height
        ))
    )) * u.Mpc
    sd.gas.velocities = np.vstack((
        np.random.rand(n_g_b, 3) * 2 - 1,  # 1 km/s for background
        np.hstack((
            # solid body, 100 km/s at edge
            R * np.sin(phi) * 100,
            R * np.cos(phi) * 100,
            np.random.rand(n_g, 1) * 20 - 10  # 10 km/s vertical motions
        ))
    )) * u.km / u.s
    sd.gas.masses = np.concatenate((
        np.ones(n_g_b, dtype=float),
        np.ones(n_g, dtype=float)
    )) * 1e3 * u.msun
    sd.gas.internal_energy = np.concatenate((
        np.ones(n_g_b, dtype=float),  # 1e4 K
        np.ones(n_g, dtype=float) / 10  # 1e3 K
    )) * 1e4 * u.kb * u.K / (1e3 * u.msun)
    sd.gas.generate_smoothing_lengths(
        boxsize=boxsize * u.Mpc, dimension=3)

    # Insert a uniform DM background plus a galaxy halo
    n_dm_all = 32 ** 3
    n_dm = 10000
    n_dm_b = n_dm_all - n_dm
    phi = np.random.rand(n_dm, 1) * 2 * np.pi
    theta = np.arccos(np.random.rand(n_dm, 1) * 2 - 1)
    r = np.random.rand(n_dm, 1)
    sd.dark_matter.coordinates = np.vstack((
        np.random.rand(n_dm_b, 3) * 10,
        np.hstack((
            # 100 kpc halo radius, centred in box
            5 + r * np.cos(phi) * np.sin(theta) * .1,
            5 + r * np.sin(phi) * np.sin(theta) * .1,
            5 + r * np.cos(theta) * .1
        ))
    )) * u.Mpc
    sd.dark_matter.velocities = np.vstack((
        np.random.rand(n_dm_b, 3) * 2 - 1,  # 1 km/s background
        (np.random.rand(n_dm, 3) * 2 - 1) * 100 / np.sqrt(3)  # 100 km/s halo
    )) * u.km / u.s
    sd.dark_matter.masses = np.concatenate((
        np.ones(n_dm_b, dtype=float),
        np.ones(n_dm, dtype=float)
    )) * 1e4 * u.msun
    sd.dark_matter.generate_smoothing_lengths(
        boxsize=boxsize * u.Mpc, dimension=3)

    # Insert a galaxy stellar disc
    n_s = 10000
    phi = np.random.rand(n_s, 1) * 2 * np.pi
    R = np.random.rand(n_s, 1)
    sd.stars.coordinates = np.hstack((
        # 5 kpc disc radius, centred in box
        5 + R * np.cos(phi) * .005,
        5 + R * np.sin(phi) * .005,
        5 + (np.random.rand(n_s, 1) * 2 - 1) * .0005  # 500 pc disc height
    )) * u.Mpc
    sd.stars.velocities = np.hstack((
        # solid body, 50 km/s at edge
        R * np.sin(phi) * 50,
        R * np.cos(phi) * 50,
        np.random.rand(n_g, 1) * 20 - 10  # 10 km/s vertical motions
    )) * u.km / u.s
    sd.stars.masses = np.ones(n_g, dtype=float) * 1e3 * u.msun
    sd.stars.generate_smoothing_lengths(
        boxsize=boxsize * u.Mpc, dimension=3)

    n_bh = 1
    sd.black_holes.coordinates = (5 + np.ones((n_bh, 3), dtype=float)) * u.Mpc
    sd.black_holes.velocities = np.zeros((n_bh, 3), dtype=float) * u.km / u.s
    sd.black_holes.masses = np.ones(n_bh, dtype=float) * 1e6 * u.msun
    sd.black_holes.generate_smoothing_lengths(
        boxsize=boxsize * u.Mpc, dimension=3)

    sd.write(toysnap_filename)  # IDs auto-generated

    with h5py.File(toysnap_filename, 'r+') as f:
        g = f.create_group('Cells')
        g.create_dataset('Centres', data=np.array([[5, 5, 5]], dtype=float))
        cg = g.create_group('Counts')
        cg.create_dataset('PartType0', data=np.array([n_g_all]), dtype=int)
        cg.create_dataset('PartType1', data=np.array([n_dm_all]), dtype=int)
        cg.create_dataset('PartType4', data=np.array([n_s]), dtype=int)
        cg.create_dataset('PartType5', data=np.array([n_bh]), dtype=int)
        fg = g.create_group('Files')
        fg.create_dataset('PartType0', data=np.array([0], dtype=int))
        fg.create_dataset('PartType1', data=np.array([0], dtype=int))
        fg.create_dataset('PartType4', data=np.array([0], dtype=int))
        fg.create_dataset('PartType5', data=np.array([0], dtype=int))
        mdg = g.create_group('Meta-data')
        mdg.attrs['dimension'] = np.array([[1, 1, 1]], dtype=int)
        mdg.attrs['nr_cells'] = np.array([1], dtype=int)
        mdg.attrs['size'] = np.array([[boxsize, boxsize, boxsize]], dtype=int)
        og = g.create_group('OffsetsInFile')
        og.create_dataset('PartType0', data=np.array([0], dtype=int))
        og.create_dataset('PartType1', data=np.array([0], dtype=int))
        og.create_dataset('PartType4', data=np.array([0], dtype=int))
        og.create_dataset('PartType5', data=np.array([0], dtype=int))

    return


def remove_toysnap():
    os.remove(toysnap_filename)
    return
