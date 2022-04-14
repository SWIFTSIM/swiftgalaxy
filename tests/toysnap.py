import os
import h5py
import numpy as np
import unyt as u
from swiftsimio.objects import cosmo_array
from swiftsimio import Writer
from swiftgalaxy import MaskCollection
from swiftgalaxy.halo_finders import _HaloFinder
from swiftsimio.units import cosmo_units

toysnap_filename = 'toysnap.hdf5'


class ToyHF(_HaloFinder):

    def __init__(self, snapfile=toysnap_filename):
        self.snapfile = snapfile
        super().__init__()
        return

    def _load(self):
        return

    def _get_spatial_mask(self, SG):
        import swiftsimio
        spatial_mask = [None, None, None]
        swift_mask = swiftsimio.mask(self.snapfile, spatial_only=True)
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
        return cosmo_array([2, 2, 2], u.Mpc)

    def _vcentre(self):
        return cosmo_array([200, 200, 200], u.km / u.s)


def create_toysnap(
        snapfile=toysnap_filename,
        alt_coord_name=None,
        alt_vel_name=None,
        alt_id_name=None
):
    """
    Creates a sample dataset of a toy galaxy.
    """

    boxsize = 10
    sd = Writer(cosmo_units, np.ones(3, dtype=float) * boxsize * u.Mpc)

    # Insert a uniform gas background plus a galaxy disc
    n_g_all = 32 ** 3
    n_g = 10000
    n_g_b = n_g_all - n_g
    phi = np.random.rand(n_g, 1) * 2 * np.pi
    R = np.random.rand(n_g, 1)
    sd.gas.coordinates = np.vstack((
        np.random.rand(n_g_b, 3) * 10,
        np.hstack((
            # 10 kpc disc radius, offcentred in box
            2 + R * np.cos(phi) * .01,
            2 + R * np.sin(phi) * .01,
            2 + (np.random.rand(n_g, 1) * 2 - 1) * .001  # 1 kpc height
        ))
    )) * u.Mpc
    sd.gas.velocities = np.vstack((
        np.random.rand(n_g_b, 3) * 2 - 1,  # 1 km/s for background
        np.hstack((
            # solid body, 100 km/s at edge
            200 + R * np.sin(phi) * 100,
            200 + R * np.cos(phi) * 100,
            200 + np.random.rand(n_g, 1) * 20 - 10  # 10 km/s vertical
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
            # 100 kpc halo radius, offcentred in box
            2 + r * np.cos(phi) * np.sin(theta) * .1,
            2 + r * np.sin(phi) * np.sin(theta) * .1,
            2 + r * np.cos(theta) * .1
        ))
    )) * u.Mpc
    sd.dark_matter.velocities = np.vstack((
        # 1 km/s background, 100 km/s halo
        np.random.rand(n_dm_b, 3) * 2 - 1,
        200 + (np.random.rand(n_dm, 3) * 2 - 1) * 100
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
        # 5 kpc disc radius, offcentred in box
        2 + R * np.cos(phi) * .005,
        2 + R * np.sin(phi) * .005,
        2 + (np.random.rand(n_s, 1) * 2 - 1) * .0005  # 500 pc height
    )) * u.Mpc
    sd.stars.velocities = np.hstack((
        # solid body, 50 km/s at edge
        200 + R * np.sin(phi) * 50,
        200 + R * np.cos(phi) * 50,
        200 + np.random.rand(n_g, 1) * 20 - 10  # 10 km/s vertical motions
    )) * u.km / u.s
    sd.stars.masses = np.ones(n_g, dtype=float) * 1e3 * u.msun
    sd.stars.generate_smoothing_lengths(
        boxsize=boxsize * u.Mpc, dimension=3)

    n_bh = 1
    sd.black_holes.coordinates = (2 + np.zeros((n_bh, 3), dtype=float)) * u.Mpc
    sd.black_holes.velocities = \
        (200 + np.zeros((n_bh, 3), dtype=float)) * u.km / u.s
    sd.black_holes.masses = np.ones(n_bh, dtype=float) * 1e6 * u.msun
    sd.black_holes.generate_smoothing_lengths(
        boxsize=boxsize * u.Mpc, dimension=3)

    sd.write(snapfile)  # IDs auto-generated

    with h5py.File(snapfile, 'r+') as f:
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

        for pt in (0, 1, 4, 5):
            g = f[f'PartType{pt}']
            g['ExtraCoordinates'] = g['Coordinates']
            g['ExtraVelocities'] = g['Velocities']
            if alt_id_name is not None:
                g[alt_id_name] = g['ParticleIDs']
                del g['ParticleIDs']
            if alt_coord_name is not None:
                g[alt_coord_name] = g['Coordinates']
                del g['Coordinates']
            if alt_vel_name is not None:
                g[alt_vel_name] = g['Velocities']
                del g['Velocities']

    return


def remove_toysnap(snapfile=toysnap_filename):
    os.remove(snapfile)
    return
