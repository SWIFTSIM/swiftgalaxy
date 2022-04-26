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
present_particle_types = {
    0: 'gas',
    1: 'dark_matter',
    4: 'stars',
    5: 'black_holes'
}


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
    getattr(sd, present_particle_types[0]).coordinates = np.vstack((
        np.random.rand(n_g_b, 3) * 10,
        np.hstack((
            # 10 kpc disc radius, offcentred in box
            2 + R * np.cos(phi) * .01,
            2 + R * np.sin(phi) * .01,
            2 + (np.random.rand(n_g, 1) * 2 - 1) * .001  # 1 kpc height
        ))
    )) * u.Mpc
    getattr(sd, present_particle_types[0]).velocities = np.vstack((
        np.random.rand(n_g_b, 3) * 2 - 1,  # 1 km/s for background
        np.hstack((
            # solid body, 100 km/s at edge
            200 + R * np.sin(phi) * 100,
            200 + R * np.cos(phi) * 100,
            200 + np.random.rand(n_g, 1) * 20 - 10  # 10 km/s vertical
        ))
    )) * u.km / u.s
    getattr(sd, present_particle_types[0]).masses = np.concatenate((
        np.ones(n_g_b, dtype=float),
        np.ones(n_g, dtype=float)
    )) * 1e3 * u.msun
    getattr(sd, present_particle_types[0]).internal_energy = np.concatenate((
        np.ones(n_g_b, dtype=float),  # 1e4 K
        np.ones(n_g, dtype=float) / 10  # 1e3 K
    )) * 1e4 * u.kb * u.K / (1e3 * u.msun)
    getattr(sd, present_particle_types[0]).generate_smoothing_lengths(
        boxsize=boxsize * u.Mpc, dimension=3)

    # Insert a uniform DM background plus a galaxy halo
    n_dm_all = 32 ** 3
    n_dm = 10000
    n_dm_b = n_dm_all - n_dm
    phi = np.random.rand(n_dm, 1) * 2 * np.pi
    theta = np.arccos(np.random.rand(n_dm, 1) * 2 - 1)
    r = np.random.rand(n_dm, 1)
    getattr(sd, present_particle_types[1]).coordinates = np.vstack((
        np.random.rand(n_dm_b, 3) * 10,
        np.hstack((
            # 100 kpc halo radius, offcentred in box
            2 + r * np.cos(phi) * np.sin(theta) * .1,
            2 + r * np.sin(phi) * np.sin(theta) * .1,
            2 + r * np.cos(theta) * .1
        ))
    )) * u.Mpc
    getattr(sd, present_particle_types[1]).velocities = np.vstack((
        # 1 km/s background, 100 km/s halo
        np.random.rand(n_dm_b, 3) * 2 - 1,
        200 + (np.random.rand(n_dm, 3) * 2 - 1) * 100
    )) * u.km / u.s
    getattr(sd, present_particle_types[1]).masses = np.concatenate((
        np.ones(n_dm_b, dtype=float),
        np.ones(n_dm, dtype=float)
    )) * 1e4 * u.msun
    getattr(sd, present_particle_types[1]).generate_smoothing_lengths(
        boxsize=boxsize * u.Mpc, dimension=3)

    # Insert a galaxy stellar disc
    n_s = 10000
    phi = np.random.rand(n_s, 1) * 2 * np.pi
    R = np.random.rand(n_s, 1)
    getattr(sd, present_particle_types[4]).coordinates = np.hstack((
        # 5 kpc disc radius, offcentred in box
        2 + R * np.cos(phi) * .005,
        2 + R * np.sin(phi) * .005,
        2 + (np.random.rand(n_s, 1) * 2 - 1) * .0005  # 500 pc height
    )) * u.Mpc
    getattr(sd, present_particle_types[4]).velocities = np.hstack((
        # solid body, 50 km/s at edge
        200 + R * np.sin(phi) * 50,
        200 + R * np.cos(phi) * 50,
        200 + np.random.rand(n_g, 1) * 20 - 10  # 10 km/s vertical motions
    )) * u.km / u.s
    getattr(sd, present_particle_types[4]).masses = \
        np.ones(n_g, dtype=float) * 1e3 * u.msun
    getattr(sd, present_particle_types[4]).generate_smoothing_lengths(
        boxsize=boxsize * u.Mpc, dimension=3)

    n_bh = 1
    getattr(sd, present_particle_types[5]).coordinates = \
        (2 + np.zeros((n_bh, 3), dtype=float)) * u.Mpc
    getattr(sd, present_particle_types[5]).velocities = \
        (200 + np.zeros((n_bh, 3), dtype=float)) * u.km / u.s
    getattr(sd, present_particle_types[5]).masses = \
        np.ones(n_bh, dtype=float) * 1e6 * u.msun
    getattr(sd, present_particle_types[5]).generate_smoothing_lengths(
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
        hsg = f.create_group('HydroScheme')
        hsg.attrs['Adiabatic index'] = 5.0 / 3.0

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

        ssg = f.create_group('SubgridScheme')
        ncg = ssg.create_group('NamedColumns')
        ncg.create_dataset(
            'HydrogenIonizationFractions',
            data=np.array([b'Neutral', b'Ionized'], dtype='|S32')
        )
        g = f['PartType0']
        f_neutral = np.random.rand(n_g_all)
        f_ion = 1 - f_neutral
        hifd = g.create_dataset(
            'HydrogenIonizationFractions',
            data=np.array([f_neutral, f_ion], dtype=float).T
        )
        hifd.attrs[
            'Conversion factor to CGS'
            ' (not including cosmological corrections)'
        ] = np.array([1.], dtype=float)
        hifd.attrs[
            'Conversion factor to physical CGS'
            ' (including cosmological corrections)'
        ] = np.array([1.], dtype=float)
        hifd.attrs['U_I exponent'] = np.array([0.], dtype=float)
        hifd.attrs['U_L exponent'] = np.array([0.], dtype=float)
        hifd.attrs['U_M exponent'] = np.array([0.], dtype=float)
        hifd.attrs['U_T exponent'] = np.array([0.], dtype=float)
        hifd.attrs['U_t exponent'] = np.array([0.], dtype=float)
        hifd.attrs['a-scale exponent'] = np.array([0.], dtype=float)
        hifd.attrs['h-scale exponent'] = np.array([0.], dtype=float)

    return


def remove_toysnap(snapfile=toysnap_filename):
    os.remove(snapfile)
    return
