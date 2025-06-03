"""
Functions and definitions to retrieve, generate and use illustrative example data.
"""

import os
import h5py
import numpy as np
import unyt as u
from typing import Optional
from astropy.cosmology import LambdaCDM
from astropy import units as U
from swiftsimio.objects import cosmo_array
import swiftsimio
from swiftsimio import Writer, SWIFTMask
from swiftgalaxy import MaskCollection, SWIFTGalaxy
from swiftgalaxy.halo_catalogues import _HaloCatalogue
from swiftsimio.units import cosmo_units


my_soap_script_path = os.path.expanduser("~/code/SOAP/")
if os.path.exists(my_soap_script_path):
    soap_script_path = my_soap_script_path
try:
    soap_script_path = os.path.join(os.environ["GITHUB_WORKSPACE"], "SOAP")
except KeyError:
    pass  # not on github CI
soap_script = os.path.join(soap_script_path, "make_virtual_snapshot.py")
assert os.path.exists(soap_script)

toysnap_filename = "toysnap.hdf5"
toyvr_filebase = "toyvr"
toysoap_filename = "toysoap.hdf5"
toysoap_membership_filebase = "toysoap_membership"
toysoap_virtual_snapshot_filename = "toysnap_virtual.hdf5"
toycaesar_filename = "toycaesar.hdf5"
present_particle_types = {0: "gas", 1: "dark_matter", 4: "stars", 5: "black_holes"}
boxsize = 10.0 * u.Mpc
n_g_all = 32**3
centre_1 = 2
centre_2 = 8
vcentre_1 = 200
vcentre_2 = 600
n_g_1 = 5000
n_g_2 = 5000
n_g_b = n_g_all - n_g_1 - n_g_2
n_dm_all = 32**3
n_dm_1 = 5000
n_dm_2 = 5000
n_dm_b = n_dm_all - n_dm_1 - n_dm_2
n_s_1 = 5000
n_s_2 = 5000
n_s_all = n_s_1 + n_s_2
n_bh_1 = 1
n_bh_2 = 1
n_bh_all = n_bh_1 + n_bh_2
m_g = 1e3 * u.msun
T_g = 1e4 * u.K
m_dm = 1e4 * u.msun
m_s = 1e3 * u.msun
m_bh = 1e6 * u.msun

Om_m = 0.3
Om_l = 0.7
Om_b = 0.05
h0 = 0.7
w0 = -1.0
rho_c = (3 * (h0 * 100 * u.km / u.s / u.Mpc) ** 2 / 8 / np.pi / u.G).to(
    u.msun / u.kpc**3
)
age = u.unyt_quantity.from_astropy(
    LambdaCDM(
        H0=h0 * 100 * U.km / U.s / U.Mpc, Om0=Om_m, Ode0=Om_l, Ob0=Om_b
    ).lookback_time(np.inf)
)


class ToyHF(_HaloCatalogue):
    """
    A minimalist halo catalogue class for demo use with
    :class:`~swiftgalaxy.reader.SWIFTGalaxy`.

    Parameters
    ----------
    snapfile : :obj:`str`
        The snapshot filename. (Default: ``'toysnap.hdf5'``)

    index : :obj:`int`
        The index (position in the catalogue) of the target galaxy.
    """

    _index_attr = "_index"

    def __init__(self, snapfile: str = toysnap_filename, index: int = 0) -> None:
        self.snapfile = snapfile
        self._index = index
        super().__init__()
        return

    def _load(self) -> None:
        """
        Any non-trivial i/o operations needed at initialization go here.
        """
        return

    @property
    def index(self) -> int:
        """
        The position in the catalogue of the target galaxy.

        Returns
        -------
        out : int
            The position in the catalogue of the target galaxy.
        """
        return self._mask_index()

    def _generate_spatial_mask(self, sg: SWIFTGalaxy) -> SWIFTMask:
        """
        Evaluate the spatial mask (:class:`~swiftsimio.masks.SWIFTMask`) for the target
        galaxy.

        Parameters
        ----------
        sg : :class:`~swiftgalaxy.reader.SWIFTGalaxy`
            The :class:`~swiftgalaxy.reader.SWIFTGalaxy` for which the mask is being
            evaluated.

        Returns
        -------
        out : swiftsimio.masks.SWIFTMask
            The spatial mask.
        """
        if self.index == 0:
            spatial_mask = cosmo_array(
                [[centre_1 - 0.1, centre_1 + 0.1] for ax in range(3)],
                u.Mpc,
                comoving=True,
                scale_factor=1.0,
                scale_exponent=1,
            )
        elif self.index == 1:
            spatial_mask = cosmo_array(
                [[centre_2 - 0.1, centre_2 + 0.1] for ax in range(3)],
                u.Mpc,
                comoving=True,
                scale_factor=1.0,
                scale_exponent=1,
            )
        swift_mask = swiftsimio.mask(self.snapfile, spatial_only=True)
        swift_mask.constrain_spatial(spatial_mask)
        return swift_mask

    def _generate_bound_only_mask(self, sg: SWIFTGalaxy) -> MaskCollection:
        """
        Evaluate the extra mask (to apply after the spatial mask) selecting particles
        belonging to the target galaxy.

        Parameters
        ----------
        sg : :class:`~swiftgalaxy.reader.SWIFTGalaxy`
            The :class:`~swiftgalaxy.reader.SWIFTGalaxy` for which the mask is being
            evaluated.

        Returns
        -------
        out : :class:`~swiftgalaxy.masks.MaskCollection`
            The extra mask.
        """
        # the two objects are in different cells, remember we're masking cell particles
        if self.index == 0:
            extra_mask = MaskCollection(
                gas=np.s_[-n_g_1:],
                dark_matter=np.s_[-n_dm_1:],
                stars=np.s_[...],
                black_holes=np.s_[...],
            )
        elif self.index == 1:
            extra_mask = MaskCollection(
                gas=np.s_[-n_g_2:],
                dark_matter=np.s_[-n_dm_2:],
                stars=np.s_[...],
                black_holes=np.s_[...],
            )

        return extra_mask

    @property
    def centre(self) -> cosmo_array:
        """
        Coordinate centre of the target galaxy.

        Returns
        -------
        out : :class:`~swiftsimio.objects.cosmo_array`
            The coordinate centre of the target galaxy.
        """
        if self.index == 0:
            return cosmo_array(
                [centre_1, centre_1, centre_1],
                u.Mpc,
                comoving=True,
                scale_factor=1.0,
                scale_exponent=1,
            )
        elif self.index == 1:
            return cosmo_array(
                [centre_2, centre_2, centre_2],
                u.Mpc,
                comoving=True,
                scale_factor=1.0,
                scale_exponent=1,
            )

    @property
    def velocity_centre(self) -> cosmo_array:
        """
        Velocity centre of the target galaxy.

        Returns
        -------
        out : :class:`~swiftsimio.objects.cosmo_array`
            The velocity centre of the target galaxy.
        """
        if self.index == 0:
            return cosmo_array(
                [vcentre_1, vcentre_1, vcentre_1],
                u.km / u.s,
                comoving=True,
                scale_factor=1.0,
                scale_exponent=1,
            )
        if self.index == 1:
            return cosmo_array(
                [vcentre_2, vcentre_2, vcentre_2],
                u.km / u.s,
                comoving=True,
                scale_factor=1.0,
                scale_exponent=1,
            )

    @property
    def _region_centre(self) -> cosmo_array:
        """
        Centre of the bounding box that defines the spatial mask for the target galaxy.

        Returns
        -------
        out : :class:`~swiftsimio.objects.cosmo_array`
            The bounding box centre.
        """
        return cosmo_array(
            [[centre_1, centre_1, centre_1], [centre_2, centre_2, centre_2]],
            u.Mpc,
            comoving=True,
            scale_factor=1.0,
            scale_exponent=1,
        )[(self.index,)]

    @property
    def _region_aperture(self) -> cosmo_array:
        """
        Half-side length of the bounding box that defines the spatial mask for the target
        galaxy.

        Returns
        -------
        out : :class:`~swiftsimio.objects.cosmo_array`
            The half-length of the bounding box used to construct the spatial mask.
        """
        return cosmo_array(
            [0.5, 0.5],
            u.Mpc,
            comoving=True,
            scale_factor=1.0,
            scale_exponent=1,
        )[(self.index,)]

    def _get_preload_fields(self, server: SWIFTGalaxy) -> set:
        """
        Preload data needed to evaluate masks when in multi-galaxy mode.

        Parameters
        ----------
        server : :class:`~swiftgalaxy.reader.SWIFTGalaxy`
            The server object spawned by :class:`swiftgalaxy.iterator.SWIFTGalaxies`.

        Returns
        -------
        out : :obj:`set`
            The set of fields to preload.
        """
        return set()


def create_toysnap(
    snapfile: str = toysnap_filename,
    alt_coord_name: Optional[str] = None,
    alt_vel_name: Optional[str] = None,
    alt_id_name: Optional[str] = None,
    withfof: bool = False,
) -> None:
    """
    Creates a sample 'snapshot file' dataset containing 2 galaxies and a 'background' of
    unassigned particles.

    The data are created entirely "by hand" by drawing from random distributions (uniform,
    exponential disk, etc.). They are not the result of any actual simulation. Their
    purpose is to illustrate :mod:`swiftgalaxy` use by providing files with formats
    identical to actual SWIFT snapshot files without the need for additional downloads.

    Parameters
    ----------
    snapfile : :obj:`str`
        Filename for snapshot file. (Default: ``"toysnap.hdf5"``)

    alt_coord_name : :obj:`str`
        Intended for continuous integration testing purposes. Create additional
        coordinate-like data arrays with this name. (Default: ``None``)

    alt_vel_name : :obj:`str`
        Intended for continuous integration testing purposes. Create additional
        velocity-like data arrays with this name. (Default: ``None``)

    alt_id_name : :obj:`str`
        Intended for continuous integration testing purposes. Create additional
        particle ID-like data arrays with this name. (Default: ``None``)

    withfof : :obj:`bool`
        If ``True``, include friends-of-friends (FOF) group identifiers for each
        particle.
    """
    if os.path.isfile(snapfile):
        return

    sd = Writer(cosmo_units, np.ones(3, dtype=float) * boxsize)

    # Insert a uniform gas background plus two galaxy discs
    phi_1 = np.random.rand(n_g_1, 1) * 2 * np.pi
    R_1 = np.random.rand(n_g_1, 1)
    phi_2 = np.random.rand(n_g_2, 1) * 2 * np.pi
    R_2 = np.random.rand(n_g_2, 1)
    getattr(sd, present_particle_types[0]).particle_ids = np.arange(n_g_all)
    getattr(sd, present_particle_types[0]).coordinates = (
        np.vstack(
            (
                np.random.rand(n_g_b // 2, 3) * np.array([5, 10, 10]),
                np.hstack(
                    (
                        # 10 kpc disc radius, offcentred in box
                        centre_1 + R_1 * np.cos(phi_1) * 0.01,
                        centre_1 + R_1 * np.sin(phi_1) * 0.01,
                        centre_1
                        + (np.random.rand(n_g_1, 1) * 2 - 1) * 0.001,  # 1 kpc height
                    )
                ),
                np.random.rand(n_g_b // 2, 3) * np.array([5, 10, 10])
                + np.array([5, 0, 0]),
                np.hstack(
                    (
                        # 10 kpc disc radius, offcentred in box
                        centre_2 + R_2 * np.cos(phi_2) * 0.01,
                        centre_2 + R_2 * np.sin(phi_2) * 0.01,
                        centre_2
                        + (np.random.rand(n_g_2, 1) * 2 - 1) * 0.001,  # 1 kpc height
                    )
                ),
            )
        )
        * u.Mpc
    )
    getattr(sd, present_particle_types[0]).velocities = (
        np.vstack(
            (
                np.random.rand(n_g_b // 2, 3) * 2 - 1,  # 1 km/s for background
                np.hstack(
                    (
                        # solid body, 100 km/s at edge
                        vcentre_1 + R_1 * np.sin(phi_1) * 100,
                        vcentre_1 + R_1 * np.cos(phi_1) * 100,
                        vcentre_1
                        + np.random.rand(n_g_1, 1) * 20
                        - 10,  # 10 km/s vertical
                    )
                ),
                np.random.rand(n_g_b // 2, 3) * 2 - 1,  # 1 km/s for background
                np.hstack(
                    (
                        # solid body, 100 km/s at edge
                        vcentre_2 + R_2 * np.sin(phi_2) * 100,
                        vcentre_2 + R_2 * np.cos(phi_2) * 100,
                        vcentre_2
                        + np.random.rand(n_g_2, 1) * 20
                        - 10,  # 10 km/s vertical
                    )
                ),
            )
        )
        * u.km
        / u.s
    )
    getattr(sd, present_particle_types[0]).masses = np.ones(n_g_all, dtype=float) * m_g
    getattr(sd, present_particle_types[0]).internal_energy = (
        np.concatenate(
            (
                np.ones(n_g_b // 2, dtype=float),  # 1e4 K
                np.ones(n_g_1, dtype=float) / 10,  # 1e3 K
                np.ones(n_g_b // 2, dtype=float),  # 1e4 K
                np.ones(n_g_2, dtype=float) / 10,  # 1e3 K
            )
        )
        * T_g
        * u.kb
        / (m_g)
    )
    getattr(sd, present_particle_types[0]).generate_smoothing_lengths(
        boxsize=boxsize, dimension=3
    )

    # Insert a uniform DM background plus two galaxy halos
    phi_1 = np.random.rand(n_dm_1, 1) * 2 * np.pi
    theta_1 = np.arccos(np.random.rand(n_dm_1, 1) * 2 - 1)
    r_1 = np.random.rand(n_dm_1, 1)
    phi_2 = np.random.rand(n_dm_2, 1) * 2 * np.pi
    theta_2 = np.arccos(np.random.rand(n_dm_2, 1) * 2 - 1)
    r_2 = np.random.rand(n_dm_2, 1)
    getattr(sd, present_particle_types[1]).particle_ids = np.arange(
        n_g_all, n_g_all + n_dm_all
    )
    getattr(sd, present_particle_types[1]).coordinates = (
        np.vstack(
            (
                np.random.rand(n_dm_b // 2, 3) * np.array([5, 10, 10]),
                np.hstack(
                    (
                        # 100 kpc halo radius, offcentred in box
                        centre_1 + r_1 * np.cos(phi_1) * np.sin(theta_1) * 0.1,
                        centre_1 + r_1 * np.sin(phi_1) * np.sin(theta_1) * 0.1,
                        centre_1 + r_1 * np.cos(theta_1) * 0.1,
                    )
                ),
                np.random.rand(n_dm_b // 2, 3) * np.array([5, 10, 10])
                + np.array([5, 0, 0]),
                np.hstack(
                    (
                        # 100 kpc halo radius, offcentred in box
                        centre_2 + r_2 * np.cos(phi_2) * np.sin(theta_2) * 0.1,
                        centre_2 + r_2 * np.sin(phi_2) * np.sin(theta_2) * 0.1,
                        centre_2 + r_2 * np.cos(theta_2) * 0.1,
                    )
                ),
            )
        )
        * u.Mpc
    )
    getattr(sd, present_particle_types[1]).velocities = (
        np.vstack(
            (
                # 1 km/s background, 100 km/s halo
                np.random.rand(n_dm_b // 2, 3) * 2 - 1,
                vcentre_1 + (np.random.rand(n_dm_1, 3) * 2 - 1) * 100,
                np.random.rand(n_dm_b // 2, 3) * 2 - 1,
                vcentre_2 + (np.random.rand(n_dm_2, 3) * 2 - 1) * 100,
            )
        )
        * u.km
        / u.s
    )
    getattr(sd, present_particle_types[1]).masses = (
        np.ones(n_dm_all, dtype=float) * m_dm
    )
    getattr(sd, present_particle_types[1]).generate_smoothing_lengths(
        boxsize=boxsize, dimension=3
    )

    # Insert two galaxy stellar discs
    phi_1 = np.random.rand(n_s_1, 1) * 2 * np.pi
    R_1 = np.random.rand(n_s_1, 1)
    phi_2 = np.random.rand(n_s_2, 1) * 2 * np.pi
    R_2 = np.random.rand(n_s_2, 1)
    getattr(sd, present_particle_types[4]).particle_ids = np.arange(
        n_g_all + n_dm_all, n_g_all + n_dm_all + n_s_1 + n_s_2
    )
    getattr(sd, present_particle_types[4]).coordinates = (
        np.vstack(
            (
                np.hstack(
                    (
                        # 5 kpc disc radius, offcentred in box
                        centre_1 + R_1 * np.cos(phi_1) * 0.005,
                        centre_1 + R_1 * np.sin(phi_1) * 0.005,
                        centre_1
                        + (np.random.rand(n_s_1, 1) * 2 - 1) * 0.0005,  # 500 pc height
                    )
                ),
                np.hstack(
                    (
                        # 5 kpc disc radius, offcentred in box
                        centre_2 + R_2 * np.cos(phi_2) * 0.005,
                        centre_2 + R_2 * np.sin(phi_2) * 0.005,
                        centre_2
                        + (np.random.rand(n_s_2, 1) * 2 - 1) * 0.0005,  # 500 pc height
                    )
                ),
            )
        )
        * u.Mpc
    )
    getattr(sd, present_particle_types[4]).velocities = (
        np.vstack(
            (
                np.hstack(
                    (
                        # solid body, 50 km/s at edge
                        vcentre_1 + R_1 * np.sin(phi_1) * 50,
                        vcentre_1 + R_1 * np.cos(phi_1) * 50,
                        vcentre_1
                        + np.random.rand(n_s_1, 1) * 20
                        - 10,  # 10 km/s vertical motions
                    )
                ),
                np.hstack(
                    (
                        # solid body, 50 km/s at edge
                        vcentre_2 + R_2 * np.sin(phi_2) * 50,
                        vcentre_2 + R_2 * np.cos(phi_2) * 50,
                        vcentre_2
                        + np.random.rand(n_s_2, 1) * 20
                        - 10,  # 10 km/s vertical motions
                    )
                ),
            )
        )
        * u.km
        / u.s
    )
    getattr(sd, present_particle_types[4]).masses = (
        np.ones(n_s_1 + n_s_2, dtype=float) * m_s
    )
    getattr(sd, present_particle_types[4]).generate_smoothing_lengths(
        boxsize=boxsize, dimension=3
    )
    # Insert a black hole in two galaxies
    getattr(sd, present_particle_types[5]).particle_ids = np.arange(
        n_g_all + n_dm_all + n_s_1 + n_s_2,
        n_g_all + n_dm_all + n_s_1 + n_s_2 + n_bh_1 + n_bh_2,
    )
    getattr(sd, present_particle_types[5]).coordinates = (
        np.concatenate(
            (
                centre_1
                - 0.000003
                * np.ones((n_bh_1, 3), dtype=float),  # 3 pc to avoid r==0 warnings
                centre_2
                - 0.000003
                * np.ones((n_bh_2, 3), dtype=float),  # 3 pc to avoid r==0 warnings
            )
        )
        * u.Mpc
    )
    getattr(sd, present_particle_types[5]).velocities = (
        np.concatenate(
            (
                vcentre_1 + np.zeros((n_bh_1, 3), dtype=float),
                vcentre_2 + np.zeros((n_bh_2, 3), dtype=float),
            )
        )
        * u.km
        / u.s
    )
    getattr(sd, present_particle_types[5]).masses = (
        np.ones(n_bh_1 + n_bh_2, dtype=float) * m_bh
    )
    getattr(sd, present_particle_types[5]).generate_smoothing_lengths(
        boxsize=boxsize, dimension=3
    )

    sd.write(snapfile)  # IDs auto-generated

    with h5py.File(snapfile, "r+") as f:
        g = f.create_group("Cells")
        g.create_dataset(
            "Centres", data=np.array([[2.5, 5, 5], [7.5, 5, 5]], dtype=float)
        )
        cg = g.create_group("Counts")
        cg.create_dataset(
            "PartType0",
            data=np.array(
                [
                    np.sum(
                        getattr(sd, present_particle_types[0]).coordinates[:, 0] <= 5
                    ),
                    np.sum(
                        getattr(sd, present_particle_types[0]).coordinates[:, 0] > 5
                    ),
                ]
            ),
            dtype=int,
        )
        cg.create_dataset(
            "PartType1",
            data=np.array(
                [
                    np.sum(
                        getattr(sd, present_particle_types[1]).coordinates[:, 0] <= 5
                    ),
                    np.sum(
                        getattr(sd, present_particle_types[1]).coordinates[:, 0] > 5
                    ),
                ]
            ),
            dtype=int,
        )
        cg.create_dataset(
            "PartType4",
            data=np.array(
                [
                    np.sum(
                        getattr(sd, present_particle_types[4]).coordinates[:, 0] <= 5
                    ),
                    np.sum(
                        getattr(sd, present_particle_types[4]).coordinates[:, 0] > 5
                    ),
                ]
            ),
            dtype=int,
        )
        cg.create_dataset(
            "PartType5",
            data=np.array(
                [
                    np.sum(
                        getattr(sd, present_particle_types[5]).coordinates[:, 0] <= 5
                    ),
                    np.sum(
                        getattr(sd, present_particle_types[5]).coordinates[:, 0] > 5
                    ),
                ]
            ),
            dtype=int,
        )
        fg = g.create_group("Files")
        fg.create_dataset("PartType0", data=np.array([0, 0], dtype=int))
        fg.create_dataset("PartType1", data=np.array([0, 0], dtype=int))
        fg.create_dataset("PartType4", data=np.array([0, 0], dtype=int))
        fg.create_dataset("PartType5", data=np.array([0, 0], dtype=int))
        bbming = g.create_group("MinPositions")
        bbming.create_dataset(
            "PartType0", data=np.array([[0, 0, 0], [5, 0, 0]], dtype=int)
        )
        bbming.create_dataset(
            "PartType1", data=np.array([[0, 0, 0], [5, 0, 0]], dtype=int)
        )
        bbming.create_dataset(
            "PartType4", data=np.array([[0, 0, 0], [5, 0, 0]], dtype=int)
        )
        bbming.create_dataset(
            "PartType5", data=np.array([[0, 0, 0], [5, 0, 0]], dtype=int)
        )
        bbmaxg = g.create_group("MaxPositions")
        bbmaxg.create_dataset(
            "PartType0", data=np.array([[5, 10, 10], [10, 10, 10]], dtype=int)
        )
        bbmaxg.create_dataset(
            "PartType1", data=np.array([[5, 10, 10], [10, 10, 10]], dtype=int)
        )
        bbmaxg.create_dataset(
            "PartType4", data=np.array([[5, 10, 10], [10, 10, 10]], dtype=int)
        )
        bbmaxg.create_dataset(
            "PartType5", data=np.array([[5, 10, 10], [10, 10, 10]], dtype=int)
        )
        mdg = g.create_group("Meta-data")
        mdg.attrs["dimension"] = np.array([[2, 1, 1]], dtype=int)
        mdg.attrs["nr_cells"] = np.array([2], dtype=int)
        mdg.attrs["size"] = np.array(
            [
                0.5 * boxsize.to_value(u.Mpc),
                boxsize.to_value(u.Mpc),
                boxsize.to_value(u.Mpc),
            ],
            dtype=int,
        )
        og = g.create_group("OffsetsInFile")
        og.create_dataset(
            "PartType0",
            data=np.array(
                [
                    0,
                    np.sum(
                        getattr(sd, present_particle_types[0]).coordinates[:, 0] <= 5
                    ),
                ],
                dtype=int,
            ),
        )
        og.create_dataset(
            "PartType1",
            data=np.array(
                [
                    0,
                    np.sum(
                        getattr(sd, present_particle_types[1]).coordinates[:, 0] <= 5
                    ),
                ],
                dtype=int,
            ),
        )
        og.create_dataset(
            "PartType4",
            data=np.array(
                [
                    0,
                    np.sum(
                        getattr(sd, present_particle_types[4]).coordinates[:, 0] <= 5
                    ),
                ],
                dtype=int,
            ),
        )
        og.create_dataset(
            "PartType5",
            data=np.array(
                [
                    0,
                    np.sum(
                        getattr(sd, present_particle_types[5]).coordinates[:, 0] <= 5
                    ),
                ],
                dtype=int,
            ),
        )
        hsg = f.create_group("HydroScheme")
        hsg.attrs["Adiabatic index"] = 5.0 / 3.0

        for pt in (0, 1, 4, 5):
            g = f[f"PartType{pt}"]
            g["ExtraCoordinates"] = g["Coordinates"]
            g["ExtraVelocities"] = g["Velocities"]
            if alt_id_name is not None:
                g[alt_id_name] = g["ParticleIDs"]
                del g["ParticleIDs"]
            if alt_coord_name is not None:
                g[alt_coord_name] = g["Coordinates"]
                del g["Coordinates"]
            if alt_vel_name is not None:
                g[alt_vel_name] = g["Velocities"]
                del g["Velocities"]

        ssg = f.create_group("SubgridScheme")
        ncg = ssg.create_group("NamedColumns")
        ncg.create_dataset(
            "HydrogenIonizationFractions",
            data=np.array([b"Neutral", b"Ionized"], dtype="|S32"),
        )
        g = f["PartType0"]
        f_neutral = np.random.rand(n_g_all)
        f_ion = 1 - f_neutral
        hifd = g.create_dataset(
            "HydrogenIonizationFractions",
            data=np.array([f_neutral, f_ion], dtype=float).T,
        )
        hifd.attrs[
            "Conversion factor to CGS" " (not including cosmological corrections)"
        ] = np.array([1.0], dtype=float)
        hifd.attrs[
            "Conversion factor to physical CGS" " (including cosmological corrections)"
        ] = np.array([1.0], dtype=float)
        hifd.attrs["U_I exponent"] = np.array([0.0], dtype=float)
        hifd.attrs["U_L exponent"] = np.array([0.0], dtype=float)
        hifd.attrs["U_M exponent"] = np.array([0.0], dtype=float)
        hifd.attrs["U_T exponent"] = np.array([0.0], dtype=float)
        hifd.attrs["U_t exponent"] = np.array([0.0], dtype=float)
        hifd.attrs["a-scale exponent"] = np.array([0.0], dtype=float)
        hifd.attrs["h-scale exponent"] = np.array([0.0], dtype=float)
        if withfof:
            for ptype, n_group_1, n_group_2, n_notgroup in [
                (0, n_g_1, n_g_2, n_g_b),
                (1, n_dm_1, n_dm_2, n_dm_b),
                (4, n_s_1, n_s_2, 0),
                (5, n_bh_1, n_bh_2, 0),
            ]:
                f[f"PartType{ptype}"].create_dataset(
                    "FOFGroupIDs",
                    data=np.concatenate(
                        (
                            np.ones(n_notgroup // 2, dtype=int) * 2**31 - 1,
                            np.ones(n_group_1, dtype=int),
                            np.ones(n_notgroup // 2, dtype=int) * 2**31 - 1,
                            np.ones(n_group_2, dtype=int) * 2,
                        )
                    ),
                    dtype=int,
                )
                f[f"PartType{ptype}/FOFGroupIDs"].attrs[
                    "Conversion factor to CGS "
                    "(not including cosmological corrections)"
                ] = np.array([1.0])
                f[f"PartType{ptype}/FOFGroupIDs"].attrs[
                    "Conversion factor to physical CGS "
                    "(including cosmological corrections)"
                ] = np.array([1.0])
                f[f"PartType{ptype}/FOFGroupIDs"].attrs[
                    "Description"
                ] = b"Friends-Of-Friends ID of the group the particles belong to"
                f[f"PartType{ptype}/FOFGroupIDs"].attrs[
                    "Expression for physical CGS units"
                ] = b"[ - ] "
                f[f"PartType{ptype}/FOFGroupIDs"].attrs[
                    "Lossy compression filter"
                ] = b"None"
                f[f"PartType{ptype}/FOFGroupIDs"].attrs["U_I exponent"] = np.array(
                    [0.0]
                )
                f[f"PartType{ptype}/FOFGroupIDs"].attrs["U_L exponent"] = np.array(
                    [0.0]
                )
                f[f"PartType{ptype}/FOFGroupIDs"].attrs["U_M exponent"] = np.array(
                    [0.0]
                )
                f[f"PartType{ptype}/FOFGroupIDs"].attrs["U_T exponent"] = np.array(
                    [0.0]
                )
                f[f"PartType{ptype}/FOFGroupIDs"].attrs["U_t exponent"] = np.array(
                    [0.0]
                )
                f[f"PartType{ptype}/FOFGroupIDs"].attrs["a-scale exponent"] = np.array(
                    [0.0]
                )
                f[f"PartType{ptype}/FOFGroupIDs"].attrs["h-scale exponent"] = np.array(
                    [0.0]
                )

    return


def remove_toysnap(snapfile: str = toysnap_filename) -> None:
    """
    Removes file created by :func:`~swiftgalaxy.demo_data.create_toysnap`.

    Parameters
    ----------
    snapfile : :obj:`str`
        Filename for snapshot file. (Default: ``"toysnap.hdf5"``)
    """
    if os.path.isfile(snapfile):
        os.remove(snapfile)
    return


def create_toyvr(filebase: str = toyvr_filebase) -> None:
    """
    Creates a sample Velociraptor catalogue containing 2 galaxies matching the snapshot
    file created by :func:`~swiftgalaxy.demo_data.create_toysnap`.

    The data are created entirely "by hand". They are not the result of any actual
    simulation. Their purpose is to illustrate :mod:`swiftgalaxy` use by providing files
    with formats identical to actual Velociraptor catalogue files without the need for
    additional downloads.

    Parameters
    ----------
    filebase : :obj:`str`
        The base name for catalogue files (several files ``base.properties``,
        ``base.catalog_groups``, etc. will be created). (Default: ``"toyvr"``)
    """
    with h5py.File(f"{toyvr_filebase}.properties", "w") as f:
        f.create_group("SimulationInfo")
        f["SimulationInfo"].attrs["ScaleFactor"] = 1.0
        f["SimulationInfo"].attrs["Cosmological_Sim"] = 1
        for coord in "XYZ":
            f.create_dataset(
                f"{coord}c", data=np.array([centre_1, centre_2], dtype=float)
            )
            (
                f.create_dataset(
                    f"{coord}cminpot",
                    data=np.array([centre_1, centre_2], dtype=float) + 0.001,
                )
            )
            (
                f.create_dataset(
                    f"{coord}cmbp",
                    data=np.array([centre_1, centre_2], dtype=float) + 0.002,
                )
            )
            f.create_dataset(
                f"{coord}c_gas", data=np.array([0.003, 0.003], dtype=float)
            )
            f.create_dataset(
                f"{coord}c_stars", data=np.array([0.004, 0.004], dtype=float)
            )
            f.create_dataset(
                f"V{coord}c", data=np.array([vcentre_1, vcentre_2], dtype=float)
            )
            (
                f.create_dataset(
                    f"V{coord}cminpot",
                    data=np.array([vcentre_1, vcentre_2], dtype=float) + 1.0,
                )
            )
            (
                f.create_dataset(
                    f"V{coord}cmbp",
                    data=np.array([vcentre_1, vcentre_2], dtype=float) + 2.0,
                )
            )
            f.create_dataset(f"V{coord}c_gas", data=np.array([3.0, 3.0], dtype=float))
            f.create_dataset(f"V{coord}c_stars", data=np.array([4.0, 4.0], dtype=float))
            for ct in ("c", "cminpot", "cmbp", "c_gas", "c_stars"):
                f[f"{coord}{ct}"].attrs["Dimension_Length"] = 1.0
                f[f"{coord}{ct}"].attrs["Dimension_Mass"] = 0.0
                f[f"{coord}{ct}"].attrs["Dimension_Time"] = 0.0
                f[f"{coord}{ct}"].attrs["Dimension_Velocity"] = 0.0
                f[f"V{coord}{ct}"].attrs["Dimension_Length"] = 0.0
                f[f"V{coord}{ct}"].attrs["Dimension_Mass"] = 0.0
                f[f"V{coord}{ct}"].attrs["Dimension_Time"] = 0.0
                f[f"V{coord}{ct}"].attrs["Dimension_Velocity"] = 1.0
        f.create_group("Configuration")
        f["Configuration"].attrs["h_val"] = h0
        f["Configuration"].attrs["w_of_DE"] = w0
        f["Configuration"].attrs["Omega_DE"] = Om_l
        f["Configuration"].attrs["Omega_b"] = Om_b
        f["Configuration"].attrs["Omega_m"] = Om_m
        f["Configuration"].attrs["Period"] = boxsize.to_value(u.Mpc)
        f.create_dataset("File_id", data=np.array([0, 0], dtype=int))
        f.create_dataset("ID", data=np.array([1, 2], dtype=int))
        f["ID"].attrs["Dimension_Length"] = 0.0
        f["ID"].attrs["Dimension_Mass"] = 0.0
        f["ID"].attrs["Dimension_Time"] = 0.0
        f["ID"].attrs["Dimension_Velocity"] = 0.0
        # pick arbitrary particle in the galaxy to be most bound
        f.create_dataset(
            "ID_mbp",
            data=np.array([32**3 // 2 - n_g_1 + 1, 32**3 - n_g_2 + 1], dtype=int),
        )
        f["ID_mbp"].attrs["Dimension_Length"] = 0.0
        f["ID_mbp"].attrs["Dimension_Mass"] = 0.0
        f["ID_mbp"].attrs["Dimension_Time"] = 0.0
        f["ID_mbp"].attrs["Dimension_Velocity"] = 0.0
        # pick arbitrary particle in the galaxy to be potential minimum
        f.create_dataset(
            "ID_minpot",
            data=np.array([32**3 // 2 - n_g_1 + 2, 32**3 - n_g_2 + 2], dtype=int),
        )
        f["ID_minpot"].attrs["Dimension_Length"] = 0.0
        f["ID_minpot"].attrs["Dimension_Mass"] = 0.0
        f["ID_minpot"].attrs["Dimension_Time"] = 0.0
        f["ID_minpot"].attrs["Dimension_Velocity"] = 0.0
        f.create_dataset("Mvir", data=np.array([100.0, 110.0], dtype=float))
        f.create_dataset("Mass_200crit", data=np.array([100.0, 110.0], dtype=float))
        f.create_dataset("Mass_200mean", data=np.array([100.0, 110.0], dtype=float))
        f.create_dataset("Mass_BN98", data=np.array([100.0, 110.0], dtype=float))
        f.create_dataset("Mass_FOF", data=np.array([100.0, 110.0], dtype=float))
        for field in ("Mvir", "Mass_200crit", "Mass_200mean", "Mass_BN98", "Mass_FOF"):
            f[field].attrs["Dimension_Length"] = 0.0
            f[field].attrs["Dimension_Mass"] = 1.0
            f[field].attrs["Dimension_Time"] = 0.0
            f[field].attrs["Dimension_Velocity"] = 0.0
        f.create_dataset("R_200crit", data=np.array([0.3, 0.35], dtype=float))
        f.create_dataset("R_200mean", data=np.array([0.3, 0.35], dtype=float))
        f.create_dataset("R_BN98", data=np.array([0.3, 0.35], dtype=float))
        f.create_dataset("R_size", data=np.array([0.3, 0.35], dtype=float))
        f.create_dataset("Rmax", data=np.array([0.3, 0.35], dtype=float))
        f.create_dataset("Rvir", data=np.array([0.3, 0.35], dtype=float))
        for field in ("R_200crit", "R_200mean", "R_BN98", "R_size", "Rmax", "Rvir"):
            f[field].attrs["Dimension_Length"] = 1.0
            f[field].attrs["Dimension_Mass"] = 0.0
            f[field].attrs["Dimension_Time"] = 0.0
            f[field].attrs["Dimension_Velocity"] = 0.0
        f.create_dataset("Num_of_files", data=np.array([1], dtype=int))
        f.create_dataset("Num_of_groups", data=np.array([2], dtype=int))
        f.create_dataset("Structuretype", data=np.array([10, 10], dtype=int))
        f["Structuretype"].attrs["Dimension_Length"] = 0.0
        f["Structuretype"].attrs["Dimension_Mass"] = 0.0
        f["Structuretype"].attrs["Dimension_Time"] = 0.0
        f["Structuretype"].attrs["Dimension_Velocity"] = 0.0
        f.create_dataset("Total_num_of_groups", data=np.array([2], dtype=int))
        f.create_group("UnitInfo")
        # have not checked UnitInfo in detail
        f["UnitInfo"].attrs["Comoving_or_Physical"] = b"0"
        f["UnitInfo"].attrs["Cosmological_Sim"] = b"1"
        f["UnitInfo"].attrs["Length_unit_to_kpc"] = b"1000.000000"
        f["UnitInfo"].attrs["Mass_unit_to_solarmass"] = b"10000000000.000000"
        f["UnitInfo"].attrs["Metallicity_unit_to_solar"] = b"83.330000"
        f["UnitInfo"].attrs["SFR_unit_to_solarmassperyear"] = b"97.780000"
        f["UnitInfo"].attrs["Stellar_age_unit_to_yr"] = b"977813413600.000000"
        f["UnitInfo"].attrs["Velocity_unit_to_kms"] = b"1.000000"
        f.attrs["Comoving_or_Physical"] = 0
        f.attrs["Cosmological_Sim"] = 1
        f.attrs["Length_unit_to_kpc"] = 1000.000000
        f.attrs["Mass_unit_to_solarmass"] = 10000000000.000000
        f.attrs["Metallicity_unit_to_solar"] = 83.330000
        f.attrs["Period"] = boxsize.to_value(u.Mpc)
        f.attrs["SFR_unit_to_solarmassperyear"] = 97.780000
        f.attrs["Stellar_age_unit_to_yr"] = 977813413600.000000
        f.attrs["Time"] = 1.0
        f.attrs["Velocity_to_kms"] = 1.000000
        f.create_dataset("hostHaloID", data=np.array([-1], dtype=int))
        f["hostHaloID"].attrs["Dimension_Length"] = 0.0
        f["hostHaloID"].attrs["Dimension_Mass"] = 0.0
        f["hostHaloID"].attrs["Dimension_Time"] = 0.0
        f["hostHaloID"].attrs["Dimension_Velocity"] = 0.0
        f.create_dataset("n_bh", data=np.array([n_bh_1, n_bh_2], dtype=int))
        f.create_dataset("n_gas", data=np.array([n_g_1, n_g_2], dtype=int))
        f.create_dataset("n_star", data=np.array([n_s_1, n_s_2], dtype=int))
        f.create_dataset(
            "npart",
            data=np.array(
                [n_g_1 + n_dm_1 + n_s_1 + n_bh_1, n_g_2 + n_dm_2 + n_s_2 + n_bh_2],
                dtype=int,
            ),
        )
        for pt in ("_bh", "_gas", "_star", "part"):
            f[f"n{pt}"].attrs["Dimension_Length"] = 0.0
            f[f"n{pt}"].attrs["Dimension_Mass"] = 0.0
            f[f"n{pt}"].attrs["Dimension_Time"] = 0.0
            f[f"n{pt}"].attrs["Dimension_Velocity"] = 0.0
        f.create_dataset("numSubStruct", data=np.array([0, 0], dtype=int))
        f["numSubStruct"].attrs["Dimension_Length"] = 0.0
        f["numSubStruct"].attrs["Dimension_Mass"] = 0.0
        f["numSubStruct"].attrs["Dimension_Time"] = 0.0
        f["numSubStruct"].attrs["Dimension_Velocity"] = 0.0
    with h5py.File(f"{toyvr_filebase}.catalog_groups", "w") as f:
        f.create_dataset("File_id", data=np.array([0, 0], dtype=int))
        f.create_dataset(
            "Group_Size",
            data=np.array(
                [
                    n_g_all // 2 + n_dm_all // 2 + n_s_1 + n_bh_2,
                    n_g_all // 2 + n_dm_all // 2 + n_s_2 + n_bh_2,
                ],
                dtype=int,
            ),
        )
        f.create_dataset("Num_of_files", data=np.array([1], dtype=int))
        f.create_dataset("Num_of_groups", data=np.array([2], dtype=int))
        f.create_dataset(
            "Number_of_substructures_in_halo", data=np.array([0, 0], dtype=int)
        )
        f.create_dataset(
            "Offset", data=np.array([0, n_g_1 + n_dm_1 + n_s_1 + n_bh_1], dtype=int)
        )
        f.create_dataset(
            "Offset_unbound", data=np.array([0, n_g_b // 2 + n_dm_b // 2], dtype=int)
        )
        f.create_dataset("Parent_halo_ID", data=np.array([-1, -1], dtype=int))
        f.create_dataset("Total_num_of_groups", data=np.array([2], dtype=int))
    with h5py.File(f"{toyvr_filebase}.catalog_particles", "w") as f:
        f.create_dataset("File_id", data=np.array([0], dtype=int))
        f.create_dataset("Num_of_files", data=np.array([1], dtype=int))
        f.create_dataset(
            "Num_of_particles_in_groups",
            data=np.array(
                [n_g_1 + n_dm_1 + n_s_1 + n_bh_1, n_g_2 + n_dm_2 + n_s_2 + n_bh_2],
                dtype=int,
            ),
        )
        f.create_dataset(
            "Particle_IDs",
            data=np.concatenate(
                (
                    # gas IDs group 0
                    np.arange(n_g_b // 2, n_g_b // 2 + n_g_1, dtype=int),
                    # dm IDs group 0
                    np.arange(
                        n_g_all + n_dm_b // 2, n_g_all + n_dm_b // 2 + n_dm_1, dtype=int
                    ),
                    # star IDs group 0
                    np.arange(
                        n_g_all + n_dm_all, n_g_all + n_dm_all + n_s_1, dtype=int
                    ),
                    # bh IDs group 0
                    np.arange(
                        n_g_all + n_dm_all + n_s_1 + n_s_2,
                        n_g_all + n_dm_all + n_s_1 + n_s_2 + n_bh_1,
                        dtype=int,
                    ),
                    # gas IDs group 1
                    np.arange(n_g_b + n_g_1, n_g_all, dtype=int),
                    # dm IDs group 1
                    np.arange(n_g_all + n_dm_b + n_dm_1, n_g_all + n_dm_all, dtype=int),
                    # star IDs group 1
                    np.arange(
                        n_g_all + n_dm_all + n_s_1,
                        n_g_all + n_dm_all + n_s_1 + n_s_2,
                        dtype=int,
                    ),
                    # bh IDs group 1
                    np.arange(
                        n_g_all + n_dm_all + n_s_1 + n_s_2 + n_bh_1,
                        n_g_all + n_dm_all + n_s_1 + n_s_2 + n_bh_1 + n_bh_2,
                        dtype=int,
                    ),
                )
            ),
        )
        f.create_dataset(
            "Total_num_of_particles_in_all_groups",
            data=np.array(
                [n_g_1 + n_g_2 + n_dm_1 + n_dm_2 + n_s_1 + n_s_2 + n_bh_1 + n_bh_2],
                dtype=int,
            ),
        )
    with h5py.File(f"{toyvr_filebase}.catalog_particles.unbound", "w") as f:
        f.create_dataset("File_id", data=np.array([0], dtype=int))
        f.create_dataset("Num_of_files", data=np.array([1], dtype=int))
        f.create_dataset(
            "Num_of_particles_in_groups",
            data=np.array([n_g_b // 2 + n_dm_b // 2, n_g_b // 2 + n_dm_b // 2]),
        )
        f.create_dataset(
            "Particle_IDs",
            data=np.concatenate(
                (
                    np.arange(n_g_b // 2, dtype=int),
                    np.arange(n_g_b // 2 + n_g_1, n_g_all - n_g_2, dtype=int),
                    np.arange(n_g_all, n_g_all + n_dm_b // 2, dtype=int),
                    np.arange(
                        n_g_all + n_dm_b // 2 + n_dm_1,
                        n_g_all + n_dm_all - n_dm_2,
                        dtype=int,
                    ),
                )
            ),
        )
        f.create_dataset(
            "Total_num_of_particles_in_all_groups",
            data=np.array([n_g_b + n_dm_b], dtype=int),
        )
    with h5py.File(f"{toyvr_filebase}.catalog_parttypes", "w") as f:
        f.create_dataset("File_id", data=np.array([0], dtype=int))
        f.create_dataset("Num_of_files", data=np.array([1], dtype=int))
        f.create_dataset(
            "Num_of_particles_in_groups",
            data=np.array(
                [n_g_1 + n_dm_1 + n_s_1 + n_bh_1, n_g_2 + n_dm_2 + n_s_2 + n_bh_2],
                dtype=int,
            ),
        )
        f.create_dataset(
            "Particle_types",
            data=np.concatenate(
                (
                    0 * np.ones(n_g_1, dtype=int),
                    1 * np.ones(n_dm_1, dtype=int),
                    4 * np.ones(n_s_1, dtype=int),
                    5 * np.ones(n_bh_1, dtype=int),
                    0 * np.ones(n_g_2, dtype=int),
                    1 * np.ones(n_dm_2, dtype=int),
                    4 * np.ones(n_s_2, dtype=int),
                    5 * np.ones(n_bh_2, dtype=int),
                )
            ),
        )
        f.create_dataset(
            "Total_num_of_particles_in_all_groups",
            data=np.array(
                [n_g_1 + n_dm_1 + n_s_1 + n_bh_1 + n_g_2 + n_dm_2 + n_s_2 + n_bh_2],
                dtype=int,
            ),
        )
    with h5py.File(f"{toyvr_filebase}.catalog_parttypes.unbound", "w") as f:
        f.create_dataset("File_id", data=np.array([0], dtype=int))
        f.create_dataset("Num_of_files", data=np.array([1], dtype=int))
        f.create_dataset(
            "Num_of_particles_in_groups",
            data=np.array(
                [n_g_b // 2 + n_dm_b // 2, n_g_b // 2 + n_dm_b // 2], dtype=int
            ),
        )
        f.create_dataset(
            "Particle_types",
            data=np.concatenate(
                (0 * np.ones(n_g_b, dtype=int), 1 * np.ones(n_dm_b, dtype=int))
            ),
        )
        f.create_dataset(
            "Total_num_of_particles_in_all_groups",
            data=np.array([n_g_b + n_dm_b], dtype=int),
        )
    return


def remove_toyvr(filebase: str = toyvr_filebase) -> None:
    """
    Removes files created by :func:`~swiftgalaxy.demo_data.create_toyvr`.

    Parameters
    ----------
    filebase : :obj:`str`
        The base name for catalogue files (several files ``base.properties``,
        ``base.catalog_groups``, etc. will be removed). (Default: ``"toyvr"``)
    """
    os.remove(f"{toyvr_filebase}.properties")
    os.remove(f"{toyvr_filebase}.catalog_groups")
    os.remove(f"{toyvr_filebase}.catalog_particles")
    os.remove(f"{toyvr_filebase}.catalog_particles.unbound")
    os.remove(f"{toyvr_filebase}.catalog_parttypes")
    os.remove(f"{toyvr_filebase}.catalog_parttypes.unbound")
    return


def create_toycaesar(filename: str = toycaesar_filename) -> None:
    """
    Creates a sample Caesar catalogue containing 2 galaxies matching the snapshot
    file created by :func:`~swiftgalaxy.demo_data.create_toysnap`.

    The data are created entirely "by hand". They are not the result of any actual
    simulation. Their purpose is to illustrate :mod:`swiftgalaxy` use by providing files
    with formats identical to actual Caesar catalogue files without the need for
    additional downloads.

    Parameters
    ----------
    filename : :obj:`str`
        The file name for the catalogue file to be created.
        (Default: ``"toycaesar.hdf5"``)
    """
    with h5py.File(filename, "w") as f:
        f.attrs["caesar"] = "fake"
        f.attrs["nclouds"] = 0
        f.attrs["ngalaxies"] = 2
        f.attrs["nhalos"] = 2
        with open(
            os.path.join(os.path.dirname(__file__), "json/caesar_unit_registry.json")
        ) as json:
            f.attrs["unit_registry_json"] = json.read()
        f.create_group("galaxy_data")
        f["/galaxy_data"].create_dataset("GroupID", data=np.array([0, 1], dtype=int))
        f["/galaxy_data"].create_dataset(
            "bhlist_end", data=np.array([n_bh_1, n_bh_2], dtype=int)
        )
        f["/galaxy_data"].create_dataset(
            "bhlist_start", data=np.array([0, n_bh_1], dtype=int)
        )
        f["/galaxy_data"].create_dataset(
            "central", data=np.array([True, True], dtype=bool)
        )
        f["/galaxy_data"].create_group("dicts")
        f["/galaxy_data/dicts"].create_dataset(
            "masses.total",
            data=np.array(
                [
                    (n_g_1 * m_g + n_s_1 * m_s + n_bh_1 * m_bh).to_value(u.msun),
                    (n_g_2 * m_g + n_s_2 * m_s + n_bh_2 * m_bh).to_value(u.msun),
                ],
                dtype=float,
            ),
        )
        f["/galaxy_data/dicts/masses.total"].attrs["unit"] = "Msun"
        f["/galaxy_data/dicts"].create_dataset(
            "radii.total_rmax", data=np.array([100, 100], dtype=float)
        )
        f["/galaxy_data/dicts/radii.total_rmax"].attrs["unit"] = "kpccm"
        f["/galaxy_data"].create_dataset(
            "glist_end", data=np.array([n_g_1, n_g_1 + n_g_2], dtype=int)
        )
        f["/galaxy_data"].create_dataset(
            "glist_start", data=np.array([0, n_g_1], dtype=int)
        )
        f["/galaxy_data"].create_group("lists")
        f["/galaxy_data/lists"].create_dataset(
            "bhlist", data=np.array([0, 1], dtype=int)
        )
        f["/galaxy_data/lists"].create_dataset(
            "glist",
            data=np.concatenate(
                (
                    np.arange(n_g_b // 2, n_g_b // 2 + n_g_1, dtype=int),
                    np.arange(n_g_b + n_g_1, n_g_all, dtype=int),
                )
            ),
        )
        f["/galaxy_data/lists"].create_dataset(
            "slist", data=np.arange(n_s_1 + n_s_2, dtype=int)
        )
        f["/galaxy_data"].create_dataset(
            "minpotpos",
            data=np.array(
                [
                    [
                        centre_1 * 1000 + 1.0,
                        centre_1 * 1000 + 1.0,
                        centre_1 * 1000 + 1.0,
                    ],
                    [
                        centre_2 * 1000 + 1.0,
                        centre_2 * 1000 + 1.0,
                        centre_2 * 1000 + 1.0,
                    ],
                ],
                dtype=float,
            ),
        )
        f["/galaxy_data/minpotpos"].attrs["unit"] = "kpccm"
        f["/galaxy_data"].create_dataset(
            "minpotvel",
            data=np.array(
                [
                    [vcentre_1 + 1.0, vcentre_1 + 1.0, vcentre_1 + 1.0],
                    [vcentre_2 + 1.0, vcentre_2 + 1.0, vcentre_2 + 1.0],
                ],
                dtype=float,
            ),
        )
        f["/galaxy_data/minpotvel"].attrs["unit"] = "km/s"
        f["/galaxy_data"].create_dataset(
            "nbh", data=np.array([n_bh_1, n_bh_2], dtype=int)
        )
        f["/galaxy_data"].create_dataset("ndm", data=np.array([0, 0], dtype=int))
        f["/galaxy_data"].create_dataset("ndm2", data=np.array([0, 0], dtype=int))
        f["/galaxy_data"].create_dataset("ndm3", data=np.array([0, 0], dtype=int))
        f["/galaxy_data"].create_dataset("ndust", data=np.array([0, 0], dtype=int))
        f["/galaxy_data"].create_dataset(
            "ngas", data=np.array([n_g_1, n_g_2], dtype=int)
        )
        f["/galaxy_data"].create_dataset(
            "nstar", data=np.array([n_s_1, n_s_2], dtype=int)
        )
        f["/galaxy_data"].create_dataset(
            "parent_halo_index", data=np.array([0, 1], dtype=int)
        )
        f["/galaxy_data"].create_dataset(
            "pos",
            data=np.array(
                [
                    [centre_1 * 1000, centre_1 * 1000, centre_1 * 1000],
                    [centre_2 * 1000, centre_2 * 1000, centre_2 * 1000],
                ],
                dtype=float,
            ),
        )
        f["/galaxy_data/pos"].attrs["unit"] = "kpccm"
        f["/galaxy_data"].create_dataset(
            "slist_end", data=np.array([n_s_1, n_s_2], dtype=int)
        )
        f["/galaxy_data"].create_dataset(
            "slist_start", data=np.array([0, n_s_1], dtype=int)
        )
        f["/galaxy_data"].create_dataset(
            "vel",
            data=np.array(
                [[vcentre_1, vcentre_1, vcentre_1], [vcentre_2, vcentre_2, vcentre_2]],
                dtype=float,
            ),
        )
        f["/galaxy_data/vel"].attrs["unit"] = "km/s"
        f.create_group("global_lists")
        f["/global_lists"].create_dataset(
            "galaxy_bhlist",
            data=np.r_[np.zeros(n_bh_1, dtype=int), np.ones(n_bh_2, dtype=int)],
        )
        f["/global_lists"].create_dataset(
            "galaxy_glist",
            data=np.r_[
                -np.ones(n_g_b // 2, dtype=int),
                np.zeros(n_g_1, dtype=int),
                -np.ones(n_g_b // 2, dtype=int),
                np.ones(n_g_2, dtype=int),
            ],
        )
        f["/global_lists"].create_dataset(
            "galaxy_slist",
            data=np.r_[np.zeros(n_s_1, dtype=int), np.ones(n_s_2, dtype=int)],
        )
        f["/global_lists"].create_dataset(
            "halo_bhlist",
            data=np.r_[np.zeros(n_bh_1, dtype=int), np.ones(n_bh_1, dtype=int)],
        )
        f["/global_lists"].create_dataset(
            "halo_dmlist",
            data=np.r_[
                -np.ones(n_dm_b // 2, dtype=int),
                np.zeros(n_dm_1, dtype=int),
                -np.ones(n_dm_b // 2, dtype=int),
                np.ones(n_dm_2, dtype=int),
            ],
        )
        f["/global_lists"].create_dataset(
            "halo_glist",
            data=np.r_[
                -np.ones(n_g_b // 2, dtype=int),
                np.zeros(n_g_1, dtype=int),
                -np.ones(n_g_b // 2, dtype=int),
                np.ones(n_g_2, dtype=int),
            ],
        )
        f["/global_lists"].create_dataset(
            "halo_slist",
            data=np.r_[np.zeros(n_s_1, dtype=int), np.ones(n_s_2, dtype=int)],
        )
        f.create_group("halo_data")
        f["/halo_data"].create_dataset("GroupID", data=np.array([0, 1], dtype=int))
        f["/halo_data"].create_dataset(
            "bhlist_end", data=np.array([n_bh_1, n_bh_1 + n_bh_2], dtype=int)
        )
        f["/halo_data"].create_dataset(
            "bhlist_start", data=np.array([0, n_bh_1], dtype=int)
        )
        f["/halo_data"].create_dataset(
            "central_galaxy", data=np.array([0, 1], dtype=int)
        )
        f["/halo_data"].create_dataset(
            "child", data=np.array([False, False], dtype=bool)
        )
        f["/halo_data"].create_group("dicts")
        f["/halo_data/dicts"].create_dataset(
            "masses.total",
            data=np.array(
                [
                    (
                        n_g_1 * m_g + n_dm_1 * m_dm + n_s_1 * m_s + n_bh_1 * m_bh
                    ).to_value(u.msun),
                    (
                        n_g_2 * m_g + n_dm_2 * m_dm + n_s_2 * m_s + n_bh_2 * m_bh
                    ).to_value(u.msun),
                ],
                dtype=float,
            ),
        )
        f["/halo_data/dicts"].create_dataset(
            "radii.total_rmax", data=np.array([100, 100], dtype=float)
        )
        f["/halo_data/dicts/radii.total_rmax"].attrs["unit"] = "kpccm"

        f["/halo_data/dicts"].create_dataset(
            "virial_quantities.m200c", data=np.array([1.0e12, 2.0e12], dtype=float)
        )
        f["/halo_data/dicts/virial_quantities.m200c"].attrs["unit"] = "Msun"
        f["/halo_data"].create_dataset(
            "dmlist_end", data=np.array([n_dm_1, n_dm_1 + n_dm_2], dtype=int)
        )
        f["/halo_data"].create_dataset(
            "dmlist_start", data=np.array([0, n_dm_1], dtype=int)
        )
        f["/halo_data"].create_dataset(
            "galaxy_index_list_end", data=np.array([1, 2], dtype=int)
        )
        f["/halo_data"].create_dataset(
            "galaxy_index_list_start", data=np.array([0, 1], dtype=int)
        )
        f["/halo_data"].create_dataset(
            "glist_end", data=np.array([n_g_1, n_g_1 + n_g_2], dtype=int)
        )
        f["/halo_data"].create_dataset(
            "glist_start", data=np.array([0, n_g_1], dtype=int)
        )
        f["/halo_data"].create_group("lists")
        f["/halo_data/lists"].create_dataset("bhlist", data=np.array([0, 1], dtype=int))
        f["/halo_data/lists"].create_dataset(
            "dmlist",
            data=np.r_[
                np.arange(n_dm_b // 2, n_dm_b // 2 + n_dm_1, dtype=int),
                np.arange(n_dm_b + n_dm_1, n_dm_all, dtype=int),
            ],
        )
        f["/halo_data/lists"].create_dataset(
            "galaxy_index_list", data=np.array([0, 1], dtype=int)
        )
        f["/halo_data/lists"].create_dataset(
            "glist",
            data=np.r_[
                np.arange(n_g_b // 2, n_g_b // 2 + n_g_1, dtype=int),
                np.arange(n_g_b + n_g_1, n_g_all, dtype=int),
            ],
        )
        f["/halo_data/lists"].create_dataset(
            "slist", data=np.arange(n_s_1 + n_s_2, dtype=int)
        )
        f["/halo_data"].create_dataset(
            "minpotpos",
            data=np.array(
                [
                    [
                        centre_1 * 1000 + 1.0,
                        centre_1 * 1000 + 1.0,
                        centre_1 * 1000 + 1.0,
                    ],
                    [
                        centre_2 * 1000 + 1.0,
                        centre_2 * 1000 + 1.0,
                        centre_2 * 1000 + 1.0,
                    ],
                ],
                dtype=float,
            ),
        )
        f["/halo_data/minpotpos"].attrs["unit"] = "kpccm"
        f["/halo_data"].create_dataset(
            "minpotvel",
            data=np.array(
                [
                    [vcentre_1 + 1.0, vcentre_1 + 1.0, vcentre_1 + 1.0],
                    [vcentre_2 + 1.0, vcentre_2 + 1.0, vcentre_2 + 1.0],
                ],
                dtype=float,
            ),
        )
        f["/halo_data/minpotvel"].attrs["unit"] = "km/s"
        f["/halo_data"].create_dataset(
            "nbh", data=np.array([n_bh_1, n_bh_2], dtype=int)
        )
        f["/halo_data"].create_dataset("ndm", data=np.array([0, 0], dtype=int))
        f["/halo_data"].create_dataset("ndm2", data=np.array([0, 0], dtype=int))
        f["/halo_data"].create_dataset("ndm3", data=np.array([0, 0], dtype=int))
        f["/halo_data"].create_dataset("ndust", data=np.array([0, 0], dtype=int))
        f["/halo_data"].create_dataset("ngas", data=np.array([n_g_1, n_g_2], dtype=int))
        f["/halo_data"].create_dataset(
            "nstar", data=np.array([n_s_2, n_s_2], dtype=int)
        )
        f["/halo_data"].create_dataset(
            "pos",
            data=np.array(
                [
                    [centre_1 * 1000, centre_1 * 1000, centre_1 * 1000],
                    [centre_2 * 1000, centre_2 * 1000, centre_2 * 1000],
                ],
                dtype=float,
            ),
        )
        f["/halo_data/pos"].attrs["unit"] = "kpccm"
        f["/halo_data"].create_dataset(
            "slist_end", data=np.array([n_s_1, n_s_2], dtype=int)
        )
        f["/halo_data"].create_dataset(
            "slist_start", data=np.array([0, n_s_1], dtype=int)
        )
        f["/halo_data"].create_dataset(
            "vel",
            data=np.array(
                [[vcentre_1, vcentre_1, vcentre_1], [vcentre_2, vcentre_2, vcentre_2]],
                dtype=float,
            ),
        )
        f["/halo_data/vel"].attrs["unit"] = "km/s"
        f.create_group("simulation_attributes")
        f["/simulation_attributes"].attrs["Densities"] = [
            200 * rho_c.to_value(u.msun / u.kpc**3),
            500 * rho_c.to_value(u.msun / u.kpc**3),
            2500 * rho_c.to_value(u.msun / u.kpc**3),
        ]
        # f["/simulation_attributes"].attrs["E_z"] = ...
        f["/simulation_attributes"].attrs["G"] = 4.51691362044e-39
        f["/simulation_attributes"].attrs["H_z"] = (
            h0 * 100 * u.km / u.s / u.Mpc
        ).to_value(u.s**-1)
        # f["/simulation_attributes"].attrs["Om_z"] = ...
        f["/simulation_attributes"].attrs["XH"] = 0.76
        f["/simulation_attributes"].attrs["baryons_present"] = True
        f["/simulation_attributes"].attrs["basename"] = "toysnap.hdf5"
        f["/simulation_attributes"].attrs["boxsize"] = boxsize.to_value(u.kpc)
        f["/simulation_attributes"].attrs["boxsize_units"] = "kpccm"
        f["/simulation_attributes"].attrs["cosmological_simulation"] = True
        f["/simulation_attributes"].attrs["critical_density"] = rho_c.to_value(
            u.msun / u.kpc**3
        )
        f["/simulation_attributes"].attrs["ds_type"] = "SwiftDataset"
        # f["/simulation_attributes"].attrs["effective_resolution"] = ...
        # f["/simulation_attributes"].attrs["fullpath"] = ...
        f["/simulation_attributes"].attrs["hubble_constant"] = 0.7
        f["/simulation_attributes"].attrs["mean_interparticle_separation"] = (
            boxsize.to_value(u.kpc) / n_dm_all ** (1 / 3)
        )
        f["/simulation_attributes"].attrs["nbh"] = n_bh_1 + n_bh_2
        f["/simulation_attributes"].attrs["ndm"] = n_dm_all
        f["/simulation_attributes"].attrs["ndust"] = 0
        f["/simulation_attributes"].attrs["ngas"] = n_g_all
        f["/simulation_attributes"].attrs["nstar"] = n_s_1 + n_s_2
        f["/simulation_attributes"].attrs["ntot"] = (
            n_g_all + n_dm_all + n_s_1 + n_s_2 + n_bh_1 + n_bh_2
        )
        f["/simulation_attributes"].attrs["omega_baryon"] = 0.05
        f["/simulation_attributes"].attrs["omega_lambda"] = 0.7
        f["/simulation_attributes"].attrs["omega_matter"] = 0.3
        f["/simulation_attributes"].attrs["redshift"] = 0.0
        f["/simulation_attributes"].attrs["scale_factor"] = 1.0
        f["/simulation_attributes"].attrs["search_radius"] = [300.0, 1000.0, 3000.0]
        f["/simulation_attributes"].attrs["time"] = age.to_value(u.s)
        f["/simulation_attributes"].attrs["unbind_galaxies"] = False
        f["/simulation_attributes"].attrs["unbind_halos"] = False
        f["/simulation_attributes"].create_group("parameters")  # attrs not needed
        f["/simulation_attributes"].create_group("units")
        f["/simulation_attributes/units"].attrs["Densities"] = "Msun/kpc**3"
        f["/simulation_attributes/units"].attrs["G"] = "kpc**3/(Msun*s**2)"
        f["/simulation_attributes/units"].attrs["H_z"] = "1/s"
        f["/simulation_attributes/units"].attrs["boxsize"] = "kpccm"
        f["/simulation_attributes/units"].attrs["critical_density"] = "Msun/kpc**3"
        f["/simulation_attributes/units"].attrs[
            "mean_interparticle_separation"
        ] = "kpccm"
        f["/simulation_attributes/units"].attrs["search_radius"] = "kpccm"
        f["/simulation_attributes/units"].attrs["time"] = "s"
    return


def remove_toycaesar(filename: str = toycaesar_filename) -> None:
    """
    Removes files created by :func:`~swiftgalaxy.demo_data.create_toycaesar`.

    Parameters
    ----------
    filename : :obj:`str`
        The file name for the catalogue file to be removed.
        (Default: ``"toycaesar.hdf5"``)
    """
    os.remove(filename)
    return


def create_toysoap(
    filename: str = toysoap_filename,
    membership_filebase: str = toysoap_membership_filebase,
    create_membership: bool = True,
    create_virtual_snapshot: bool = False,
    create_virtual_snapshot_from: str = toysnap_filename,
    virtual_snapshot_filename: str = toysoap_virtual_snapshot_filename,
) -> None:
    """
    Creates a sample SOAP catalogue containing 2 galaxies matching the snapshot
    file created by :func:`~swiftgalaxy.demo_data.create_toysnap`. Files containing
    particle membership information and a virtual snapshot linking this information
    into the snapshot file can optionally be created.

    The data are created entirely "by hand". They are not the result of any actual
    simulation. Their purpose is to illustrate :mod:`swiftgalaxy` use by providing files
    with formats identical to actual SOAP catalogue files without the need for
    additional downloads.

    Parameters
    ----------
    filename : :obj:`str`
        The file name for the catalogue file to be created.
        (Default: ``"toysoap.hdf5"``)

    membership_filebase : :obj:`str`
        The base name for membership files, completed as ``base.N.hdf5`` where ``N`` is
        an integer. Ignored if ``create_membership`` is ``False``.
        (Default: ``"toysoap_membership"``)

    create_membership : :obj:`bool`
        If ``True``, create membership files. (Default: ``True``)

    create_virtual_snapshot : :obj:`bool`
        If ``True``, create virtual snapshot with real snapshot and membership information
        as links to other hdf5 files. (Default: ``False``)

    create_virtual_snapshot_from : :obj:`str`
        Snapshot file to use as the basis for the virtual snapshot file. Ignored if
        ``create_virtual_snapshot`` is ``False``. (Default: ``"toysnap.hdf5"``)

    virtual_snapshot_filename : :obj:`str`
        Filename for virtual snapshot file. Ignored if ``create_virtual_snapshot`` is
        ``False``. (Default: ``"toysnap_virtual.hdf5"``)
    """

    with h5py.File(filename, "w") as f:
        f.create_group("Cells")
        f.create_group("Code")
        f["Code"].attrs["Code"] = "SOAP"
        f["Code"].attrs["git_hash"] = "undefined"
        f.create_group("Cosmology")
        f["Cosmology"].attrs["Cosmological run"] = np.array([1])
        f["Cosmology"].attrs["Critical density [internal units]"] = np.array(
            [12.87106552]
        )
        f["Cosmology"].attrs["H [internal units]"] = np.array([68.09999997])
        f["Cosmology"].attrs["H0 [internal units]"] = np.array([68.09999997])
        f["Cosmology"].attrs["Hubble time [internal units]"] = np.array([0.01468429])
        f["Cosmology"].attrs["Lookback time [internal units]"] = np.array(
            [9.02056208e-16]
        )
        f["Cosmology"].attrs["M_nu_eV"] = np.array([0.06])
        f["Cosmology"].attrs["N_eff"] = np.array([3.04400163])
        f["Cosmology"].attrs["N_nu"] = np.array([1.0])
        f["Cosmology"].attrs["N_ur"] = np.array([2.0308])
        f["Cosmology"].attrs["Omega_b"] = np.array([0.0486])
        f["Cosmology"].attrs["Omega_cdm"] = np.array([0.256011])
        f["Cosmology"].attrs["Omega_g"] = np.array([5.33243487e-05])
        f["Cosmology"].attrs["Omega_k"] = np.array([2.5212783e-09])
        f["Cosmology"].attrs["Omega_lambda"] = np.array([0.693922])
        f["Cosmology"].attrs["Omega_m"] = np.array([0.304611])
        f["Cosmology"].attrs["Omega_nu"] = np.array([0.00138908])
        f["Cosmology"].attrs["Omega_nu_0"] = np.array([0.00138908])
        f["Cosmology"].attrs["Omega_r"] = np.array([7.79180471e-05])
        f["Cosmology"].attrs["Omega_ur"] = np.array([2.45936984e-05])
        f["Cosmology"].attrs["Redshift"] = np.array([0.0])
        f["Cosmology"].attrs["Scale-factor"] = np.array([1.0])
        f["Cosmology"].attrs["T_CMB_0 [K]"] = np.array([2.7255])
        f["Cosmology"].attrs["T_CMB_0 [internal units]"] = np.array([2.7255])
        f["Cosmology"].attrs["T_nu_0 [eV]"] = np.array([0.00016819])
        f["Cosmology"].attrs["T_nu_0 [internal units]"] = np.array([1.9517578])
        f["Cosmology"].attrs["Universe age [internal units]"] = np.array([0.01407376])
        f["Cosmology"].attrs["a_beg"] = np.array([0.03125])
        f["Cosmology"].attrs["a_end"] = np.array([1.0])
        f["Cosmology"].attrs["deg_nu"] = np.array([1.0])
        f["Cosmology"].attrs["deg_nu_tot"] = np.array([1.0])
        f["Cosmology"].attrs["h"] = np.array([0.681])
        f["Cosmology"].attrs["time_beg [internal units]"] = np.array([9.66296122e-05])
        f["Cosmology"].attrs["time_end [internal units]"] = np.array([0.01407376])
        f["Cosmology"].attrs["w"] = np.array([-1.0])
        f["Cosmology"].attrs["w_0"] = np.array([-1.0])
        f["Cosmology"].attrs["w_a"] = np.array([0.0])
        f.create_group("Header")
        f["Header"].attrs["BoxSize"] = np.array(
            [boxsize.to_value(u.Mpc), boxsize.to_value(u.Mpc), boxsize.to_value(u.Mpc)]
        )
        f["Header"].attrs["Code"] = "SOAP"
        f["Header"].attrs["Dimension"] = np.array([3])
        f["Header"].attrs["NumFilesPerSnapshot"] = np.array([1])
        f["Header"].attrs["NumPartTypes"] = np.array([6])
        f["Header"].attrs["NumPart_ThisFile"] = np.array([0, 0, 0, 0, 0, 0])
        f["Header"].attrs["NumPart_Total"] = np.array([0, 0, 0, 0, 0, 0])
        f["Header"].attrs["NumPart_Total_Highword"] = np.array([0, 0, 0, 0, 0, 0])
        f["Header"].attrs["NumSubhalos_ThisFile"] = np.array([2])
        f["Header"].attrs["NumSubhalos_Total"] = np.array([2])
        f["Header"].attrs["OutputType"] = "SOAP"
        f["Header"].attrs["Redshift"] = np.array([0.0])
        f["Header"].attrs["RunName"] = "swiftgalaxy-test"
        f["Header"].attrs["Scale-factor"] = np.array([1.0])
        f["Header"].attrs["SnapshotDate"] = "00:00:00 1900-01-01 GMT"
        f["Header"].attrs["SubhaloTypes"] = [
            "BoundSubhalo",
            "ExclusiveSphere/1000kpc",
            "ExclusiveSphere/100kpc",
            "ExclusiveSphere/10kpc",
            "ExclusiveSphere/3000kpc",
            "ExclusiveSphere/300kpc",
            "ExclusiveSphere/30kpc",
            "ExclusiveSphere/500kpc",
            "ExclusiveSphere/50kpc",
            "InclusiveSphere/1000kpc",
            "InclusiveSphere/100kpc",
            "InclusiveSphere/10kpc",
            "InclusiveSphere/3000kpc",
            "InclusiveSphere/300kpc",
            "InclusiveSphere/30kpc",
            "InclusiveSphere/500kpc",
            "InclusiveSphere/50kpc",
            "InputHalos",
            "InputHalos/FOF",
            "InputHalos/HBTplus",
            "ProjectedAperture/100kpc/projx",
            "ProjectedAperture/100kpc/projy",
            "ProjectedAperture/100kpc/projz",
            "ProjectedAperture/10kpc/projx",
            "ProjectedAperture/10kpc/projy",
            "ProjectedAperture/10kpc/projz",
            "ProjectedAperture/30kpc/projx",
            "ProjectedAperture/30kpc/projy",
            "ProjectedAperture/30kpc/projz",
            "ProjectedAperture/50kpc/projx",
            "ProjectedAperture/50kpc/projy",
            "ProjectedAperture/50kpc/projz",
            "SO/1000_crit",
            "SO/100_crit",
            "SO/200_crit",
            "SO/200_mean",
            "SO/2500_crit",
            "SO/500_crit",
            "SO/50_crit",
            "SO/5xR_500_crit",
            "SO/BN98",
            "SOAP",
        ]
        f["Header"].attrs["System"] = "swiftgalaxy-test"
        f["Header"].attrs["ThisFile"] = np.array([0])
        f.create_group("Parameters")
        f["Parameters"].attrs["calculations"] = [
            "SO_1000_crit",
            "SO_100_crit",
            "SO_200_crit",
            "SO_200_mean",
            "SO_2500_crit",
            "SO_500_crit",
            "SO_50_crit",
            "SO_5xR_500_crit",
            "SO_BN98",
            "bound_subhalo_properties",
            "exclusive_sphere_1000kpc",
            "exclusive_sphere_100kpc",
            "exclusive_sphere_10kpc",
            "exclusive_sphere_3000kpc",
            "exclusive_sphere_300kpc",
            "exclusive_sphere_30kpc",
            "exclusive_sphere_500kpc",
            "exclusive_sphere_50kpc",
            "inclusive_sphere_1000kpc",
            "inclusive_sphere_100kpc",
            "inclusive_sphere_10kpc",
            "inclusive_sphere_3000kpc",
            "inclusive_sphere_300kpc",
            "inclusive_sphere_30kpc",
            "inclusive_sphere_500kpc",
            "inclusive_sphere_50kpc",
            "projected_aperture_100kpc",
            "projected_aperture_10kpc",
            "projected_aperture_30kpc",
            "projected_aperture_50kpc",
        ]
        f["Parameters"].attrs["centrals_only"] = 0
        f["Parameters"].attrs["halo_basename"] = "undefined"
        f["Parameters"].attrs["halo_format"] = "HBTplus"
        f["Parameters"].attrs["halo_indices"] = np.array([])
        f["Parameters"].attrs["snapshot_nr"] = 0
        f["Parameters"].attrs["swift_filename"] = "toysnap.hdf5"
        f.create_group("PhysicalConstants")
        f["PhysicalConstants"].create_group("CGS")
        f["PhysicalConstants/CGS"].attrs["T_CMB_0"] = np.array([2.7255])
        f["PhysicalConstants/CGS"].attrs["astronomical_unit"] = np.array(
            [1.49597871e13]
        )
        f["PhysicalConstants/CGS"].attrs["avogadro_number"] = np.array([6.02214076e23])
        f["PhysicalConstants/CGS"].attrs["boltzmann_k"] = np.array([1.380649e-16])
        f["PhysicalConstants/CGS"].attrs["caseb_recomb"] = np.array([2.6e-13])
        f["PhysicalConstants/CGS"].attrs["earth_mass"] = np.array([5.97217e27])
        f["PhysicalConstants/CGS"].attrs["electron_charge"] = np.array([1.60217663e-19])
        f["PhysicalConstants/CGS"].attrs["electron_mass"] = np.array([9.1093837e-28])
        f["PhysicalConstants/CGS"].attrs["electron_volt"] = np.array([1.60217663e-12])
        f["PhysicalConstants/CGS"].attrs["light_year"] = np.array([9.46063e17])
        f["PhysicalConstants/CGS"].attrs["newton_G"] = np.array([6.6743e-08])
        f["PhysicalConstants/CGS"].attrs["parsec"] = np.array([3.08567758e18])
        f["PhysicalConstants/CGS"].attrs["planck_h"] = np.array([6.62607015e-27])
        f["PhysicalConstants/CGS"].attrs["planck_hbar"] = np.array([1.05457182e-27])
        f["PhysicalConstants/CGS"].attrs["primordial_He_fraction"] = np.array([0.248])
        f["PhysicalConstants/CGS"].attrs["proton_mass"] = np.array([1.67262192e-24])
        f["PhysicalConstants/CGS"].attrs["reduced_hubble"] = np.array([3.24077929e-18])
        f["PhysicalConstants/CGS"].attrs["solar_mass"] = np.array([1.98841e33])
        f["PhysicalConstants/CGS"].attrs["speed_light_c"] = np.array([2.99792458e10])
        f["PhysicalConstants/CGS"].attrs["stefan_boltzmann"] = np.array(
            [5.67037442e-05]
        )
        f["PhysicalConstants/CGS"].attrs["thomson_cross_section"] = np.array(
            [6.65245873e-25]
        )
        f["PhysicalConstants/CGS"].attrs["year"] = np.array([31556925.1])
        f.create_group("SWIFT")
        f.create_group("Units")
        f["Units"].attrs["Unit current in cgs (U_I)"] = np.array([1.0])
        f["Units"].attrs["Unit length in cgs (U_L)"] = np.array([3.08567758e24])
        f["Units"].attrs["Unit mass in cgs (U_M)"] = np.array([1.98841586e43])
        f["Units"].attrs["Unit temperature in cgs (U_T)"] = np.array([1.0])
        f["Units"].attrs["Unit time in cgs (U_t)"] = np.array([3.08567758e19])
        f.create_group("BoundSubhalo")
        f["BoundSubhalo"].attrs["Masked"] = False
        f.create_group("ExclusiveSphere")
        f.create_group("ExclusiveSphere/1000kpc")
        f["ExclusiveSphere/1000kpc"].attrs["Mask Dataset Combination"] = "sum"
        f["ExclusiveSphere/1000kpc"].attrs["Mask Datasets"] = [
            "BoundSubhalo/NumberOfGasParticles",
            "BoundSubhalo/NumberOfDarkMatterParticles",
            "BoundSubhalo/NumberOfStarParticles",
            "BoundSubhalo/NumberOfBlackHoleParticles",
        ]
        f["ExclusiveSphere/1000kpc"].attrs["Mask Threshold"] = 100
        f["ExclusiveSphere/1000kpc"].attrs["Masked"] = True
        f.create_group("ExclusiveSphere/100kpc")
        f["ExclusiveSphere/100kpc"].attrs["Masked"] = False
        f.create_group("ExclusiveSphere/10kpc")
        f["ExclusiveSphere/10kpc"].attrs["Masked"] = False
        f.create_group("ExclusiveSphere/3000kpc")
        f["ExclusiveSphere/3000kpc"].attrs["Mask Dataset Combination"] = "sum"
        f["ExclusiveSphere/3000kpc"].attrs["Mask Datasets"] = [
            "BoundSubhalo/NumberOfGasParticles",
            "BoundSubhalo/NumberOfDarkMatterParticles",
            "BoundSubhalo/NumberOfStarParticles",
            "BoundSubhalo/NumberOfBlackHoleParticles",
        ]
        f["ExclusiveSphere/3000kpc"].attrs["Mask Threshold"] = 100
        f["ExclusiveSphere/3000kpc"].attrs["Masked"] = True
        f.create_group("ExclusiveSphere/300kpc")
        f["ExclusiveSphere/300kpc"].attrs["Masked"] = False
        f.create_group("ExclusiveSphere/30kpc")
        f["ExclusiveSphere/30kpc"].attrs["Masked"] = False
        f.create_group("ExclusiveSphere/500kpc")
        f["ExclusiveSphere/500kpc"].attrs["Mask Dataset Combination"] = "sum"
        f["ExclusiveSphere/500kpc"].attrs["Mask Datasets"] = [
            "BoundSubhalo/NumberOfGasParticles",
            "BoundSubhalo/NumberOfDarkMatterParticles",
            "BoundSubhalo/NumberOfStarParticles",
            "BoundSubhalo/NumberOfBlackHoleParticles",
        ]
        f["ExclusiveSphere/500kpc"].attrs["Mask Threshold"] = 100
        f["ExclusiveSphere/500kpc"].attrs["Masked"] = True
        f.create_group("ExclusiveSphere/50kpc")
        f["ExclusiveSphere/50kpc"].attrs["Masked"] = False
        f.create_group("InclusiveSphere")
        f.create_group("InclusiveSphere/1000kpc")
        f["InclusiveSphere/1000kpc"].attrs["Mask Dataset Combination"] = "sum"
        f["InclusiveSphere/1000kpc"].attrs["Mask Datasets"] = [
            "BoundSubhalo/NumberOfGasParticles",
            "BoundSubhalo/NumberOfDarkMatterParticles",
            "BoundSubhalo/NumberOfStarParticles",
            "BoundSubhalo/NumberOfBlackHoleParticles",
        ]
        f["InclusiveSphere/1000kpc"].attrs["Mask Threshold"] = 100
        f["InclusiveSphere/1000kpc"].attrs["Masked"] = True
        f.create_group("InclusiveSphere/100kpc")
        f["InclusiveSphere/100kpc"].attrs["Masked"] = False
        f.create_group("InclusiveSphere/10kpc")
        f["InclusiveSphere/10kpc"].attrs["Masked"] = False
        f.create_group("InclusiveSphere/3000kpc")
        f["InclusiveSphere/3000kpc"].attrs["Mask Dataset Combination"] = "sum"
        f["InclusiveSphere/3000kpc"].attrs["Mask Datasets"] = [
            "BoundSubhalo/NumberOfGasParticles",
            "BoundSubhalo/NumberOfDarkMatterParticles",
            "BoundSubhalo/NumberOfStarParticles",
            "BoundSubhalo/NumberOfBlackHoleParticles",
        ]
        f["InclusiveSphere/3000kpc"].attrs["Mask Threshold"] = 100
        f["InclusiveSphere/3000kpc"].attrs["Masked"] = True
        f.create_group("InclusiveSphere/300kpc")
        f["InclusiveSphere/300kpc"].attrs["Masked"] = False
        f.create_group("InclusiveSphere/30kpc")
        f["InclusiveSphere/30kpc"].attrs["Masked"] = False
        f.create_group("InclusiveSphere/500kpc")
        f["InclusiveSphere/500kpc"].attrs["Mask Dataset Combination"] = "sum"
        f["InclusiveSphere/500kpc"].attrs["Mask Datasets"] = [
            "BoundSubhalo/NumberOfGasParticles",
            "BoundSubhalo/NumberOfDarkMatterParticles",
            "BoundSubhalo/NumberOfStarParticles",
            "BoundSubhalo/NumberOfBlackHoleParticles",
        ]
        f["InclusiveSphere/500kpc"].attrs["Mask Threshold"] = 100
        f["InclusiveSphere/500kpc"].attrs["Masked"] = True
        f.create_group("InclusiveSphere/50kpc")
        f["InclusiveSphere/50kpc"].attrs["Masked"] = False
        f.create_group("InputHalos")
        f.create_group("InputHalos/FOF")
        f.create_group("InputHalos/HBTplus")
        f.create_group("ProjectedAperture")
        f.create_group("ProjectedAperture/100kpc")
        f.create_group("ProjectedAperture/10kpc")
        f.create_group("ProjectedAperture/30kpc")
        f.create_group("ProjectedAperture/50kpc")
        f.create_group("ProjectedAperture/100kpc/projx")
        f.create_group("ProjectedAperture/100kpc/projy")
        f.create_group("ProjectedAperture/100kpc/projz")
        f.create_group("ProjectedAperture/10kpc/projx")
        f.create_group("ProjectedAperture/10kpc/projy")
        f.create_group("ProjectedAperture/10kpc/projz")
        f.create_group("ProjectedAperture/30kpc/projx")
        f.create_group("ProjectedAperture/30kpc/projy")
        f.create_group("ProjectedAperture/30kpc/projz")
        f.create_group("ProjectedAperture/50kpc/projx")
        f.create_group("ProjectedAperture/50kpc/projy")
        f.create_group("ProjectedAperture/50kpc/projz")
        f.create_group("SO")
        f.create_group("SO/1000_crit")
        f["SO/1000_crit"].attrs["Mask Dataset Combination"] = "sum"
        f["SO/1000_crit"].attrs["Mask Datasets"] = [
            "BoundSubhalo/NumberOfGasParticles",
            "BoundSubhalo/NumberOfDarkMatterParticles",
            "BoundSubhalo/NumberOfStarParticles",
            "BoundSubhalo/NumberOfBlackHoleParticles",
        ]
        f["SO/1000_crit"].attrs["Mask Threshold"] = 100
        f["SO/1000_crit"].attrs["Masked"] = True
        f.create_group("SO/100_crit")
        f["SO/100_crit"].attrs["Mask Dataset Combination"] = "sum"
        f["SO/100_crit"].attrs["Mask Datasets"] = [
            "BoundSubhalo/NumberOfGasParticles",
            "BoundSubhalo/NumberOfDarkMatterParticles",
            "BoundSubhalo/NumberOfStarParticles",
            "BoundSubhalo/NumberOfBlackHoleParticles",
        ]
        f["SO/100_crit"].attrs["Mask Threshold"] = 100
        f["SO/100_crit"].attrs["Masked"] = True
        f.create_group("SO/200_crit")
        f["SO/200_crit"].attrs["Masked"] = False
        f.create_group("SO/200_mean")
        f["SO/200_mean"].attrs["Masked"] = False
        f.create_group("SO/2500_crit")
        f["SO/2500_crit"].attrs["Mask Dataset Combination"] = "sum"
        f["SO/2500_crit"].attrs["Mask Datasets"] = [
            "BoundSubhalo/NumberOfGasParticles",
            "BoundSubhalo/NumberOfDarkMatterParticles",
            "BoundSubhalo/NumberOfStarParticles",
            "BoundSubhalo/NumberOfBlackHoleParticles",
        ]
        f["SO/2500_crit"].attrs["Mask Threshold"] = 100
        f["SO/2500_crit"].attrs["Masked"] = True
        f.create_group("SO/500_crit")
        f["SO/500_crit"].attrs["Masked"] = False
        f.create_group("SO/50_crit")
        f["SO/50_crit"].attrs["Mask Dataset Combination"] = "sum"
        f["SO/50_crit"].attrs["Mask Datasets"] = [
            "BoundSubhalo/NumberOfGasParticles",
            "BoundSubhalo/NumberOfDarkMatterParticles",
            "BoundSubhalo/NumberOfStarParticles",
            "BoundSubhalo/NumberOfBlackHoleParticles",
        ]
        f["SO/50_crit"].attrs["Mask Threshold"] = 100
        f["SO/50_crit"].attrs["Masked"] = True
        f.create_group("SO/5xR_500_crit")
        f["SO/5xR_500_crit"].attrs["Mask Dataset Combination"] = "sum"
        f["SO/5xR_500_crit"].attrs["Mask Datasets"] = [
            "BoundSubhalo/NumberOfGasParticles",
            "BoundSubhalo/NumberOfDarkMatterParticles",
            "BoundSubhalo/NumberOfStarParticles",
            "BoundSubhalo/NumberOfBlackHoleParticles",
        ]
        f["SO/5xR_500_crit"].attrs["Mask Threshold"] = 100
        f["SO/5xR_500_crit"].attrs["Masked"] = True
        f.create_group("SO/BN98")
        f["SO/BN98"].attrs["Mask Dataset Combination"] = "sum"
        f["SO/BN98"].attrs["Mask Datasets"] = [
            "BoundSubhalo/NumberOfGasParticles",
            "BoundSubhalo/NumberOfDarkMatterParticles",
            "BoundSubhalo/NumberOfStarParticles",
            "BoundSubhalo/NumberOfBlackHoleParticles",
        ]
        f["SO/BN98"].attrs["Mask Threshold"] = 100
        f["SO/BN98"].attrs["Masked"] = True
        f.create_group("SOAP")
        soap_hhi = f["SOAP"].create_dataset(
            "HostHaloIndex", data=np.array([-1, -1]), dtype=int
        )
        soap_hhi.attrs[
            "Conversion factor to CGS " "(not including cosmological corrections)"
        ] = np.array([1.0])
        soap_hhi.attrs[
            "Conversion factor to physical CGS " "(including cosmological corrections)"
        ] = np.array([1.0])
        soap_hhi.attrs["Description"] = (
            "Index (within the SOAP arrays) of the top level "
            "parent of this subhalo. -1 for central subhalos."
        )
        soap_hhi.attrs["Is Compressed"] = np.True_
        soap_hhi.attrs["Lossy compression filter"] = "None"
        soap_hhi.attrs["Masked"] = np.False_
        soap_hhi.attrs["Property can be converted to comoving"] = np.array([0])
        soap_hhi.attrs["U_I exponent"] = np.array([0.0])
        soap_hhi.attrs["U_L exponent"] = np.array([0.0])
        soap_hhi.attrs["U_M exponent"] = np.array([0.0])
        soap_hhi.attrs["U_T exponent"] = np.array([0.0])
        soap_hhi.attrs["U_t exponent"] = np.array([0.0])
        soap_hhi.attrs["Value stored as physical"] = np.array([1])
        soap_hhi.attrs["a-scale exponent"] = np.array([0.0])
        soap_hhi.attrs["h-scale exponent"] = np.array([0.0])
        for i_so, so in enumerate(
            [
                "/BoundSubhalo",
                "/ExclusiveSphere/1000kpc",
                "/ExclusiveSphere/100kpc",
                "/ExclusiveSphere/10kpc",
                "/ExclusiveSphere/3000kpc",
                "/ExclusiveSphere/300kpc",
                "/ExclusiveSphere/30kpc",
                "/ExclusiveSphere/500kpc",
                "/ExclusiveSphere/50kpc",
                "/InclusiveSphere/1000kpc",
                "/InclusiveSphere/100kpc",
                "/InclusiveSphere/10kpc",
                "/InclusiveSphere/3000kpc",
                "/InclusiveSphere/300kpc",
                "/InclusiveSphere/30kpc",
                "/InclusiveSphere/500kpc",
                "/InclusiveSphere/50kpc",
                "/ProjectedAperture/100kpc/projx",
                "/ProjectedAperture/100kpc/projy",
                "/ProjectedAperture/100kpc/projz",
                "/ProjectedAperture/10kpc/projx",
                "/ProjectedAperture/10kpc/projy",
                "/ProjectedAperture/10kpc/projz",
                "/ProjectedAperture/30kpc/projx",
                "/ProjectedAperture/30kpc/projy",
                "/ProjectedAperture/30kpc/projz",
                "/ProjectedAperture/50kpc/projx",
                "/ProjectedAperture/50kpc/projy",
                "/ProjectedAperture/50kpc/projz",
                "/SO/1000_crit",
                "/SO/100_crit",
                "/SO/200_crit",
                "/SO/200_mean",
                "/SO/2500_crit",
                "/SO/500_crit",
                "/SO/50_crit",
                "/SO/5xR_500_crit",
                "/SO/BN98",
            ],
            1,
        ):
            com_ds = f[so].create_dataset(
                "CentreOfMass",
                data=np.array(
                    [[centre_1, centre_1, centre_1], [centre_2, centre_2, centre_2]]
                )
                + i_so * 0.001,
                dtype=float,
            )
            com_ds.attrs[
                "Conversion factor to CGS (not including cosmological corrections)"
            ] = np.array([3.08567758e24])
            com_ds.attrs[
                "Conversion factor to physical CGS (including cosmological corrections)"
            ] = np.array([3.08567758e24])
            com_ds.attrs["Description"] = "Centre of mass."
            com_ds.attrs["Is Compressed"] = np.True_
            com_ds.attrs["Lossy compression filter"] = "DScale5"
            com_ds.attrs["Masked"] = np.False_
            com_ds.attrs["Property can be converted to comoving"] = np.array([1])
            com_ds.attrs["U_I exponent"] = np.array([0.0])
            com_ds.attrs["U_L exponent"] = np.array([1.0])
            com_ds.attrs["U_M exponent"] = np.array([0.0])
            com_ds.attrs["U_T exponent"] = np.array([0.0])
            com_ds.attrs["U_t exponent"] = np.array([0.0])
            com_ds.attrs["Value stored as physical"] = np.array([0])
            com_ds.attrs["a-scale exponent"] = np.array([1])
            com_ds.attrs["h-scale exponent"] = np.array([0.0])
            comv_ds = f[so].create_dataset(
                "CentreOfMassVelocity",
                data=np.array(
                    [
                        [vcentre_1, vcentre_1, vcentre_1],
                        [vcentre_2, vcentre_2, vcentre_2],
                    ]
                )
                + i_so,
                dtype=float,
            )
            comv_ds.attrs[
                "Conversion factor to CGS (not including cosmological corrections)"
            ] = np.array([100000.0])
            comv_ds.attrs[
                "Conversion factor to physical CGS (including cosmological corrections)"
            ] = np.array([100000.0])
            comv_ds.attrs["Description"] = "Centre of mass velocity."
            comv_ds.attrs["Is Compressed"] = np.True_
            comv_ds.attrs["Lossy compression filter"] = "DScale1"
            comv_ds.attrs["Masked"] = np.False_
            comv_ds.attrs["Property can be converted to comoving"] = np.array([1])
            comv_ds.attrs["U_I exponent"] = np.array([0.0])
            comv_ds.attrs["U_L exponent"] = np.array([1.0])
            comv_ds.attrs["U_M exponent"] = np.array([0.0])
            comv_ds.attrs["U_T exponent"] = np.array([0.0])
            comv_ds.attrs["U_t exponent"] = np.array([-1.0])
            comv_ds.attrs["Value stored as physical"] = np.array([0])
            comv_ds.attrs["a-scale exponent"] = np.array([1])
            comv_ds.attrs["h-scale exponent"] = np.array([0.0])
        er = f["BoundSubhalo"].create_dataset(
            "EncloseRadius", data=np.array([0.1, 0.1]), dtype=float
        )
        er.attrs[
            "Conversion factor to CGS " "(not including cosmological corrections)"
        ] = np.array([3.08567758e24])
        er.attrs[
            "Conversion factor to physical CGS " "(including cosmological corrections)"
        ] = np.array([3.08567758e24])
        er.attrs["Description"] = "Radius of the particle furthest from the halo centre"
        er.attrs["Is Compressed"] = np.True_
        er.attrs["Lossy compression filter"] = "FMantissa9"
        er.attrs["Masked"] = np.False_
        er.attrs["Property can be converted to comoving"] = np.array([1])
        er.attrs["U_I exponent"] = np.array([0.0])
        er.attrs["U_L exponent"] = np.array([1.0])
        er.attrs["U_M exponent"] = np.array([0.0])
        er.attrs["U_T exponent"] = np.array([0.0])
        er.attrs["U_t exponent"] = np.array([0.0])
        er.attrs["Value stored as physical"] = np.array([0])
        er.attrs["a-scale exponent"] = np.array([1])
        er.attrs["h-scale exponent"] = np.array([0.0])
        f["Cells"].create_dataset(
            "Centres", data=np.array([[2.5, 5, 5], [7.5, 5, 5]], dtype=float)
        )
        f["Cells"].create_group("Counts")
        f["Cells/Counts"].create_dataset("Subhalos", data=np.array([1, 1]), dtype=int)
        f["Cells"].create_group("Files")
        f["Cells/Files"].create_dataset("Subhalos", data=np.array([0, 0], dtype=int))
        f["Cells"].create_group("Meta-data")
        f["Cells/Meta-data"].attrs["dimension"] = np.array([[2, 1, 1]], dtype=int)
        f["Cells/Meta-data"].attrs["nr_cells"] = np.array([2], dtype=int)
        f["Cells/Meta-data"].attrs["size"] = np.array(
            [
                0.5 * boxsize.to_value(u.Mpc),
                boxsize.to_value(u.Mpc),
                boxsize.to_value(u.Mpc),
            ],
            dtype=int,
        )
        f["Cells"].create_group("OffsetsInFile")
        f["Cells/OffsetsInFile"].create_dataset(
            "Subhalos", data=np.array([0, 1]), dtype=int
        )
        fof_c = f["InputHalos/FOF"].create_dataset(
            "Centres",
            data=np.array(
                [[centre_1, centre_1, centre_1], [centre_2, centre_2, centre_2]]
            ),
            dtype=float,
        )
        fof_c.attrs[
            "Conversion factor to CGS " "(not including cosmological corrections)"
        ] = np.array([3.08567758e24])
        fof_c.attrs[
            "Conversion factor to physical CGS " "(including cosmological corrections)"
        ] = np.array([3.08567758e24])
        fof_c.attrs["Description"] = (
            "Centre of mass of the host FOF halo of this subhalo. "
            "Zero for satellite and hostless subhalos."
        )
        fof_c.attrs["Is Compressed"] = np.True_
        fof_c.attrs["Lossy compression filter"] = "DScale5"
        fof_c.attrs["Masked"] = np.False_
        fof_c.attrs["Property can be converted to comoving"] = np.array([1])
        fof_c.attrs["U_I exponent"] = np.array([0.0])
        fof_c.attrs["U_L exponent"] = np.array([1.0])
        fof_c.attrs["U_M exponent"] = np.array([0.0])
        fof_c.attrs["U_T exponent"] = np.array([0.0])
        fof_c.attrs["U_t exponent"] = np.array([0.0])
        fof_c.attrs["Value stored as physical"] = np.array([0])
        fof_c.attrs["a-scale exponent"] = np.array([1])
        fof_c.attrs["h-scale exponent"] = np.array([0.0])
        fof_m = f["InputHalos/FOF"].create_dataset(
            "Masses",
            data=np.array(
                [
                    (
                        n_g_1 * m_g + n_dm_1 * m_dm + n_s_1 * m_s + n_bh_1 * m_bh
                    ).to_value(u.msun),
                    (
                        n_g_2 * m_g + n_dm_2 * m_dm + n_s_2 * m_s + n_bh_2 * m_bh
                    ).to_value(u.msun),
                ]
            ),
            dtype=float,
        )
        fof_m.attrs[
            "Conversion factor to CGS " "(not including cosmological corrections)"
        ] = np.array([1.98841e43])
        fof_m.attrs[
            "Conversion factor to physical CGS " "(including cosmological corrections)"
        ] = np.array([1.98841e43])
        fof_m.attrs["Description"] = (
            "Mass of the host FOF halo of this subhalo. "
            "Zero for satellite and hostless subhalos."
        )
        fof_m.attrs["Is Compressed"] = np.True_
        fof_m.attrs["Lossy compression filter"] = "FMantissa9"
        fof_m.attrs["Masked"] = np.False_
        fof_m.attrs["Property can be converted to comoving"] = np.array([1])
        fof_m.attrs["U_I exponent"] = np.array([0.0])
        fof_m.attrs["U_L exponent"] = np.array([0.0])
        fof_m.attrs["U_M exponent"] = np.array([1.0])
        fof_m.attrs["U_T exponent"] = np.array([0.0])
        fof_m.attrs["U_t exponent"] = np.array([0.0])
        fof_m.attrs["Value stored as physical"] = np.array([1])
        fof_m.attrs["a-scale exponent"] = np.array([0])
        fof_m.attrs["h-scale exponent"] = np.array([0.0])
        fof_s = f["InputHalos/FOF"].create_dataset(
            "Sizes",
            data=np.array(
                [n_g_1 + n_dm_1 + n_s_1 + n_bh_1, n_g_2 + n_dm_2 + n_s_2 + n_bh_2]
            ),
            dtype=int,
        )
        fof_s.attrs[
            "Conversion factor to CGS " "(not including cosmological corrections)"
        ] = np.array([1.0])
        fof_s.attrs[
            "Conversion factor to physical CGS " "(including cosmological corrections)"
        ] = np.array([1.0])
        fof_s.attrs["Description"] = (
            "Number of particles in the host FOF halo of this subhalo. "
            "Zero for satellite and hostless subhalos."
        )
        fof_s.attrs["Is Compressed"] = np.True_
        fof_s.attrs["Lossy compression filter"] = "None"
        fof_s.attrs["Masked"] = np.False_
        fof_s.attrs["Property can be converted to comoving"] = np.array([0])
        fof_s.attrs["U_I exponent"] = np.array([0.0])
        fof_s.attrs["U_L exponent"] = np.array([0.0])
        fof_s.attrs["U_M exponent"] = np.array([0.0])
        fof_s.attrs["U_T exponent"] = np.array([0.0])
        fof_s.attrs["U_t exponent"] = np.array([0.0])
        fof_s.attrs["Value stored as physical"] = np.array([1])
        fof_s.attrs["a-scale exponent"] = np.array([0.0])
        fof_s.attrs["h-scale exponent"] = np.array([0.0])
        hbt_hostfof = f["InputHalos/HBTplus"].create_dataset(
            "HostFOFId", data=np.array([1, 2]), dtype=int
        )
        hbt_hostfof.attrs[
            "Conversion factor to CGS " "(not including cosmological corrections)"
        ] = np.array([1.0])
        hbt_hostfof.attrs[
            "Conversion factor to physical CGS " "(including cosmological corrections)"
        ] = np.array([1.0])
        hbt_hostfof.attrs["Description"] = (
            "ID of the host FOF halo of this subhalo. "
            "Hostless halos have HostFOFId == -1"
        )
        hbt_hostfof.attrs["Is Compressed"] = np.True_
        hbt_hostfof.attrs["Lossy compression filter"] = "None"
        hbt_hostfof.attrs["Masked"] = np.False_
        hbt_hostfof.attrs["Property can be converted to comoving"] = np.array([0])
        hbt_hostfof.attrs["U_I exponent"] = np.array([0.0])
        hbt_hostfof.attrs["U_L exponent"] = np.array([0.0])
        hbt_hostfof.attrs["U_M exponent"] = np.array([0.0])
        hbt_hostfof.attrs["U_T exponent"] = np.array([0.0])
        hbt_hostfof.attrs["U_t exponent"] = np.array([0.0])
        hbt_hostfof.attrs["Value stored as physical"] = np.array([1])
        hbt_hostfof.attrs["a-scale exponent"] = np.array([0.0])
        hbt_hostfof.attrs["h-scale exponent"] = np.array([0.0])
        hci = f["InputHalos"].create_dataset(
            "HaloCatalogueIndex", data=np.array([1111, 2222]), dtype=int
        )
        hci.attrs[
            "Conversion factor to CGS " "(not including cosmological corrections)"
        ] = np.array([1.0])
        hci.attrs[
            "Conversion factor to physical CGS " "(including cosmological corrections)"
        ] = np.array([1.0])
        hci.attrs["Description"] = (
            "Index of this halo in the original halo finder catalogue "
            "(first halo has index=0)."
        )
        hci.attrs["Is Compressed"] = np.True_
        hci.attrs["Lossy compression filter"] = "None"
        hci.attrs["Masked"] = np.False_
        hci.attrs["Property can be converted to comoving"] = np.array([0])
        hci.attrs["U_I exponent"] = np.array([0.0])
        hci.attrs["U_L exponent"] = np.array([0.0])
        hci.attrs["U_M exponent"] = np.array([0.0])
        hci.attrs["U_T exponent"] = np.array([0.0])
        hci.attrs["U_t exponent"] = np.array([0.0])
        hci.attrs["Value stored as physical"] = np.array([1])
        hci.attrs["a-scale exponent"] = np.array([0.0])
        hci.attrs["h-scale exponent"] = np.array([0.0])
        hcentre = f["InputHalos"].create_dataset(
            "HaloCentre",
            data=np.array(
                [
                    [centre_1 + 0.001, centre_1 + 0.001, centre_1 + 0.001],
                    [centre_2 + 0.001, centre_2 + 0.001, centre_2 + 0.001],
                ]
            ),
            dtype=float,
        )
        hcentre.attrs[
            "Conversion factor to CGS " "(not including cosmological corrections)"
        ] = np.array([3.08567758e24])
        hcentre.attrs[
            "Conversion factor to physical CGS " "(including cosmological corrections)"
        ] = np.array([3.08567758e24])
        hcentre.attrs["Description"] = (
            "The centre of the subhalo as given by the halo finder. "
            "Used as reference for all relative positions. "
            "For VR and HBTplus this is equal to the position of the most bound particle "
            "in the subhalo."
        )
        hcentre.attrs["Is Compressed"] = np.True_
        hcentre.attrs["Lossy compression filter"] = "DScale5"
        hcentre.attrs["Masked"] = np.False_
        hcentre.attrs["Property can be converted to comoving"] = np.array([1])
        hcentre.attrs["U_I exponent"] = np.array([0.0])
        hcentre.attrs["U_L exponent"] = np.array([1.0])
        hcentre.attrs["U_M exponent"] = np.array([0.0])
        hcentre.attrs["U_T exponent"] = np.array([0.0])
        hcentre.attrs["U_t exponent"] = np.array([0.0])
        hcentre.attrs["Value stored as physical"] = np.array([0])
        hcentre.attrs["a-scale exponent"] = np.array([1])
        hcentre.attrs["h-scale exponent"] = np.array([0.0])
        iscent = f["InputHalos"].create_dataset(
            "IsCentral", data=np.array([1, 1]), dtype=int
        )
        iscent.attrs[
            "Conversion factor to CGS " "(not including cosmological corrections)"
        ] = np.array([1.0])
        iscent.attrs[
            "Conversion factor to physical CGS " "(including cosmological corrections)"
        ] = np.array([1.0])
        iscent.attrs["Description"] = (
            "Whether the halo finder flagged the halo as "
            "central (1) or satellite (0)."
        )
        iscent.attrs["Is Compressed"] = np.True_
        iscent.attrs["Lossy compression filter"] = "None"
        iscent.attrs["Masked"] = np.False_
        iscent.attrs["Property can be converted to comoving"] = np.array([0])
        iscent.attrs["U_I exponent"] = np.array([0.0])
        iscent.attrs["U_L exponent"] = np.array([0.0])
        iscent.attrs["U_M exponent"] = np.array([0.0])
        iscent.attrs["U_T exponent"] = np.array([0.0])
        iscent.attrs["U_t exponent"] = np.array([0.0])
        iscent.attrs["Value stored as physical"] = np.array([1])
        iscent.attrs["a-scale exponent"] = np.array([0.0])
        iscent.attrs["h-scale exponent"] = np.array([0.0])
        nbp = f["InputHalos"].create_dataset(
            "NumberOfBoundParticles",
            data=np.array(
                [n_g_1 + n_dm_1 + n_s_1 + n_bh_1, n_g_2 + n_dm_2 + n_s_2 + n_bh_2]
            ),
            dtype=int,
        )
        nbp.attrs[
            "Conversion factor to CGS " "(not including cosmological corrections)"
        ] = np.array([1.0])
        nbp.attrs[
            "Conversion factor to physical CGS " "(including cosmological corrections)"
        ] = np.array([1.0])
        nbp.attrs["Description"] = "Total number of particles bound to the subhalo."
        nbp.attrs["Is Compressed"] = np.True_
        nbp.attrs["Lossy compression filter"] = "None"
        nbp.attrs["Masked"] = np.False_
        nbp.attrs["Property can be converted to comoving"] = np.array([0])
        nbp.attrs["U_I exponent"] = np.array([0.0])
        nbp.attrs["U_L exponent"] = np.array([0.0])
        nbp.attrs["U_M exponent"] = np.array([0.0])
        nbp.attrs["U_T exponent"] = np.array([0.0])
        nbp.attrs["U_t exponent"] = np.array([0.0])
        nbp.attrs["Value stored as physical"] = np.array([1])
        nbp.attrs["a-scale exponent"] = np.array([0.0])
        nbp.attrs["h-scale exponent"] = np.array([0.0])

    if create_membership:
        with h5py.File(f"{membership_filebase}.0.hdf5", "w") as f:
            fof_ids = {
                0: np.concatenate(
                    (
                        np.ones(n_g_b // 2, dtype=int) * 2147483647,
                        np.ones(n_g_1, dtype=int),
                        np.ones(n_g_b // 2, dtype=int) * 2147483647,
                        np.ones(n_g_2, dtype=int) * 2,
                    )
                ),
                1: np.concatenate(
                    (
                        np.ones(n_dm_b // 2, dtype=int) * 2147483647,
                        np.ones(n_dm_1, dtype=int),
                        np.ones(n_dm_b // 2, dtype=int) * 2147483647,
                        np.ones(n_dm_2, dtype=int) * 2,
                    )
                ),
                4: np.concatenate(
                    (np.ones(n_s_1, dtype=int), np.ones(n_s_2, dtype=int) * 2)
                ),
                5: np.concatenate(
                    (np.ones(n_bh_1, dtype=int), np.ones(n_bh_2, dtype=int) * 2)
                ),
            }
            grnrs = {
                0: np.concatenate(
                    (
                        -np.ones(n_g_b // 2, dtype=int),
                        np.ones(n_g_1, dtype=int) * 1111,
                        -np.ones(n_g_b // 2, dtype=int),
                        np.ones(n_g_2, dtype=int) * 2222,
                    )
                ),
                1: np.concatenate(
                    (
                        -np.ones(n_dm_b // 2, dtype=int),
                        np.ones(n_dm_1, dtype=int) * 1111,
                        -np.ones(n_dm_b // 2, dtype=int),
                        np.ones(n_dm_2, dtype=int) * 2222,
                    )
                ),
                4: np.concatenate(
                    (np.ones(n_s_1, dtype=int) * 1111, np.ones(n_s_2, dtype=int) * 2222)
                ),
                5: np.concatenate(
                    (
                        np.ones(n_bh_1, dtype=int) * 1111,
                        np.ones(n_bh_2, dtype=int) * 2222,
                    )
                ),
            }
            ranks = {
                0: np.concatenate(
                    (
                        -np.ones(n_g_b // 2, dtype=int),
                        np.arange(n_g_1, dtype=int),
                        -np.ones(n_g_b // 2, dtype=int),
                        np.arange(n_g_2, dtype=int),
                    )
                ),
                1: np.concatenate(
                    (
                        -np.ones(n_dm_b // 2, dtype=int),
                        np.arange(n_dm_1, dtype=int),
                        -np.ones(n_dm_b // 2, dtype=int),
                        np.arange(n_dm_2, dtype=int),
                    )
                ),
                4: np.concatenate(
                    (np.arange(n_s_1, dtype=int), np.arange(n_s_2, dtype=int))
                ),
                5: np.concatenate(
                    (np.arange(n_bh_1, dtype=int), np.arange(n_bh_2, dtype=int))
                ),
            }
            for ptype in (0, 1, 4, 5):
                g = f.create_group(f"PartType{ptype}")
                ds_fof = g.create_dataset("FOFGroupIDs", data=fof_ids[ptype], dtype=int)
                ds_grnr = g.create_dataset(
                    "GroupNr_bound", data=grnrs[ptype], dtype=int
                )
                ds_rank = g.create_dataset("Rank_bound", data=ranks[ptype], dtype=int)
                ds_fof.attrs[
                    "Conversion factor to CGS " "(including cosmological corrections)"
                ] = np.array([1.0])
                ds_fof.attrs[
                    "Conversion factor to CGS "
                    "(not including cosmological corrections)"
                ] = np.array([1.0])
                ds_fof.attrs["Description"] = (
                    "Friends-Of-Friends ID of the group "
                    "in which this particle is a member, of -1 if none"
                )
                ds_fof.attrs["U_I exponent"] = np.array([0.0])
                ds_fof.attrs["U_L exponent"] = np.array([0.0])
                ds_fof.attrs["U_M exponent"] = np.array([0.0])
                ds_fof.attrs["U_T exponent"] = np.array([0.0])
                ds_fof.attrs["U_t exponent"] = np.array([0.0])
                ds_fof.attrs["a-scale exponent"] = np.array([0.0])
                ds_fof.attrs["h-scale exponent"] = np.array([0.0])
                ds_grnr.attrs[
                    "Conversion factor to CGS (including cosmological corrections)"
                ] = np.array([1.0])
                ds_grnr.attrs[
                    "Conversion factor to CGS (not including cosmological corrections)"
                ] = np.array([1.0])
                ds_grnr.attrs["Description"] = (
                    "Index of halo in which this particle "
                    "is a bound member, or -1 if none"
                )
                ds_grnr.attrs["U_I exponent"] = np.array([0.0])
                ds_grnr.attrs["U_L exponent"] = np.array([0.0])
                ds_grnr.attrs["U_M exponent"] = np.array([0.0])
                ds_grnr.attrs["U_T exponent"] = np.array([0.0])
                ds_grnr.attrs["U_t exponent"] = np.array([0.0])
                ds_grnr.attrs["a-scale exponent"] = np.array([0.0])
                ds_grnr.attrs["h-scale exponent"] = np.array([0.0])
                ds_rank.attrs[
                    "Conversion factor to CGS (including cosmological corrections)"
                ] = np.array([1.0])
                ds_rank.attrs[
                    "Conversion factor to CGS (not including cosmological corrections)"
                ] = np.array([1.0])
                ds_rank.attrs["Description"] = (
                    "Ranking by binding energy of the "
                    "bound particles (first in halo=0), or -1 if not bound"
                )
                ds_rank.attrs["U_I exponent"] = np.array([0.0])
                ds_rank.attrs["U_L exponent"] = np.array([0.0])
                ds_rank.attrs["U_M exponent"] = np.array([0.0])
                ds_rank.attrs["U_T exponent"] = np.array([0.0])
                ds_rank.attrs["U_t exponent"] = np.array([0.0])
                ds_rank.attrs["a-scale exponent"] = np.array([0.0])
                ds_rank.attrs["h-scale exponent"] = np.array([0.0])
    if create_virtual_snapshot:
        os.system(
            f"python {soap_script} "
            f"'{create_virtual_snapshot_from}' "
            f"'{membership_filebase}."
            "{file_nr}.hdf5' "
            f"'{virtual_snapshot_filename}' "
            "0"
        )


def remove_toysoap(
    filename: str = toysoap_filename,
    membership_filebase: str = toysoap_membership_filebase,
    virtual_snapshot_filename: str = toysoap_virtual_snapshot_filename,
) -> None:
    """
    Removes files created by :func:`~swiftgalaxy.demo_data.create_toysoap`. Any files
    not found are ignored.

    Parameters
    ----------
    filename : :obj:`str`
        The file name for the catalogue file to be removed. (Default: ``"toysoap.hdf5"``)

    membership_filebase : :obj:`str`
        The base name for membership files, completed as ``base.N.hdf5`` where ``N`` is
        an integer. (Default: ``"toysoap_membership"``)

    virtual_snapshot_filename : :obj:`str`
        Filename for virtual snapshot file. (Default: ``"toysnap_virtual.hdf5"``)
    """
    if os.path.isfile(filename):
        os.remove(filename)
    i = 0
    while True:
        membership_file = f"{membership_filebase}.{i}.hdf5"
        if os.path.isfile(membership_file):
            os.remove(membership_file)
            i += 1
        else:
            break
    if os.path.isfile(virtual_snapshot_filename):
        os.remove(virtual_snapshot_filename)
