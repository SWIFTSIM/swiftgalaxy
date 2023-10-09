import os
import h5py
import numpy as np
import unyt as u
from astropy.cosmology import LambdaCDM
from astropy import units as U
from swiftsimio.objects import cosmo_array
from swiftsimio import Writer
from swiftgalaxy import MaskCollection
from swiftgalaxy.halo_finders import _HaloFinder
from swiftsimio.units import cosmo_units

toysnap_filename = "toysnap.hdf5"
toyvr_filebase = "toyvr"
toycaesar_filename = "toycaesar.hdf5"
present_particle_types = {0: "gas", 1: "dark_matter", 4: "stars", 5: "black_holes"}
boxsize = 10.0 * u.Mpc
n_g_all = 32**3
n_g = 10000
n_g_b = n_g_all - n_g
n_dm_all = 32**3
n_dm = 10000
n_dm_b = n_dm_all - n_dm
n_s = 10000
n_bh = 1
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

    def _generate_bound_only_mask(self, SG):
        extra_mask = MaskCollection(
            gas=np.s_[-10000:], dark_matter=np.s_[-10000:], stars=..., black_holes=...
        )
        return extra_mask

    @property
    def centre(self):
        return cosmo_array([2, 2, 2], u.Mpc)

    @property
    def velocity_centre(self):
        return cosmo_array([200, 200, 200], u.km / u.s)


def create_toysnap(
    snapfile=toysnap_filename, alt_coord_name=None, alt_vel_name=None, alt_id_name=None
):
    """
    Creates a sample dataset of a toy galaxy.
    """

    sd = Writer(cosmo_units, np.ones(3, dtype=float) * boxsize)

    # Insert a uniform gas background plus a galaxy disc
    phi = np.random.rand(n_g, 1) * 2 * np.pi
    R = np.random.rand(n_g, 1)
    getattr(sd, present_particle_types[0]).particle_ids = np.arange(n_g_all)
    getattr(sd, present_particle_types[0]).coordinates = (
        np.vstack(
            (
                np.random.rand(n_g_b // 2, 3) * np.array([5, 10, 10]),
                np.hstack(
                    (
                        # 10 kpc disc radius, offcentred in box
                        2 + R * np.cos(phi) * 0.01,
                        2 + R * np.sin(phi) * 0.01,
                        2 + (np.random.rand(n_g, 1) * 2 - 1) * 0.001,  # 1 kpc height
                    )
                ),
                np.random.rand(n_g_b // 2, 3) * np.array([5, 10, 10])
                + np.array([5, 0, 0]),
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
                        200 + R * np.sin(phi) * 100,
                        200 + R * np.cos(phi) * 100,
                        200 + np.random.rand(n_g, 1) * 20 - 10,  # 10 km/s vertical
                    )
                ),
                np.random.rand(n_g_b // 2, 3) * 2 - 1,  # 1 km/s for background
            )
        )
        * u.km
        / u.s
    )
    getattr(sd, present_particle_types[0]).masses = (
        np.concatenate((np.ones(n_g_b, dtype=float), np.ones(n_g, dtype=float))) * m_g
    )
    getattr(sd, present_particle_types[0]).internal_energy = (
        np.concatenate(
            (
                np.ones(n_g_b, dtype=float),  # 1e4 K
                np.ones(n_g, dtype=float) / 10,  # 1e3 K
            )
        )
        * T_g
        * u.kb
        / (m_g)
    )
    getattr(sd, present_particle_types[0]).generate_smoothing_lengths(
        boxsize=boxsize, dimension=3
    )

    # Insert a uniform DM background plus a galaxy halo
    phi = np.random.rand(n_dm, 1) * 2 * np.pi
    theta = np.arccos(np.random.rand(n_dm, 1) * 2 - 1)
    r = np.random.rand(n_dm, 1)
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
                        2 + r * np.cos(phi) * np.sin(theta) * 0.1,
                        2 + r * np.sin(phi) * np.sin(theta) * 0.1,
                        2 + r * np.cos(theta) * 0.1,
                    )
                ),
                np.random.rand(n_dm_b // 2, 3) * np.array([5, 10, 10])
                + np.array([5, 0, 0]),
            )
        )
        * u.Mpc
    )
    getattr(sd, present_particle_types[1]).velocities = (
        np.vstack(
            (
                # 1 km/s background, 100 km/s halo
                np.random.rand(n_dm_b, 3) * 2 - 1,
                200 + (np.random.rand(n_dm, 3) * 2 - 1) * 100,
            )
        )
        * u.km
        / u.s
    )
    getattr(sd, present_particle_types[1]).masses = (
        np.concatenate((np.ones(n_dm_b, dtype=float), np.ones(n_dm, dtype=float)))
        * m_dm
    )
    getattr(sd, present_particle_types[1]).generate_smoothing_lengths(
        boxsize=boxsize, dimension=3
    )

    # Insert a galaxy stellar disc
    phi = np.random.rand(n_s, 1) * 2 * np.pi
    R = np.random.rand(n_s, 1)
    getattr(sd, present_particle_types[4]).particle_ids = np.arange(
        n_g_all + n_dm_all, n_g_all + n_dm_all + n_s
    )
    getattr(sd, present_particle_types[4]).coordinates = (
        np.hstack(
            (
                # 5 kpc disc radius, offcentred in box
                2 + R * np.cos(phi) * 0.005,
                2 + R * np.sin(phi) * 0.005,
                2 + (np.random.rand(n_s, 1) * 2 - 1) * 0.0005,  # 500 pc height
            )
        )
        * u.Mpc
    )
    getattr(sd, present_particle_types[4]).velocities = (
        np.hstack(
            (
                # solid body, 50 km/s at edge
                200 + R * np.sin(phi) * 50,
                200 + R * np.cos(phi) * 50,
                200 + np.random.rand(n_g, 1) * 20 - 10,  # 10 km/s vertical motions
            )
        )
        * u.km
        / u.s
    )
    getattr(sd, present_particle_types[4]).masses = np.ones(n_g, dtype=float) * m_s
    getattr(sd, present_particle_types[4]).generate_smoothing_lengths(
        boxsize=boxsize, dimension=3
    )
    # Insert a black hole
    getattr(sd, present_particle_types[5]).particle_ids = np.arange(
        n_g_all + n_dm_all + n_s, n_g_all + n_dm_all + n_s + n_bh
    )
    getattr(sd, present_particle_types[5]).coordinates = (
        2 - 0.000003 * np.ones((n_bh, 3), dtype=float)  # 3 pc to avoid r==0 warnings
    ) * u.Mpc
    getattr(sd, present_particle_types[5]).velocities = (
        (200 + np.zeros((n_bh, 3), dtype=float)) * u.km / u.s
    )
    getattr(sd, present_particle_types[5]).masses = np.ones(n_bh, dtype=float) * m_bh
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

    return


def remove_toysnap(snapfile=toysnap_filename):
    os.remove(snapfile)
    return


def create_toyvr(filebase=toyvr_filebase):
    with h5py.File(f"{toyvr_filebase}.properties", "w") as f:
        f.create_group("SimulationInfo")
        f["SimulationInfo"].attrs["ScaleFactor"] = 1.0
        f["SimulationInfo"].attrs["Cosmological_Sim"] = 1
        for coord in "XYZ":
            f.create_dataset(f"{coord}c", data=np.array([2.0], dtype=float))
            f.create_dataset(f"{coord}cminpot", data=np.array([2.001], dtype=float))
            f.create_dataset(f"{coord}cmbp", data=np.array([2.002], dtype=float))
            f.create_dataset(f"{coord}c_gas", data=np.array([0.003], dtype=float))
            f.create_dataset(f"{coord}c_stars", data=np.array([0.004], dtype=float))
            f.create_dataset(f"V{coord}c", data=np.array([200.0], dtype=float))
            f.create_dataset(f"V{coord}cminpot", data=np.array([201.0], dtype=float))
            f.create_dataset(f"V{coord}cmbp", data=np.array([202.0], dtype=float))
            f.create_dataset(f"V{coord}c_gas", data=np.array([3.0], dtype=float))
            f.create_dataset(f"V{coord}c_stars", data=np.array([4.0], dtype=float))
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
        f.create_dataset("File_id", data=np.array([0], dtype=int))
        f.create_dataset("ID", data=np.array([1], dtype=int))
        f["ID"].attrs["Dimension_Length"] = 0.0
        f["ID"].attrs["Dimension_Mass"] = 0.0
        f["ID"].attrs["Dimension_Time"] = 0.0
        f["ID"].attrs["Dimension_Velocity"] = 0.0
        # pick arbitrary particle in the galaxy to be most bound
        f.create_dataset("ID_mbp", data=np.array([32**3 - 9999], dtype=int))
        f["ID_mbp"].attrs["Dimension_Length"] = 0.0
        f["ID_mbp"].attrs["Dimension_Mass"] = 0.0
        f["ID_mbp"].attrs["Dimension_Time"] = 0.0
        f["ID_mbp"].attrs["Dimension_Velocity"] = 0.0
        # pick arbitrary particle in the galaxy to be potential minimum
        f.create_dataset("ID_minpot", data=np.array([32**3 - 9998], dtype=int))
        f["ID_minpot"].attrs["Dimension_Length"] = 0.0
        f["ID_minpot"].attrs["Dimension_Mass"] = 0.0
        f["ID_minpot"].attrs["Dimension_Time"] = 0.0
        f["ID_minpot"].attrs["Dimension_Velocity"] = 0.0
        f.create_dataset("Mvir", data=np.array([100.0], dtype=float))
        f.create_dataset("Mass_200crit", data=np.array([100.0], dtype=float))
        f.create_dataset("Mass_200mean", data=np.array([100.0], dtype=float))
        f.create_dataset("Mass_BN98", data=np.array([100.0], dtype=float))
        f.create_dataset("Mass_FOF", data=np.array([100.0], dtype=float))
        for field in ("Mvir", "Mass_200crit", "Mass_200mean", "Mass_BN98", "Mass_FOF"):
            f[field].attrs["Dimension_Length"] = 0.0
            f[field].attrs["Dimension_Mass"] = 1.0
            f[field].attrs["Dimension_Time"] = 0.0
            f[field].attrs["Dimension_Velocity"] = 0.0
        f.create_dataset("R_200crit", data=np.array([0.3], dtype=float))
        f.create_dataset("R_200mean", data=np.array([0.3], dtype=float))
        f.create_dataset("R_BN98", data=np.array([0.3], dtype=float))
        f.create_dataset("R_size", data=np.array([0.3], dtype=float))
        f.create_dataset("Rmax", data=np.array([0.3], dtype=float))
        f.create_dataset("Rvir", data=np.array([0.3], dtype=float))
        for field in ("R_200crit", "R_200mean", "R_BN98", "R_size", "Rmax", "Rvir"):
            f[field].attrs["Dimension_Length"] = 1.0
            f[field].attrs["Dimension_Mass"] = 0.0
            f[field].attrs["Dimension_Time"] = 0.0
            f[field].attrs["Dimension_Velocity"] = 0.0
        f.create_dataset("Num_of_files", data=np.array([1], dtype=int))
        f.create_dataset("Num_of_groups", data=np.array([1], dtype=int))
        f.create_dataset("Structuretype", data=np.array([10], dtype=int))
        f["Structuretype"].attrs["Dimension_Length"] = 0.0
        f["Structuretype"].attrs["Dimension_Mass"] = 0.0
        f["Structuretype"].attrs["Dimension_Time"] = 0.0
        f["Structuretype"].attrs["Dimension_Velocity"] = 0.0
        f.create_dataset("Total_num_of_groups", data=np.array([1], dtype=int))
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
        f.create_dataset("n_bh", data=np.array([n_bh], dtype=int))
        f.create_dataset("n_gas", data=np.array([n_g], dtype=int))
        f.create_dataset("n_star", data=np.array([n_s], dtype=int))
        f.create_dataset("npart", data=np.array([n_g + n_dm + n_s + n_bh], dtype=int))
        for pt in ("_bh", "_gas", "_star", "part"):
            f[f"n{pt}"].attrs["Dimension_Length"] = 0.0
            f[f"n{pt}"].attrs["Dimension_Mass"] = 0.0
            f[f"n{pt}"].attrs["Dimension_Time"] = 0.0
            f[f"n{pt}"].attrs["Dimension_Velocity"] = 0.0
        f.create_dataset("numSubStruct", data=np.array([0], dtype=int))
        f["numSubStruct"].attrs["Dimension_Length"] = 0.0
        f["numSubStruct"].attrs["Dimension_Mass"] = 0.0
        f["numSubStruct"].attrs["Dimension_Time"] = 0.0
        f["numSubStruct"].attrs["Dimension_Velocity"] = 0.0
    with h5py.File(f"{toyvr_filebase}.catalog_groups", "w") as f:
        f.create_dataset("File_id", data=np.array([0], dtype=int))
        f.create_dataset(
            "Group_Size", data=np.array([n_g_all + n_dm_all + n_s + n_bh], dtype=int)
        )
        f.create_dataset("Num_of_files", data=np.array([1], dtype=int))
        f.create_dataset("Num_of_groups", data=np.array([1], dtype=int))
        f.create_dataset(
            "Number_of_substructures_in_halo", data=np.array([0], dtype=int)
        )
        f.create_dataset("Offset", data=np.array([0], dtype=int))
        f.create_dataset("Offset_unbound", data=np.array([0], dtype=int))
        f.create_dataset("Parent_halo_ID", data=np.array([-1], dtype=int))
        f.create_dataset("Total_num_of_groups", data=np.array([1], dtype=int))
    with h5py.File(f"{toyvr_filebase}.catalog_particles", "w") as f:
        f.create_dataset("File_id", data=np.array([0], dtype=int))
        f.create_dataset("Num_of_files", data=np.array([1], dtype=int))
        f.create_dataset(
            "Num_of_particles_in_groups",
            data=np.array([n_g + n_dm + n_s + n_bh], dtype=int),
        )
        f.create_dataset(
            "Particle_IDs",
            data=np.concatenate(
                (
                    np.arange(n_g_b // 2, n_g_b // 2 + n_g, dtype=int),
                    np.arange(
                        n_g_all + n_dm_b // 2, n_g_all + n_dm_b // 2 + n_dm, dtype=int
                    ),
                    np.arange(n_g_all + n_dm_all, n_g_all + n_dm_all + n_s, dtype=int),
                    np.arange(
                        n_g_all + n_dm_all + n_s,
                        n_g_all + n_dm_all + n_s + n_bh,
                        dtype=int,
                    ),
                )
            ),
        )
        f.create_dataset(
            "Total_num_of_particles_in_all_groups",
            data=np.array([n_g + n_dm + n_s + n_bh], dtype=int),
        )
    with h5py.File(f"{toyvr_filebase}.catalog_particles.unbound", "w") as f:
        f.create_dataset("File_id", data=np.array([0], dtype=int))
        f.create_dataset("Num_of_files", data=np.array([1], dtype=int))
        f.create_dataset(
            "Num_of_particles_in_groups", data=np.array([n_g_b + n_dm_b], dtype=int)
        )
        f.create_dataset(
            "Particle_IDs",
            data=np.concatenate(
                (
                    np.arange(n_g_b // 2, dtype=int),
                    np.arange(n_g_b // 2 + n_g, n_g_all, dtype=int),
                    np.arange(n_g_all, n_g_all + n_dm_b // 2, dtype=int),
                    np.arange(
                        n_g_all + n_dm_b // 2 + n_dm, n_g_all + n_dm_all, dtype=int
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
            data=np.array([n_g + n_dm + n_s + n_bh], dtype=int),
        )
        f.create_dataset(
            "Particle_types",
            data=np.concatenate(
                (
                    0 * np.ones(n_g, dtype=int),
                    1 * np.ones(n_dm, dtype=int),
                    4 * np.ones(n_s, dtype=int),
                    5 * np.ones(n_bh, dtype=int),
                )
            ),
        )
        f.create_dataset(
            "Total_num_of_particles_in_all_groups",
            data=np.array([n_g + n_dm + n_s + n_bh], dtype=int),
        )
    with h5py.File(f"{toyvr_filebase}.catalog_parttypes.unbound", "w") as f:
        f.create_dataset("File_id", data=np.array([0], dtype=int))
        f.create_dataset("Num_of_files", data=np.array([1], dtype=int))
        f.create_dataset(
            "Num_of_particles_in_groups", data=np.array([n_g_b + n_dm_b], dtype=int)
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


def remove_toyvr(filebase=toyvr_filebase):
    os.remove(f"{toyvr_filebase}.properties")
    os.remove(f"{toyvr_filebase}.catalog_groups")
    os.remove(f"{toyvr_filebase}.catalog_particles")
    os.remove(f"{toyvr_filebase}.catalog_particles.unbound")
    os.remove(f"{toyvr_filebase}.catalog_parttypes")
    os.remove(f"{toyvr_filebase}.catalog_parttypes.unbound")
    return


def create_toycaesar(filename=toycaesar_filename):
    with h5py.File(filename, "w") as f:
        f.attrs["caesar"] = "fake"
        f.attrs["nclouds"] = 0
        f.attrs["ngalaxies"] = 1
        f.attrs["nhalos"] = 1
        with open("tests/json/caesar_unit_registry.json") as json:
            f.attrs["unit_registry_json"] = json.read()
        f.create_group("galaxy_data")
        f["/galaxy_data"].create_dataset("GroupID", data=np.array([0], dtype=int))
        f["/galaxy_data"].create_dataset("bhlist_end", data=np.array([n_bh], dtype=int))
        f["/galaxy_data"].create_dataset("bhlist_start", data=np.array([0], dtype=int))
        f["/galaxy_data"].create_dataset("central", data=np.array([True], dtype=bool))
        f["/galaxy_data"].create_group("dicts")
        f["/galaxy_data/dicts"].create_dataset(
            "masses.total",
            data=np.array(
                [(n_g * m_g + n_s * m_s + n_bh * m_bh).to_value(u.msun)], dtype=float
            ),
        )
        f["/galaxy_data/dicts/masses.total"].attrs["unit"] = "Msun"
        f["/galaxy_data/dicts"].create_dataset(
            "radii.total_rmax", data=np.array([100], dtype=float)
        )
        f["/galaxy_data/dicts/radii.total_rmax"].attrs["unit"] = "kpccm"
        f["/galaxy_data"].create_dataset("glist_end", data=np.array([n_g], dtype=int))
        f["/galaxy_data"].create_dataset("glist_start", data=np.array([0], dtype=int))
        f["/galaxy_data"].create_group("lists")
        f["/galaxy_data/lists"].create_dataset("bhlist", data=np.array([0], dtype=int))
        f["/galaxy_data/lists"].create_dataset(
            "glist", data=np.arange(n_g_b // 2, n_g_b // 2 + n_g, dtype=int)
        )
        f["/galaxy_data/lists"].create_dataset("slist", data=np.arange(n_s, dtype=int))
        f["/galaxy_data"].create_dataset(
            "minpotpos", data=np.array([[2001.0, 2001.0, 2001.0]], dtype=float)
        )
        f["/galaxy_data/minpotpos"].attrs["unit"] = "kpccm"
        f["/galaxy_data"].create_dataset(
            "minpotvel", data=np.array([[201.0, 201.0, 201.0]], dtype=float)
        )
        f["/galaxy_data/minpotvel"].attrs["unit"] = "km/s"
        f["/galaxy_data"].create_dataset("nbh", data=np.array([n_bh], dtype=int))
        f["/galaxy_data"].create_dataset("ndm", data=np.array([0], dtype=int))
        f["/galaxy_data"].create_dataset("ndm2", data=np.array([0], dtype=int))
        f["/galaxy_data"].create_dataset("ndm3", data=np.array([0], dtype=int))
        f["/galaxy_data"].create_dataset("ndust", data=np.array([0], dtype=int))
        f["/galaxy_data"].create_dataset("ngas", data=np.array([n_g], dtype=int))
        f["/galaxy_data"].create_dataset("nstar", data=np.array([n_s], dtype=int))
        f["/galaxy_data"].create_dataset(
            "parent_halo_index", data=np.array([0], dtype=int)
        )
        f["/galaxy_data"].create_dataset(
            "pos", data=np.array([[2000.0, 2000.0, 2000.0]], dtype=float)
        )
        f["/galaxy_data/pos"].attrs["unit"] = "kpccm"
        f["/galaxy_data"].create_dataset("slist_end", data=np.array([n_s], dtype=int))
        f["/galaxy_data"].create_dataset("slist_start", data=np.array([0], dtype=int))
        f["/galaxy_data"].create_dataset(
            "vel", data=np.array([[200.0, 200.0, 200.0]], dtype=float)
        )
        f["/galaxy_data/vel"].attrs["unit"] = "km/s"
        f.create_group("global_lists")
        f["/global_lists"].create_dataset(
            "galaxy_bhlist", data=np.zeros(n_bh, dtype=int)
        )
        f["/global_lists"].create_dataset(
            "galaxy_glist",
            data=np.r_[
                -np.ones(n_g_b // 2, dtype=int),
                np.zeros(n_g, dtype=int),
                -np.ones(n_g_b // 2, dtype=int),
            ],
        )
        f["/global_lists"].create_dataset("galaxy_slist", data=np.zeros(n_s, dtype=int))
        f["/global_lists"].create_dataset("halo_bhlist", data=np.zeros(n_bh, dtype=int))
        f["/global_lists"].create_dataset(
            "halo_dmlist",
            data=np.r_[
                -np.ones(n_dm_b // 2, dtype=int),
                np.zeros(n_dm, dtype=int),
                -np.ones(n_dm_b // 2, dtype=int),
            ],
        )
        f["/global_lists"].create_dataset(
            "halo_glist",
            data=np.r_[
                -np.ones(n_g_b // 2, dtype=int),
                np.zeros(n_g, dtype=int),
                -np.ones(n_g_b // 2, dtype=int),
            ],
        )
        f["/global_lists"].create_dataset("halo_slist", data=np.zeros(n_s, dtype=int))
        f.create_group("halo_data")
        f["/halo_data"].create_dataset("GroupID", data=np.array([0], dtype=int))
        f["/halo_data"].create_dataset("bhlist_end", data=np.array([n_bh], dtype=int))
        f["/halo_data"].create_dataset("bhlist_start", data=np.array([0], dtype=int))
        f["/halo_data"].create_dataset("central_galaxy", data=np.array([0], dtype=int))
        f["/halo_data"].create_dataset("child", data=np.array([False], dtype=bool))
        f["/halo_data"].create_group("dicts")
        f["/halo_data/dicts"].create_dataset(
            "masses.total",
            data=np.array(
                [(n_g * m_g + n_dm * m_dm + n_s * m_s + n_bh * m_bh).to_value(u.msun)],
                dtype=float,
            ),
        )
        f["/halo_data/dicts"].create_dataset(
            "radii.total_rmax", data=np.array([100], dtype=float)
        )
        f["/halo_data/dicts/radii.total_rmax"].attrs["unit"] = "kpccm"

        f["/halo_data/dicts"].create_dataset(
            "virial_quantities.m200c", data=np.array([1.0e12], dtype=float)
        )
        f["/halo_data/dicts/virial_quantities.m200c"].attrs["unit"] = "Msun"
        f["/halo_data"].create_dataset("dmlist_end", data=np.array([n_dm], dtype=int))
        f["/halo_data"].create_dataset("dmlist_start", data=np.array([0], dtype=int))
        f["/halo_data"].create_dataset(
            "galaxy_index_list_end", data=np.array([1], dtype=int)
        )
        f["/halo_data"].create_dataset(
            "galaxy_index_list_start", data=np.array([0], dtype=int)
        )
        f["/halo_data"].create_dataset("glist_end", data=np.array([n_g], dtype=int))
        f["/halo_data"].create_dataset("glist_start", data=np.array([0], dtype=int))
        f["/halo_data"].create_group("lists")
        f["/halo_data/lists"].create_dataset("bhlist", data=np.array([0], dtype=int))
        f["/halo_data/lists"].create_dataset(
            "dmlist", data=np.arange(n_dm_b // 2, n_dm_b // 2 + n_dm, dtype=int)
        )
        f["/halo_data/lists"].create_dataset(
            "galaxy_index_list", data=np.array([0], dtype=int)
        )
        f["/halo_data/lists"].create_dataset(
            "glist", data=np.arange(n_g_b // 2, n_g_b // 2 + n_g, dtype=int)
        )
        f["/halo_data/lists"].create_dataset("slist", data=np.arange(n_s, dtype=int))
        f["/halo_data"].create_dataset(
            "minpotpos", data=np.array([[2001.0, 2001.0, 2001.0]], dtype=float)
        )
        f["/halo_data/minpotpos"].attrs["unit"] = "kpccm"
        f["/halo_data"].create_dataset(
            "minpotvel", data=np.array([[201.0, 201.0, 201.0]], dtype=float)
        )
        f["/halo_data/minpotvel"].attrs["unit"] = "km/s"
        f["/halo_data"].create_dataset("nbh", data=np.array([n_bh], dtype=int))
        f["/halo_data"].create_dataset("ndm", data=np.array([0], dtype=int))
        f["/halo_data"].create_dataset("ndm2", data=np.array([0], dtype=int))
        f["/halo_data"].create_dataset("ndm3", data=np.array([0], dtype=int))
        f["/halo_data"].create_dataset("ndust", data=np.array([0], dtype=int))
        f["/halo_data"].create_dataset("ngas", data=np.array([n_g], dtype=int))
        f["/halo_data"].create_dataset("nstar", data=np.array([n_s], dtype=int))
        f["/halo_data"].create_dataset(
            "pos", data=np.array([[2000.0, 2000.0, 2000.0]], dtype=float)
        )
        f["/halo_data/pos"].attrs["unit"] = "kpccm"
        f["/halo_data"].create_dataset("slist_end", data=np.array([n_s], dtype=int))
        f["/halo_data"].create_dataset("slist_start", data=np.array([0], dtype=int))
        f["/halo_data"].create_dataset(
            "vel", data=np.array([[200.0, 200.0, 200.0]], dtype=float)
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
        f["/simulation_attributes"].attrs[
            "mean_interparticle_separation"
        ] = boxsize.to_value(u.kpc) / n_dm ** (1 / 3)
        f["/simulation_attributes"].attrs["nbh"] = n_bh
        f["/simulation_attributes"].attrs["ndm"] = n_dm_all
        f["/simulation_attributes"].attrs["ndust"] = 0
        f["/simulation_attributes"].attrs["ngas"] = n_g_all
        f["/simulation_attributes"].attrs["nstar"] = n_s
        f["/simulation_attributes"].attrs["ntot"] = n_g_all + n_dm_all + n_s + n_bh
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


def remove_toycaesar(filename=toycaesar_filename):
    os.remove(filename)
    return
