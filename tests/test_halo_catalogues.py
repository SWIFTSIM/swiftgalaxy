import pytest
import h5py
import numpy as np
import unyt as u
from unyt.testing import assert_allclose_units
from toysnap import (
    toysnap_filename,
    toysoap_virtual_snapshot_filename,
    toysoap_membership_filebase,
    n_g_1,
    n_g_b,
    n_g_all,
    n_dm_1,
    n_dm_b,
    n_dm_all,
    n_s_1,
    n_s_2,
    n_bh_1,
    n_bh_2,
    m_g,
    # m_dm,
    m_s,
    m_bh,
    present_particle_types,
)
from swiftgalaxy import SWIFTGalaxy, MaskCollection
from swiftsimio.objects import cosmo_array

abstol_c = 1 * u.pc  # less than this is ~0
abstol_v = 10 * u.m / u.s  # less than this is ~0
abstol_m = 1e4 * u.Msun  # less than this is ~0
reltol_nd = 1.0e-4


class TestHaloCatalogues:
    def test_get_spatial_mask(self, hf, toysnap):
        """
        Check that we get spatial masks that we expect.
        """
        # don't use sg fixture here, just need the snapshot file
        # so don't want overhead of a SWIFTGalaxy
        spatial_mask = hf._get_spatial_mask(toysnap_filename)
        with h5py.File(toysnap_filename, "r") as snap:
            n_g_firstcell = snap["/Cells/Counts/PartType0"][0]
            n_dm_firstcell = snap["/Cells/Counts/PartType1"][0]
            n_s_firstcell = snap["/Cells/Counts/PartType4"][0]
            n_bh_firstcell = snap["/Cells/Counts/PartType5"][0]
        # We have 2 cells, covering 0-5 and 5-10 Mpc in x, and the entire range in y
        # and z. The galaxy is at (2,2,2)Mpc, so in the first cell.
        if hf._user_spatial_offsets is None:
            # all hf's except Standalone
            assert np.array_equal(spatial_mask.gas, np.array([[0, n_g_firstcell]]))
            assert np.array_equal(
                spatial_mask.dark_matter, np.array([[0, n_dm_firstcell]])
            )
            assert np.array_equal(spatial_mask.stars, np.array([[0, n_s_firstcell]]))
            assert np.array_equal(
                spatial_mask.black_holes, np.array([[0, n_bh_firstcell]])
            )
        else:
            # this is Standalone
            assert np.array_equal(
                spatial_mask.gas,
                np.array([[0, n_g_firstcell], [n_g_firstcell, n_g_all]]),
            )
            assert np.array_equal(
                spatial_mask.dark_matter,
                np.array([[0, n_dm_firstcell], [n_dm_firstcell, n_dm_all]]),
            )
            assert np.array_equal(
                spatial_mask.stars,
                np.array([[0, n_s_firstcell], [n_s_firstcell, n_s_1 + n_s_2]]),
            )
            assert np.array_equal(
                spatial_mask.black_holes,
                np.array([[0, n_bh_firstcell], [n_bh_firstcell, n_bh_1 + n_bh_2]]),
            )

    def test_get_user_spatial_mask(self, hf, toysnap):
        """
        Check that a user can override the automatic spatial mask.
        """
        # override to select both cells in the test snapshot
        hf._user_spatial_offsets = cosmo_array([[-5, 5], [-5, 5], [-5, 5]], u.Mpc)
        hf.extra_mask = None
        sg = SWIFTGalaxy(toysnap_filename, hf)
        generated_spatial_mask = sg._spatial_mask
        with h5py.File(toysnap_filename, "r") as snap:
            n_g_firstcell = snap["/Cells/Counts/PartType0"][0]
            n_dm_firstcell = snap["/Cells/Counts/PartType1"][0]
            n_s_firstcell = snap["/Cells/Counts/PartType4"][0]
            n_bh_firstcell = snap["/Cells/Counts/PartType5"][0]
        assert np.array_equal(
            generated_spatial_mask.gas,
            np.array([[0, n_g_firstcell], [n_g_firstcell, n_g_all]]),
        )
        assert np.array_equal(
            generated_spatial_mask.dark_matter,
            np.array([[0, n_dm_firstcell], [n_dm_firstcell, n_dm_all]]),
        )
        assert np.array_equal(
            generated_spatial_mask.stars,
            np.array([[0, n_s_firstcell], [n_s_firstcell, n_s_1 + n_s_2]]),
        )
        assert np.array_equal(
            generated_spatial_mask.black_holes,
            np.array([[0, n_bh_firstcell], [n_bh_firstcell, n_bh_1 + n_bh_2]]),
        )

    def test_get_bound_only_extra_mask(self, hf, toysnap_withfof):
        """
        Check that bound_only extra mask has the right shape.
        """
        hf.extra_mask = "bound_only"
        if hasattr(hf, "soap_file"):
            from toysnap import soap_script
            import os

            os.system(
                f"python {soap_script} "
                f"--absolute-paths "
                f"'{toysnap_filename}' "
                f"'{toysoap_membership_filebase}."
                "{file_nr}.hdf5' "
                f"'{toysoap_virtual_snapshot_filename}' "
                "0"
            )
            sg = SWIFTGalaxy(toysoap_virtual_snapshot_filename, hf)
        else:
            try:
                sg = SWIFTGalaxy(toysnap_filename, hf)
            except NotImplementedError:
                # expected for Standalone
                return
        generated_extra_mask = sg._extra_mask
        expected_shape = dict()
        for particle_type in present_particle_types.values():
            with h5py.File(toysnap_filename, "r") as snap:
                expected_shape[particle_type] = snap[
                    "Cells/Counts/PartType"
                    f"{dict(gas=0, dark_matter=1, stars=4, black_holes=5)[particle_type]}"
                ][0]
        if hasattr(hf, "_caesar") and hf.group_type == "galaxy":
            expected_shape["dark_matter"] = None
        for particle_type in present_particle_types.values():
            if expected_shape[particle_type] is not None:
                assert (
                    getattr(generated_extra_mask, particle_type).shape
                    == expected_shape[particle_type]
                )
                assert (
                    getattr(generated_extra_mask, particle_type).sum()
                    == dict(
                        gas=n_g_1, dark_matter=n_dm_1, stars=n_s_1, black_holes=n_bh_1
                    )[particle_type]
                )

    def test_get_void_extra_mask(self, hf, toysnap):
        """
        Check that None extra mask gives expected result.
        """
        hf.extra_mask = None
        sg = SWIFTGalaxy(toysnap_filename, hf)
        generated_extra_mask = sg._extra_mask
        for particle_type in present_particle_types.values():
            assert getattr(generated_extra_mask, particle_type) is None

    def test_get_user_extra_mask(self, hf, toysnap):
        """
        Check that extra masks of different kinds have the right shape or type.
        """
        hf.extra_mask = MaskCollection(
            gas=np.r_[np.ones(100, dtype=bool), np.zeros(n_g_all - 100, dtype=bool)],
            dark_matter=None,
            stars=np.r_[np.ones(100, dtype=bool), np.zeros(n_s_1 - 100, dtype=bool)],
            black_holes=np.ones(n_bh_1, dtype=bool),
        )
        sg = SWIFTGalaxy(toysnap_filename, hf)
        generated_extra_mask = sg._extra_mask
        for particle_type in present_particle_types.values():
            if getattr(generated_extra_mask, particle_type) is None:
                assert (
                    dict(gas=100, dark_matter=None, stars=100, black_holes=n_bh_1)[
                        particle_type
                    ]
                    is None
                )
            else:
                assert (
                    getattr(generated_extra_mask, particle_type).sum()
                    == dict(gas=100, dark_matter=None, stars=100, black_holes=n_bh_1)[
                        particle_type
                    ]
                )

    def test_centre(self, hf):
        """
        Check that the _centre function returns the expected centre.
        """
        # default is minpot == 2.001 Mpc
        assert_allclose_units(
            hf.centre,
            cosmo_array([2.001, 2.001, 2.001], u.Mpc),
            rtol=reltol_nd,
            atol=abstol_c,
        )

    def test_velocity_centre(self, hf):
        """
        Check that the velocity_centre function returns the expected velocity centre.
        """
        # default is minpot == 201. km/s
        assert_allclose_units(
            hf.velocity_centre,
            cosmo_array([201.0, 201.0, 201.0], u.km / u.s),
            rtol=reltol_nd,
            atol=abstol_v,
        )


class TestVelociraptor:
    def test_load(self, vr):
        """
        Check that the loading function is doing it's job.
        """
        # _load called during super().__init__
        assert vr._catalogue is not None

    @pytest.mark.parametrize(
        "centre_type, expected",
        (
            ("", 2.0),
            ("minpot", 2.001),
            ("mbp", 2.002),
            ("_gas", 2.003),
            ("_stars", 2.004),
        ),
    )
    def test_centre_types(self, vr, centre_type, expected):
        """
        Check that centres of each type retrieve expected values.
        """
        vr.centre_type = centre_type
        assert_allclose_units(
            vr.centre,
            cosmo_array([expected, expected, expected], u.Mpc),
            rtol=reltol_nd,
            atol=abstol_c,
        )

    @pytest.mark.parametrize(
        "centre_type, expected",
        (
            ("", 200.0),
            ("minpot", 201.0),
            ("mbp", 202.0),
            ("_gas", 203.0),
            ("_stars", 204.0),
        ),
    )
    def test_velocity_centre_types(self, vr, centre_type, expected):
        """
        Check that velocity centres of each type retrieve expected values.
        """
        vr.centre_type = centre_type
        assert_allclose_units(
            vr.velocity_centre,
            cosmo_array([expected, expected, expected], u.km / u.s),
            rtol=reltol_nd,
            atol=abstol_v,
        )

    def test_catalogue_exposed(self, vr):
        """
        Check that exposing the halo properties is working.
        """
        # pick one of the default attributes to check
        assert_allclose_units(
            vr.masses.mvir, 1.0e12 * u.Msun, rtol=reltol_nd, atol=abstol_m
        )


class TestVelociraptorWithSWIFTGalaxy:
    """
    Most interaction between the halo catalogue and swiftgalaxy.reader.SWIFTGalaxy
    is tested using the toysnap.ToyHF testing class (that inherits from
    swiftgalaxy.halo_catalogues._HaloCatalogue). Here we just want to test anything
    velociraptor-specific.
    """

    def test_catalogue_exposed(self, sg_vr):
        """
        Check that exposing the halo properties is working, through the
        SWIFTGalaxy object.
        """
        assert_allclose_units(
            sg_vr.halo_catalogue.masses.mvir,
            1.0e12 * u.Msun,
            rtol=reltol_nd,
            atol=abstol_m,
        )

    @pytest.mark.parametrize("particle_type", present_particle_types.values())
    def test_masks_compatible(self, sg_vr, particle_type):
        """
        Check that the bound_only default mask works with the spatial mask,
        giving the expected shapes for arrays.
        """
        assert (
            getattr(sg_vr, particle_type).masses.size
            == dict(gas=n_g_1, dark_matter=n_dm_1, stars=n_s_1, black_holes=n_bh_1)[
                particle_type
            ]
        )


class TestCaesar:
    def test_load(self, caesar):
        """
        Check that the loading function is doing it's job.
        """
        # _load called during super().__init__
        pass  # Caesar has nothing to do in _load

    @pytest.mark.parametrize(
        "centre_type, expected",
        (
            ("", 2.0),
            ("minpot", 2.001),
        ),
    )
    def test_centre_types(self, caesar, centre_type, expected):
        """
        Check that centres of each type retrieve expected values.
        """
        caesar.centre_type = centre_type
        assert_allclose_units(
            caesar.centre,
            cosmo_array([expected, expected, expected], u.Mpc),
            rtol=reltol_nd,
            atol=abstol_c,
        )

    @pytest.mark.parametrize(
        "centre_type, expected",
        (
            ("", 200.0),
            ("minpot", 201.0),
        ),
    )
    def test_vcentre_types(self, caesar, centre_type, expected):
        """
        Check that velocity centres of each type retrieve expected values.
        """
        caesar.centre_type = centre_type
        assert_allclose_units(
            caesar.velocity_centre,
            cosmo_array([expected, expected, expected], u.km / u.s),
            rtol=reltol_nd,
            atol=abstol_v,
        )

    def test_catalogue_exposed(self, caesar):
        """
        Check that exposing the halo properties is working.
        """
        # pick one of the default attributes to check
        if hasattr(caesar, "virial_quantities"):
            assert_allclose_units(
                caesar.virial_quantities["m200c"],
                1.0e12 * u.Msun,
                rtol=reltol_nd,
                atol=abstol_m,
            )
        elif hasattr(caesar, "masses"):
            assert_allclose_units(
                caesar.masses["total"],
                n_g_1 * m_g + n_s_1 * m_s + n_bh_1 * m_bh,
                rtol=reltol_nd,
                atol=abstol_m,
            )
        else:
            raise AttributeError

    def test_spatial_mask_applied(self, caesar, toysnap):
        """
        Check that we get the expected number of particles when only the spatial mask is
        applied.
        """
        caesar.extra_mask = None  # apply only the spatial mask
        sg = SWIFTGalaxy(toysnap_filename, caesar)
        for particle_type in present_particle_types.values():
            assert (
                getattr(sg, particle_type).masses.size
                == dict(
                    gas=n_g_b // 2 + n_g_1,
                    dark_matter=n_dm_b // 2 + n_dm_1,
                    stars=n_s_1,
                    black_holes=n_bh_1,
                )[particle_type]
            )


class TestCaesarWithSWIFTGalaxy:
    """
    Most interaction between the halo catalogue and swiftgalaxy.reader.SWIFTGalaxy
    is tested using the toysnap.ToyHF testing class (that inherits from
    swiftgalaxy.halo_catalogues._HaloCatalogue). Here we just want to test anything
    caesar-specific.
    """

    def test_catalogue_exposed(self, sg_caesar):
        """
        Check that exposing the halo properties is working, through the
        SWIFTGalaxy object.
        """
        if hasattr(sg_caesar.halo_catalogue, "virial_quantities"):
            assert_allclose_units(
                sg_caesar.halo_catalogue.virial_quantities["m200c"],
                1.0e12 * u.Msun,
                rtol=reltol_nd,
                atol=abstol_m,
            )
        elif hasattr(sg_caesar.halo_catalogue, "masses"):
            assert_allclose_units(
                sg_caesar.halo_catalogue.masses["total"],
                n_g_1 * m_g + n_s_1 * m_s + n_bh_1 * m_bh,
                rtol=reltol_nd,
                atol=abstol_m,
            )
        else:
            raise AttributeError

    @pytest.mark.parametrize("particle_type", present_particle_types.values())
    def test_masks_compatible(self, sg_caesar, particle_type):
        """
        Check that the bound_only default mask works with the spatial mask,
        giving the expected shapes for arrays.
        """
        expected_dm = 0 if sg_caesar.halo_catalogue.group_type == "galaxy" else n_dm_1
        assert (
            getattr(sg_caesar, particle_type).masses.size
            == dict(
                gas=n_g_1, dark_matter=expected_dm, stars=n_s_1, black_holes=n_bh_1
            )[particle_type]
        )


class TestStandalone:
    def test_spatial_mask_applied(self, sa, toysnap):
        """
        Check that we get the expected number of particles when only the spatial mask is
        applied.
        """
        sa.extra_mask = None  # apply only the spatial mask
        sg = SWIFTGalaxy(toysnap_filename, sa)
        for particle_type in present_particle_types.values():
            assert (
                getattr(sg, particle_type).masses.size
                == dict(
                    gas=n_g_b // 2 + n_g_1,
                    dark_matter=n_dm_b // 2 + n_dm_1,
                    stars=n_s_1,
                    black_holes=n_bh_1,
                )[particle_type]
            )


class TestSOAP:
    def test_load(self, soap):
        """
        Check that the loading function is doing it's job.
        """
        # _load called during super().__init__
        assert soap._swift_dataset is not None

    @pytest.mark.parametrize(
        "centre_type, expected",
        (
            ("bound_subhalo.centre_of_mass", 2.001),
            ("exclusive_sphere_100kpc.centre_of_mass", 2.003),
            ("inclusive_sphere_100kpc.centre_of_mass", 2.011),
            ("input_halos_fof.centres", 2.0),
            ("input_halos.halo_centre", 2.001),
            ("projected_aperture_50kpc_projx.centre_of_mass", 2.027),
            ("spherical_overdensity_200_crit.centre_of_mass", 2.032),
            ("spherical_overdensity_bn98.centre_of_mass", 2.038),
        ),
    )
    def test_centre_types(self, soap, centre_type, expected):
        """
        Check that centres of sample types retrieve expected values.
        """
        soap.centre_type = centre_type
        assert_allclose_units(
            soap.centre,
            cosmo_array([expected, expected, expected], u.Mpc),
            rtol=reltol_nd,
            atol=abstol_c,
        )

    @pytest.mark.parametrize(
        "velocity_centre_type, expected",
        (
            ("bound_subhalo.centre_of_mass_velocity", 201),
            ("exclusive_sphere_100kpc.centre_of_mass_velocity", 203),
            ("inclusive_sphere_100kpc.centre_of_mass_velocity", 211),
            ("projected_aperture_50kpc_projx.centre_of_mass_velocity", 227),
            ("spherical_overdensity_200_crit.centre_of_mass_velocity", 232),
            ("spherical_overdensity_bn98.centre_of_mass_velocity", 238),
        ),
    )
    def test_velocity_centre_types(self, soap, velocity_centre_type, expected):
        """
        Check that velocity centres of sample types retrieve expected values.
        """
        soap.velocity_centre_type = velocity_centre_type
        assert_allclose_units(
            soap.velocity_centre,
            cosmo_array([expected, expected, expected], u.km / u.s),
            rtol=reltol_nd,
            atol=abstol_v,
        )

    def test_catalogue_exposed(self, soap):
        """
        Check that exposing the halo properties is working.
        """
        # pick a couple of attributes to check
        assert_allclose_units(
            soap.input_halos_hbtplus.host_fofid,
            cosmo_array([1], comoving=False),
        )
        assert_allclose_units(
            soap.bound_subhalo.centre_of_mass,
            cosmo_array([[2.001, 2.001, 2.001]], u.Mpc, comoving=False),
            rtol=reltol_nd,
            atol=abstol_c,
        )


class TestSOAPWithSWIFTGalaxy:
    """
    Most interaction between the halo catalogue and swiftgalaxy.reader.SWIFTGalaxy
    is tested using the toysnap.ToyHF testing class (that inherits from
    swiftgalaxy.halo_catalogues._HaloCatalogue). Here we just want to test anything
    soap-specific.
    """

    def test_catalogue_exposed(self, sg_soap):
        """
        Check that exposing the halo properties is working, through the
        SWIFTGalaxy object.
        """
        # pick a couple of attributes to check
        assert_allclose_units(
            sg_soap.halo_catalogue.input_halos_hbtplus.host_fofid,
            cosmo_array([1], comoving=False),
        )
        assert_allclose_units(
            sg_soap.halo_catalogue.bound_subhalo.centre_of_mass,
            cosmo_array([[2.001, 2.001, 2.001]], u.Mpc, comoving=False),
            rtol=reltol_nd,
            atol=abstol_c,
        )

    @pytest.mark.parametrize("particle_type", present_particle_types.values())
    def test_masks_compatible(self, sg_soap, particle_type):
        """
        Check that the bound_only default mask works with the spatial mask,
        giving the expected shapes for arrays.
        """
        assert (
            getattr(sg_soap, particle_type).masses.size
            == dict(gas=n_g_1, dark_matter=n_dm_1, stars=n_s_1, black_holes=n_bh_1)[
                particle_type
            ]
        )
