import pytest
import numpy as np
import unyt as u
from unyt.testing import assert_allclose_units
from toysnap import (
    toysnap_filename,
    n_g,
    n_g_all,
    n_dm,
    n_dm_all,
    n_s,
    n_bh,
    m_g,
    # m_dm,
    m_s,
    m_bh,
    present_particle_types,
    create_toysnap,
    remove_toysnap,
)
from swiftgalaxy import MaskCollection
from swiftsimio.objects import cosmo_array

abstol_c = 1 * u.pc  # less than this is ~0
abstol_v = 10 * u.m / u.s  # less than this is ~0
abstol_m = 1e4 * u.Msun  # less than this is ~0
reltol_nd = 1.0e-4


# FIGURE OUT HOW TO PARAMETRIZE BY FIXTURES VR, CAESAR_GALAXY, CAESAR_HALO
# THEN MERGE THE __MERGE1 __MERGE2 __MERGE3 classes
class TestHaloFinders__MERGE1:
    def test_get_spatial_mask(self, vr):
        """
        Check that we get spatial masks that we expect.
        """
        # don't use sg fixture here, just need the snapshot file
        # so don't want overhead of a SWIFTGalaxy
        create_toysnap()
        spatial_mask = vr._get_spatial_mask(toysnap_filename)
        remove_toysnap()
        assert np.array_equal(spatial_mask.gas, np.array([[0, n_g_all]]))
        assert np.array_equal(spatial_mask.dark_matter, np.array([[0, n_dm_all]]))
        assert np.array_equal(spatial_mask.stars, np.array([[0, n_s]]))
        assert np.array_equal(spatial_mask.black_holes, np.array([[0, n_bh]]))

    @pytest.mark.parametrize(
        "extra_mask, expected",
        (
            (
                "bound_only",
                dict(gas=n_g, dark_matter=n_dm, stars=n_s, black_holes=n_bh),
            ),
            (None, dict(gas=None, dark_matter=None, stars=None, black_holes=None)),
            (
                MaskCollection(
                    gas=np.r_[
                        np.ones(100, dtype=bool), np.zeros(n_g_all - 100, dtype=bool)
                    ],
                    dark_matter=None,
                    stars=np.r_[
                        np.ones(100, dtype=bool), np.zeros(n_s - 100, dtype=bool)
                    ],
                    black_holes=np.ones(n_bh, dtype=bool),
                ),
                dict(gas=100, dark_matter=None, stars=100, black_holes=n_bh),
            ),
        ),
    )
    def test_get_extra_mask(self, sg, vr, extra_mask, expected):
        """
        Check that extra masks of different kinds have the right shape or type.
        """
        if hasattr(vr, "_caesar"):
            if vr.group_type == "galaxy":
                expected["dark_matter"] = None
        vr.extra_mask = extra_mask
        generated_extra_mask = vr._get_extra_mask(sg)
        for particle_type in present_particle_types.values():
            if getattr(generated_extra_mask, particle_type) is not None:
                assert (
                    getattr(generated_extra_mask, particle_type).sum()
                    == expected[particle_type]
                )
            else:
                assert expected[particle_type] is None

    def test_centre(self, vr):
        """
        Check that the _centre function returns the expected centre.
        """
        # default is minpot == 2.001 Mpc
        assert_allclose_units(
            vr._centre(),
            cosmo_array([2.001, 2.001, 2.001], u.Mpc),
            rtol=reltol_nd,
            atol=abstol_c,
        )

    def test_vcentre(self, vr):
        """
        Check that the _vcentre function returns the expected velocity centre.
        """
        # default is minpot == 201. km/s
        assert_allclose_units(
            vr._vcentre(),
            cosmo_array([201.0, 201.0, 201.0], u.km / u.s),
            rtol=reltol_nd,
            atol=abstol_v,
        )


class TestHaloFinders__MERGE2:
    def test_get_spatial_mask(self, caesar_halo):
        """
        Check that we get spatial masks that we expect.
        """
        # don't use sg fixture here, just need the snapshot file
        # so don't want overhead of a SWIFTGalaxy
        create_toysnap()
        spatial_mask = caesar_halo._get_spatial_mask(toysnap_filename)
        remove_toysnap()
        assert np.array_equal(spatial_mask.gas, np.array([[0, n_g_all]]))
        assert np.array_equal(spatial_mask.dark_matter, np.array([[0, n_dm_all]]))
        assert np.array_equal(spatial_mask.stars, np.array([[0, n_s]]))
        assert np.array_equal(spatial_mask.black_holes, np.array([[0, n_bh]]))

    @pytest.mark.parametrize(
        "extra_mask, expected",
        (
            (
                "bound_only",
                dict(gas=n_g, dark_matter=n_dm, stars=n_s, black_holes=n_bh),
            ),
            (None, dict(gas=None, dark_matter=None, stars=None, black_holes=None)),
            (
                MaskCollection(
                    gas=np.r_[
                        np.ones(100, dtype=bool), np.zeros(n_g_all - 100, dtype=bool)
                    ],
                    dark_matter=None,
                    stars=np.r_[
                        np.ones(100, dtype=bool), np.zeros(n_s - 100, dtype=bool)
                    ],
                    black_holes=np.ones(n_bh, dtype=bool),
                ),
                dict(gas=100, dark_matter=None, stars=100, black_holes=n_bh),
            ),
        ),
    )
    def test_get_extra_mask(self, sg, caesar_halo, extra_mask, expected):
        """
        Check that extra masks of different kinds have the right shape or type.
        """
        if hasattr(caesar_halo, "_caesar"):
            if caesar_halo.group_type == "galaxy":
                expected["dark_matter"] = None
        caesar_halo.extra_mask = extra_mask
        generated_extra_mask = caesar_halo._get_extra_mask(sg)
        for particle_type in present_particle_types.values():
            if getattr(generated_extra_mask, particle_type) is not None:
                assert (
                    getattr(generated_extra_mask, particle_type).sum()
                    == expected[particle_type]
                )
            else:
                assert expected[particle_type] is None

    def test_centre(self, caesar_halo):
        """
        Check that the _centre function returns the expected centre.
        """
        # default is minpot == 2.001 Mpc
        assert_allclose_units(
            caesar_halo._centre(),
            cosmo_array([2.001, 2.001, 2.001], u.Mpc),
            rtol=reltol_nd,
            atol=abstol_c,
        )

    def test_vcentre(self, caesar_halo):
        """
        Check that the _vcentre function returns the expected velocity centre.
        """
        # default is minpot == 201. km/s
        assert_allclose_units(
            caesar_halo._vcentre(),
            cosmo_array([201.0, 201.0, 201.0], u.km / u.s),
            rtol=reltol_nd,
            atol=abstol_v,
        )


class TestHaloFinders__MERGE3:
    def test_get_spatial_mask(self, caesar_galaxy):
        """
        Check that we get spatial masks that we expect.
        """
        # don't use sg fixture here, just need the snapshot file
        # so don't want overhead of a SWIFTGalaxy
        create_toysnap()
        spatial_mask = caesar_galaxy._get_spatial_mask(toysnap_filename)
        remove_toysnap()
        assert np.array_equal(spatial_mask.gas, np.array([[0, n_g_all]]))
        assert np.array_equal(spatial_mask.dark_matter, np.array([[0, n_dm_all]]))
        assert np.array_equal(spatial_mask.stars, np.array([[0, n_s]]))
        assert np.array_equal(spatial_mask.black_holes, np.array([[0, n_bh]]))

    @pytest.mark.parametrize(
        "extra_mask, expected",
        (
            (
                "bound_only",
                dict(gas=n_g, dark_matter=n_dm, stars=n_s, black_holes=n_bh),
            ),
            (None, dict(gas=None, dark_matter=None, stars=None, black_holes=None)),
            (
                MaskCollection(
                    gas=np.r_[
                        np.ones(100, dtype=bool), np.zeros(n_g_all - 100, dtype=bool)
                    ],
                    dark_matter=None,
                    stars=np.r_[
                        np.ones(100, dtype=bool), np.zeros(n_s - 100, dtype=bool)
                    ],
                    black_holes=np.ones(n_bh, dtype=bool),
                ),
                dict(gas=100, dark_matter=None, stars=100, black_holes=n_bh),
            ),
        ),
    )
    def test_get_extra_mask(self, sg, caesar_galaxy, extra_mask, expected):
        """
        Check that extra masks of different kinds have the right shape or type.
        """
        if hasattr(caesar_galaxy, "_caesar"):
            if caesar_galaxy.group_type == "galaxy":
                expected["dark_matter"] = None
        caesar_galaxy.extra_mask = extra_mask
        generated_extra_mask = caesar_galaxy._get_extra_mask(sg)
        for particle_type in present_particle_types.values():
            if getattr(generated_extra_mask, particle_type) is not None:
                assert (
                    getattr(generated_extra_mask, particle_type).sum()
                    == expected[particle_type]
                )
            else:
                assert expected[particle_type] is None

    def test_centre(self, caesar_galaxy):
        """
        Check that the _centre function returns the expected centre.
        """
        # default is minpot == 2.001 Mpc
        assert_allclose_units(
            caesar_galaxy._centre(),
            cosmo_array([2.001, 2.001, 2.001], u.Mpc),
            rtol=reltol_nd,
            atol=abstol_c,
        )

    def test_vcentre(self, caesar_galaxy):
        """
        Check that the _vcentre function returns the expected velocity centre.
        """
        # default is minpot == 201. km/s
        assert_allclose_units(
            caesar_galaxy._vcentre(),
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
            vr._centre(),
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
    def test_vcentre_types(self, vr, centre_type, expected):
        """
        Check that velocity centres of each type retrieve expected values.
        """
        vr.centre_type = centre_type
        print(vr._vcentre())
        assert_allclose_units(
            vr._vcentre(),
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
    Most interaction between the halo finder and swiftgalaxy.reader.SWIFTGalaxy
    is tested using the toysnap.ToyHF testing class (that inherits from
    swiftgalaxy.halo_finders._HaloFinder). Here we just want to test anything
    velociraptor-specific.
    """

    def test_catalogue_exposed(self, sg_vr):
        """
        Check that exposing the halo properties is working, through the
        SWIFTGalaxy object.
        """
        assert_allclose_units(
            sg_vr.halo_finder.masses.mvir,
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
            == dict(gas=10000, dark_matter=10000, stars=10000, black_holes=1)[
                particle_type
            ]
        )


class TestCaesar__MERGE1:
    def test_load(self, caesar_halo):
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
    def test_centre_types(self, caesar_halo, centre_type, expected):
        """
        Check that centres of each type retrieve expected values.
        """
        caesar_halo.centre_type = centre_type
        assert_allclose_units(
            caesar_halo._centre(),
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
    def test_vcentre_types(self, caesar_halo, centre_type, expected):
        """
        Check that velocity centres of each type retrieve expected values.
        """
        caesar_halo.centre_type = centre_type
        print(caesar_halo._vcentre())
        assert_allclose_units(
            caesar_halo._vcentre(),
            cosmo_array([expected, expected, expected], u.km / u.s),
            rtol=reltol_nd,
            atol=abstol_v,
        )

    def test_catalogue_exposed(self, caesar_halo):
        """
        Check that exposing the halo properties is working.
        """
        # pick one of the default attributes to check
        assert_allclose_units(
            caesar_halo.virial_quantities["m200c"],
            1.0e12 * u.Msun,
            rtol=reltol_nd,
            atol=abstol_m,
        )


class TestCaesar__MERGE2:
    def test_load(self, caesar_galaxy):
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
    def test_centre_types(self, caesar_galaxy, centre_type, expected):
        """
        Check that centres of each type retrieve expected values.
        """
        caesar_galaxy.centre_type = centre_type
        assert_allclose_units(
            caesar_galaxy._centre(),
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
    def test_vcentre_types(self, caesar_galaxy, centre_type, expected):
        """
        Check that velocity centres of each type retrieve expected values.
        """
        caesar_galaxy.centre_type = centre_type
        print(caesar_galaxy._vcentre())
        assert_allclose_units(
            caesar_galaxy._vcentre(),
            cosmo_array([expected, expected, expected], u.km / u.s),
            rtol=reltol_nd,
            atol=abstol_v,
        )

    def test_catalogue_exposed(self, caesar_galaxy):
        """
        Check that exposing the halo properties is working.
        """
        # pick one of the default attributes to check
        assert_allclose_units(
            caesar_galaxy.masses["total"],
            n_g * m_g + n_s * m_s + n_bh * m_bh,
            rtol=reltol_nd,
            atol=abstol_m,
        )


class TestCaesarWithSWIFTGalaxy__MERGE1:
    """
    Most interaction between the halo finder and swiftgalaxy.reader.SWIFTGalaxy
    is tested using the toysnap.ToyHF testing class (that inherits from
    swiftgalaxy.halo_finders._HaloFinder). Here we just want to test anything
    caesar-specific.
    """

    def test_catalogue_exposed(self, sg_caesar_halo):
        """
        Check that exposing the halo properties is working, through the
        SWIFTGalaxy object.
        """
        assert_allclose_units(
            sg_caesar_halo.halo_finder.virial_quantities["m200c"],
            1.0e12 * u.Msun,
            rtol=reltol_nd,
            atol=abstol_m,
        )

    @pytest.mark.parametrize("particle_type", present_particle_types.values())
    def test_masks_compatible(self, sg_caesar_halo, particle_type):
        """
        Check that the bound_only default mask works with the spatial mask,
        giving the expected shapes for arrays.
        """
        expected_dm = 0 if sg_caesar_halo.halo_finder.group_type == "galaxy" else 10000
        assert (
            getattr(sg_caesar_halo, particle_type).masses.size
            == dict(gas=10000, dark_matter=expected_dm, stars=10000, black_holes=1)[
                particle_type
            ]
        )


class TestCaesarWithSWIFTGalaxy__MERGE2:
    """
    Most interaction between the halo finder and swiftgalaxy.reader.SWIFTGalaxy
    is tested using the toysnap.ToyHF testing class (that inherits from
    swiftgalaxy.halo_finders._HaloFinder). Here we just want to test anything
    caesar-specific.
    """

    def test_catalogue_exposed(self, sg_caesar_galaxy):
        """
        Check that exposing the halo properties is working, through the
        SWIFTGalaxy object.
        """
        assert_allclose_units(
            sg_caesar_galaxy.halo_finder.masses["total"],
            n_g * m_g + n_s * m_s + n_bh * m_bh,
            rtol=reltol_nd,
            atol=abstol_m,
        )

    @pytest.mark.parametrize("particle_type", present_particle_types.values())
    def test_masks_compatible(self, sg_caesar_galaxy, particle_type):
        """
        Check that the bound_only default mask works with the spatial mask,
        giving the expected shapes for arrays.
        """
        expected_dm = (
            0 if sg_caesar_galaxy.halo_finder.group_type == "galaxy" else 10000
        )
        assert (
            getattr(sg_caesar_galaxy, particle_type).masses.size
            == dict(gas=10000, dark_matter=expected_dm, stars=10000, black_holes=1)[
                particle_type
            ]
        )
