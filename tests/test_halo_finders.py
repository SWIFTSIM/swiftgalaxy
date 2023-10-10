import pytest
import h5py
import numpy as np
import unyt as u
from unyt.testing import assert_allclose_units
from toysnap import (
    toysnap_filename,
    n_g,
    n_g_all,
    n_dm,
    # n_dm_all,
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
from swiftgalaxy import MaskCollection, SWIFTGalaxy
from swiftsimio.objects import cosmo_array

abstol_c = 1 * u.pc  # less than this is ~0
abstol_v = 10 * u.m / u.s  # less than this is ~0
abstol_m = 1e4 * u.Msun  # less than this is ~0
reltol_nd = 1.0e-4


class TestHaloFinders:
    def test_get_spatial_mask(self, hf):
        """
        Check that we get spatial masks that we expect.
        """
        # don't use sg fixture here, just need the snapshot file
        # so don't want overhead of a SWIFTGalaxy
        create_toysnap()
        spatial_mask = hf._get_spatial_mask(toysnap_filename)
        snap = h5py.File(toysnap_filename, "r")
        n_g_firstcell = snap["/Cells/Counts/PartType0"][0]
        n_dm_firstcell = snap["/Cells/Counts/PartType1"][0]
        n_s_firstcell = snap["/Cells/Counts/PartType4"][0]
        n_bh_firstcell = snap["/Cells/Counts/PartType5"][0]
        remove_toysnap()
        # we have 2 cells, covering 0-5 and 5-10 Mpc in x, and the entire range in y and z
        # the galaxy is at (2,2,2)Mpc, so in the first cell
        assert np.array_equal(spatial_mask.gas, np.array([[0, n_g_firstcell]]))
        assert np.array_equal(spatial_mask.dark_matter, np.array([[0, n_dm_firstcell]]))
        assert np.array_equal(spatial_mask.stars, np.array([[0, n_s_firstcell]]))
        assert np.array_equal(spatial_mask.black_holes, np.array([[0, n_bh_firstcell]]))

    def test_get_bound_only_extra_mask(self, hf):
        """
        Check that bound_only extra mask has the right shape.
        """
        hf.extra_mask = "bound_only"
        create_toysnap()
        sg = SWIFTGalaxy(toysnap_filename, hf)
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
                    == dict(gas=n_g, dark_matter=n_dm, stars=n_s, black_holes=n_bh)[
                        particle_type
                    ]
                )
        remove_toysnap()

    def test_get_void_extra_mask(self, hf):
        """
        Check that None extra mask gives expected result.
        """
        hf.extra_mask = None
        create_toysnap()
        sg = SWIFTGalaxy(toysnap_filename, hf)
        generated_extra_mask = sg._extra_mask
        for particle_type in present_particle_types.values():
            assert getattr(generated_extra_mask, particle_type) is None
        remove_toysnap()

    # @pytest.mark.parametrize(
    #     "extra_mask, expected",
    #     (
    #         (
    #             "bound_only",
    #             dict(gas=n_g, dark_matter=n_dm, stars=n_s, black_holes=n_bh),
    #         ),
    #         (None, dict(gas=None, dark_matter=None, stars=None, black_holes=None)),
    #         (
    #             MaskCollection(
    #                 gas=np.r_[
    #                     np.ones(100, dtype=bool), np.zeros(n_g_all - 100, dtype=bool)
    #                 ],
    #                 dark_matter=None,
    #                 stars=np.r_[
    #                     np.ones(100, dtype=bool), np.zeros(n_s - 100, dtype=bool)
    #                 ],
    #                 black_holes=np.ones(n_bh, dtype=bool),
    #             ),
    #             dict(gas=100, dark_matter=None, stars=100, black_holes=n_bh),
    #         ),
    #     ),
    # )
    # def test_get_user_extra_mask(self, sg, hf, extra_mask, expected):
    #     """
    #     Check that extra masks of different kinds have the right shape or type.
    #     """
    #     if hasattr(hf, "_caesar"):
    #         if hf.group_type == "galaxy" and extra_mask == "bound_only":
    #             expected["dark_matter"] = 0
    #     snap = h5py.File(sg.filename, "r")
    #     for k in expected.keys():
    #         if hasattr(expected[k], "shape"):
    #             expected[k] = expected[k][
    #                 : snap[
    #                     "Cells/Counts/"
    #                     "PartType{dict(gas=0, dark_matter=1, stars=4, black_holes=5)[k]}"
    #                 ][0]
    #             ]
    #     hf.extra_mask = extra_mask
    #     generated_extra_mask = hf._get_extra_mask(sg)
    #     print(generated_extra_mask.gas.shape)
    #     for particle_type in present_particle_types.values():
    #         if getattr(generated_extra_mask, particle_type) is not None:
    #             assert (
    #                 getattr(generated_extra_mask, particle_type).sum()
    #                 == expected[particle_type]
    #             )
    #         else:
    #             assert expected[particle_type] is None

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
                n_g * m_g + n_s * m_s + n_bh * m_bh,
                rtol=reltol_nd,
                atol=abstol_m,
            )
        else:
            raise AttributeError

    def test_spatial_mask_applied(self):
        """
        Until this issue is resolved:

        https://github.com/dnarayanan/caesar/issues/92

        Caesar catalogues don't contain enough information to construct a spatial mask.
        For now we just read the whole box (!), and expect to fail this test.
        """
        raise NotImplementedError


class TestCaesarWithSWIFTGalaxy:
    """
    Most interaction between the halo finder and swiftgalaxy.reader.SWIFTGalaxy
    is tested using the toysnap.ToyHF testing class (that inherits from
    swiftgalaxy.halo_finders._HaloFinder). Here we just want to test anything
    caesar-specific.
    """

    def test_catalogue_exposed(self, sg_caesar):
        """
        Check that exposing the halo properties is working, through the
        SWIFTGalaxy object.
        """
        if hasattr(sg_caesar.halo_finder, "virial_quantities"):
            assert_allclose_units(
                sg_caesar.halo_finder.virial_quantities["m200c"],
                1.0e12 * u.Msun,
                rtol=reltol_nd,
                atol=abstol_m,
            )
        elif hasattr(sg_caesar.halo_finder, "masses"):
            assert_allclose_units(
                sg_caesar.halo_finder.masses["total"],
                n_g * m_g + n_s * m_s + n_bh * m_bh,
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
        expected_dm = 0 if sg_caesar.halo_finder.group_type == "galaxy" else 10000
        assert (
            getattr(sg_caesar, particle_type).masses.size
            == dict(gas=10000, dark_matter=expected_dm, stars=10000, black_holes=1)[
                particle_type
            ]
        )
