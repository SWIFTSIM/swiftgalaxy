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
    n_g_2,
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
    centre_1,
    centre_2,
    vcentre_1,
    present_particle_types,
)
from swiftgalaxy import SWIFTGalaxy, MaskCollection
from swiftgalaxy.halo_catalogues import Velociraptor, Caesar, SOAP, Standalone
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
            cosmo_array([centre_1 + 0.001, centre_1 + 0.001, centre_1 + 0.001], u.Mpc),
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
            cosmo_array(
                [vcentre_1 + 1.0, vcentre_1 + 1.0, vcentre_1 + 1.0], u.km / u.s
            ),
            rtol=reltol_nd,
            atol=abstol_v,
        )


class TestHaloCataloguesMulti:

    def test_multi_flags(self, hf_multi):
        """
        Check that the multi-target nature of the cataloue is recognized.
        """
        assert hf_multi._multi_galaxy
        assert hf_multi._multi_galaxy_catalogue_mask is None
        assert hf_multi._multi_galaxy_index_mask is None
        assert hf_multi._multi_count == 2
        assert hf_multi.count == 2

    def test_mask_multi_galaxy(self, hf_multi):
        """
        Check that we can mask the catalogue to focus on one object, and unmask.
        """
        assert hf_multi._multi_galaxy_catalogue_mask is None
        assert hf_multi._multi_galaxy_index_mask is None
        assert hf_multi.count > 1
        assert hf_multi._region_centre.shape == (hf_multi.count, 3)
        assert hf_multi._region_aperture.shape == (hf_multi.count,)
        assert hf_multi.centre.shape == (hf_multi.count, 3)
        assert hf_multi.velocity_centre.shape == (hf_multi.count, 3)
        mask_index = 0
        hf_multi._mask_multi_galaxy(mask_index)
        if isinstance(hf_multi._multi_galaxy_index_mask, int):
            assert hf_multi._multi_galaxy_index_mask == mask_index
        elif isinstance(hf_multi._multi_galaxy_index_mask, slice):
            assert (
                hf_multi._multi_galaxy_index_mask == np.s_[mask_index : mask_index + 1]
            )
        if hf_multi._index_attr is not None:  # skip for Standalone
            assert (
                hf_multi._multi_galaxy_catalogue_mask
                == np.argsort(np.argsort(getattr(hf_multi, hf_multi._index_attr)))[
                    mask_index
                ]
            )
        assert hf_multi.count == 1
        assert hf_multi._region_centre.shape == (3,)
        if isinstance(hf_multi._multi_galaxy_index_mask, int):
            assert hf_multi.centre.shape == (3,)
            assert hf_multi.velocity_centre.shape == (3,)
        elif isinstance(hf_multi._multi_galaxy_index_mask, slice):
            assert hf_multi.centre.shape == (1, 3)
            assert hf_multi.velocity_centre.shape == (1, 3)
        assert hf_multi._region_aperture.shape == tuple()
        hf_multi._unmask_multi_galaxy()
        assert hf_multi._multi_galaxy_catalogue_mask is None
        assert hf_multi._multi_galaxy_index_mask is None
        assert hf_multi.count > 1
        assert hf_multi._region_centre.shape == (hf_multi.count, 3)
        assert hf_multi._region_aperture.shape == (hf_multi.count,)
        assert hf_multi.centre.shape == (hf_multi.count, 3)
        assert hf_multi.velocity_centre.shape == (hf_multi.count, 3)

    def test_get_spatial_mask(self, hf_multi, toysnap):
        """
        Check that we get spatial masks that we expect.
        """
        with pytest.raises(RuntimeError, match="not currently masked"):
            hf_multi._get_spatial_mask(toysnap_filename)
        hf_multi._mask_multi_galaxy(0)
        spatial_mask = hf_multi._get_spatial_mask(toysnap_filename)
        with h5py.File(toysnap_filename, "r") as snap:
            n_g_firstcell = snap["/Cells/Counts/PartType0"][0]
            n_dm_firstcell = snap["/Cells/Counts/PartType1"][0]
            n_s_firstcell = snap["/Cells/Counts/PartType4"][0]
            n_bh_firstcell = snap["/Cells/Counts/PartType5"][0]
        if hf_multi._user_spatial_offsets is None:
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

    def test_generate_extra_mask(self, hf_multi, toysnap_withfof):
        """
        Check that bound_only extra mask has the right shape.
        """
        hf_multi.extra_mask = "bound_only"
        hf_multi._mask_multi_galaxy(0)
        if hasattr(hf_multi, "soap_file"):
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
            sg = SWIFTGalaxy(toysoap_virtual_snapshot_filename, hf_multi)
        else:
            try:
                sg = SWIFTGalaxy(toysnap_filename, hf_multi)
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
        if hasattr(hf_multi, "_caesar") and hf_multi.group_type == "galaxy":
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

    def test_mask_index_list(self, hf_multi):
        """
        Check that we mask the list of indices.
        """
        if hf_multi._index_attr is None:
            # this is Standalone, nothing to do
            return
        # strip the leading underscore with [1:] to access the property
        assert getattr(hf_multi, hf_multi._index_attr[1:]) == [0, 1]
        hf_multi._mask_multi_galaxy(0)
        assert getattr(hf_multi, hf_multi._index_attr[1:]) == 0

    def test_masked_catalogue_matches(self, hf_multi):
        mask_index = 0
        init_args = {
            "extra_mask": hf_multi.extra_mask,
        }
        if hf_multi.__class__ != Standalone:
            init_args[hf_multi._index_attr[1:]] = getattr(
                hf_multi, hf_multi._index_attr
            )[mask_index]
        else:
            init_args["centre"] = hf_multi._centre
            init_args["velocity_centre"] = hf_multi._velocity_centre
        if hasattr(hf_multi, "centre_type"):
            init_args["centre_type"] = hf_multi.centre_type
        if hasattr(hf_multi, "velocity_centre_type"):
            init_args["velocity_centre_type"] = hf_multi.velocity_centre_type
        if hf_multi.__class__ == Caesar:
            init_args["group_type"] = hf_multi.group_type
        init_args[
            ("custom_" if hf_multi.__class__ != Standalone else "") + "spatial_offsets"
        ] = hf_multi._user_spatial_offsets
        if hf_multi.__class__ == SOAP:
            init_args["soap_file"] = hf_multi.soap_file
        elif hf_multi.__class__ == Velociraptor:
            init_args["velociraptor_files"] = hf_multi.velociraptor_files
        elif hf_multi.__class__ == Caesar:
            init_args["caesar_file"] = hf_multi.caesar_file
        hf = hf_multi.__class__(**init_args)
        hf_multi._mask_multi_galaxy(mask_index)
        if hf_multi.__class__ == SOAP:
            assert (
                hf.bound_subhalo.enclose_radius == hf_multi.bound_subhalo.enclose_radius
            )
            assert (
                hf.bound_subhalo.enclose_radius.shape
                == hf_multi.bound_subhalo.enclose_radius.shape
            )
        elif hf_multi.__class__ == Velociraptor:
            assert hf_multi.masses.mass_200crit == hf.masses.mass_200crit
            assert hf_multi.masses.mass_200crit.shape == hf.masses.mass_200crit.shape
        elif hf_multi.__class__ == Caesar:
            assert hf_multi.masses["total"] == hf.masses["total"]
            assert hf_multi.masses["total"].shape == hf.masses["total"].shape
        elif hf_multi.__class__ == Standalone:
            pass  # has no catalogue to check
        else:
            raise NotImplementedError  # a new class we're not checking


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
            ("", centre_1),
            ("minpot", centre_1 + 0.001),
            ("mbp", centre_1 + 0.002),
            ("_gas", centre_1 + 0.003),
            ("_stars", centre_1 + 0.004),
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
            ("", vcentre_1),
            ("minpot", vcentre_1 + 1.0),
            ("mbp", vcentre_1 + 2.0),
            ("_gas", vcentre_1 + 3.0),
            ("_stars", vcentre_1 + 4.0),
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

    def test_masking_catalogue(self, vr_multi):
        """
        Check that we can access unmasked and masked catalogue properties.
        """
        # pick one of the default attributes to check
        assert_allclose_units(
            vr_multi.masses.mvir,
            [1.0e12 * u.Msun, 1.1e12 * u.Msun],
            rtol=reltol_nd,
            atol=abstol_m,
        )
        vr_multi._mask_multi_galaxy(0)
        self.test_catalogue_exposed(vr_multi)

    def test_dir_for_tab_completion(self, vr):
        """
        Check that we add catalogue properties to the namespace directory.

        Just check a couple, don't need to be exhaustive.
        """
        for prop in ("energies", "metallicity", "temperature"):
            assert prop in dir(vr)


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

    def test_with_swiftgalaxies(self, sgs_vr):
        for sg_from_sgs in sgs_vr:
            sg = SWIFTGalaxy(
                sg_from_sgs.snapshot_filename,
                Velociraptor(
                    velociraptor_files=sg_from_sgs.halo_catalogue.velociraptor_files,
                    halo_index=sg_from_sgs.halo_catalogue.halo_index,
                ),
            )
            for ptype in present_particle_types.values():
                assert np.all(
                    getattr(sg_from_sgs._extra_mask, ptype)
                    == getattr(sg._extra_mask, ptype)
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
            ("", centre_1),
            ("minpot", centre_1 + 0.001),
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
            ("", vcentre_1),
            ("minpot", vcentre_1 + 1.0),
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

    def test_masking_catalogue(self, caesar_multi):
        """
        Check that we can access unmasked and masked catalogue properties.
        """
        # pick one of the default attributes to check
        if hasattr(caesar_multi, "virial_quantities"):
            assert_allclose_units(
                [c["m200c"] for c in caesar_multi.virial_quantities],
                [1.0e12 * u.Msun, 2.0e12 * u.Msun],
                rtol=reltol_nd,
                atol=abstol_m,
            )
        elif hasattr(caesar_multi, "masses"):
            assert_allclose_units(
                [c["total"] for c in caesar_multi.masses],
                [
                    n_g_1 * m_g + n_s_1 * m_s + n_bh_1 * m_bh,
                    n_g_2 * m_g + n_s_2 * m_s + n_bh_2 * m_bh,
                ],
                rtol=reltol_nd,
                atol=abstol_m,
            )
        else:
            raise AttributeError
        caesar_multi._mask_multi_galaxy(0)
        self.test_catalogue_exposed(caesar_multi)

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

    def test_dir_for_tab_completion(self, caesar):
        """
        Check that we add catalogue properties to the namespace directory.

        Just check a couple, don't need to be exhaustive.
        """
        # picked these to be common between halo and galaxy catalogues:
        for prop in ("glist", "pos", "radii"):
            assert prop in dir(caesar)


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

    def test_with_swiftgalaxies(self, sgs_caesar):
        for sg_from_sgs in sgs_caesar:
            sg = SWIFTGalaxy(
                sg_from_sgs.snapshot_filename,
                Caesar(
                    caesar_file=sg_from_sgs.halo_catalogue.caesar_file,
                    group_type=sg_from_sgs.halo_catalogue.group_type,
                    group_index=sg_from_sgs.halo_catalogue.group_index,
                ),
            )
            for ptype in present_particle_types.values():
                if isinstance(getattr(sg_from_sgs._extra_mask, ptype), slice):
                    assert getattr(sg_from_sgs._extra_mask, ptype) == getattr(
                        sg._extra_mask, ptype
                    )
                else:
                    assert np.all(
                        getattr(sg_from_sgs._extra_mask, ptype)
                        == getattr(sg._extra_mask, ptype)
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
        assert soap._catalogue is not None

    @pytest.mark.parametrize(
        "centre_type, expected",
        (
            ("bound_subhalo.centre_of_mass", centre_1 + 0.001),
            ("exclusive_sphere_100kpc.centre_of_mass", centre_1 + 0.003),
            ("inclusive_sphere_100kpc.centre_of_mass", centre_1 + 0.011),
            ("input_halos_fof.centres", centre_1),
            ("input_halos.halo_centre", centre_1 + 0.001),
            ("projected_aperture_50kpc_projx.centre_of_mass", centre_1 + 0.027),
            ("spherical_overdensity_200_crit.centre_of_mass", centre_1 + 0.032),
            ("spherical_overdensity_bn98.centre_of_mass", centre_1 + 0.038),
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
            ("bound_subhalo.centre_of_mass_velocity", vcentre_1 + 1),
            ("exclusive_sphere_100kpc.centre_of_mass_velocity", vcentre_1 + 3),
            ("inclusive_sphere_100kpc.centre_of_mass_velocity", vcentre_1 + 11),
            ("projected_aperture_50kpc_projx.centre_of_mass_velocity", vcentre_1 + 27),
            ("spherical_overdensity_200_crit.centre_of_mass_velocity", vcentre_1 + 32),
            ("spherical_overdensity_bn98.centre_of_mass_velocity", vcentre_1 + 38),
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
            cosmo_array(
                [[centre_1 + 0.001, centre_1 + 0.001, centre_1 + 0.001]],
                u.Mpc,
                comoving=False,
            ),
            rtol=reltol_nd,
            atol=abstol_c,
        )

    def test_masking_catalogue(self, soap_multi):
        """
        Check that we can access unmasked and masked catalogue properties.
        """
        # pick a couple of attributes to check
        assert_allclose_units(
            soap_multi.input_halos_hbtplus.host_fofid,
            cosmo_array([1, 2], comoving=False),
        )
        assert_allclose_units(
            soap_multi.bound_subhalo.centre_of_mass,
            cosmo_array(
                [
                    [centre_1 + 0.001, centre_1 + 0.001, centre_1 + 0.001],
                    [centre_2 + 0.001, centre_2 + 0.001, centre_2 + 0.001],
                ],
                u.Mpc,
                comoving=False,
            ),
            rtol=reltol_nd,
            atol=abstol_c,
        )
        soap_multi._mask_multi_galaxy(0)
        self.test_catalogue_exposed(soap_multi)

    def test_dir_for_tab_completion(self, soap):
        """
        Check that we add catalogue properties to the namespace directory.

        Just check a couple, don't need to be exhaustive.
        """
        for prop in (
            "exclusive_sphere_100kpc",
            "input_halos",
            "spherical_overdensity_bn98",
        ):
            assert prop in dir(soap)


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
            cosmo_array(
                [[centre_1 + 0.001, centre_1 + 0.001, centre_1 + 0.001]],
                u.Mpc,
                comoving=False,
            ),
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

    def test_with_swiftgalaxies(self, sgs_soap):
        for sg_from_sgs in sgs_soap:
            sg = SWIFTGalaxy(
                sg_from_sgs.snapshot_filename,
                SOAP(
                    soap_file=sg_from_sgs.halo_catalogue.soap_file,
                    soap_index=sg_from_sgs.halo_catalogue.soap_index,
                ),
            )
            for ptype in present_particle_types.values():
                assert np.all(
                    getattr(sg_from_sgs._extra_mask, ptype)
                    == getattr(sg._extra_mask, ptype)
                )
