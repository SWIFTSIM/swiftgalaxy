import numpy as np
from copy import copy, deepcopy
import pytest
import unyt as u
from unyt.testing import assert_allclose_units
from swiftgalaxy.masks import MaskCollection

abstol_m = 1e2 * u.solMass
reltol_m = 1.0e-4
abstol_nd = 1.0e-4
reltol_nd = 1.0e-4


class TestCopySWIFTGalaxy:
    def test_copy_sg(self, sg):
        """
        Test that dataset arrays don't get copied on shallow copy.
        """
        # lazy load a dataset and a named column
        sg.gas.masses
        sg.gas.hydrogen_ionization_fractions.neutral
        sg_copy = copy(sg)
        # check private attribute to not trigger lazy loading
        assert sg_copy.gas._particle_dataset._masses is None
        assert (
            sg_copy.gas.hydrogen_ionization_fractions._named_column_dataset._neutral
            is None
        )

    @pytest.mark.parametrize("derived_coords_initialized", [True, False])
    def test_deepcopy_sg(self, sg, derived_coords_initialized):
        """
        Test that dataset arrays get copied on deep copy.
        """
        # lazy load a dataset and a named column
        sg.gas.masses
        sg.gas.hydrogen_ionization_fractions.neutral
        if derived_coords_initialized:
            sg.gas.spherical_coordinates
            sg.gas.spherical_velocities
            sg.gas.cylindrical_coordinates
            sg.gas.cylindrical_velocities
        sg_copy = deepcopy(sg)
        # check private attribute to not trigger lazy loading
        assert_allclose_units(
            sg.gas.masses,
            sg_copy.gas._particle_dataset._masses,
            rtol=reltol_m,
            atol=abstol_m,
        )
        assert_allclose_units(
            sg.gas.hydrogen_ionization_fractions.neutral,
            sg_copy.gas.hydrogen_ionization_fractions._named_column_dataset._neutral,
            rtol=reltol_nd,
            atol=abstol_nd,
        )
        if derived_coords_initialized:
            assert_allclose_units(
                sg.gas.spherical_coordinates.r,
                sg_copy.gas._spherical_coordinates["_r"],
            )
            assert_allclose_units(
                sg.gas.spherical_velocities.r,
                sg_copy.gas._spherical_velocities["_v_r"],
            )
            assert_allclose_units(
                sg.gas.cylindrical_coordinates.rho,
                sg_copy.gas._cylindrical_coordinates["_rho"],
            )
            assert_allclose_units(
                sg.gas.cylindrical_velocities.rho,
                sg_copy.gas._cylindrical_velocities["_v_rho"],
            )
        else:
            assert sg_copy.gas._spherical_coordinates is None
            assert sg_copy.gas._spherical_velocities is None
            assert sg_copy.gas._cylindrical_coordinates is None
            assert sg_copy.gas._cylindrical_velocities is None


class TestCopyDataset:
    def test_copy_dataset(self, sg):
        """
        Test that arrays don't get copied on shallow copy.
        """
        # lazy load a dataset and a named column
        sg.gas.masses
        sg.gas.hydrogen_ionization_fractions.neutral
        ds_copy = copy(sg.gas)
        # check private attribute to not trigger lazy loading
        assert ds_copy._particle_dataset._masses is None
        assert (
            ds_copy.hydrogen_ionization_fractions._named_column_dataset._neutral is None
        )

    def test_deepcopy_dataset(self, sg):
        """
        Test that arrays get copied on deep copy.
        """
        # lazy load a dataset and a named column
        sg.gas.masses
        sg.gas.hydrogen_ionization_fractions.neutral
        ds_copy = deepcopy(sg.gas)
        # check private attribute to not trigger lazy loading
        assert_allclose_units(
            sg.gas.masses,
            ds_copy._particle_dataset._masses,
            rtol=reltol_m,
            atol=abstol_m,
        )
        assert_allclose_units(
            sg.gas.hydrogen_ionization_fractions.neutral,
            ds_copy.hydrogen_ionization_fractions._named_column_dataset._neutral,
            rtol=reltol_nd,
            atol=abstol_nd,
        )


class TestCopyNamedColumns:
    def test_copy_namedcolumn(self, sg):
        """
        Test that columns don't get copied on shallow copy.
        """
        # lazy load a named column
        sg.gas.hydrogen_ionization_fractions.neutral
        nc_copy = copy(sg.gas.hydrogen_ionization_fractions)
        # check private attribute to not trigger lazy loading
        assert nc_copy._named_column_dataset._neutral is None

    def test_deepcopy_namedcolumn(self, sg):
        """
        Test that columns get copied on deep copy.
        """
        # lazy load a named column
        sg.gas.hydrogen_ionization_fractions.neutral
        nc_copy = deepcopy(sg.gas.hydrogen_ionization_fractions)
        # check private attribute to not trigger lazy loading
        assert_allclose_units(
            sg.gas.hydrogen_ionization_fractions.neutral,
            nc_copy._named_column_dataset._neutral,
            rtol=reltol_nd,
            atol=abstol_nd,
        )


class TestCopyMaskCollection:
    def test_copy_mask_collection(self):
        """
        Test that masks get copied.
        """
        mc = MaskCollection(
            gas=np.ones(100, dtype=bool),
            dark_matter=np.s_[:20],
            stars=None,
            black_holes=np.arange(3),
        )
        mc_copy = copy(mc)
        assert set(mc_copy.__dict__.keys()) == set(mc.__dict__.keys())
        for k in ("gas", "dark_matter", "stars", "black_holes"):
            comparison = getattr(mc, k) == getattr(mc_copy, k)
            if type(comparison) is bool:
                assert comparison
            else:
                assert all(comparison)

    def test_deepcopy_mask_collection(self):
        """
        Test that masks get copied along with values. Since the object isn't
        really "deep", shallow copy and deepcopy have the same expectation.
        """
        mc = MaskCollection(
            gas=np.ones(100, dtype=bool),
            dark_matter=np.s_[:20],
            stars=None,
            black_holes=np.arange(3),
        )
        mc_copy = deepcopy(mc)
        assert set(mc_copy.__dict__.keys()) == set(mc.__dict__.keys())
        for k in ("gas", "dark_matter", "stars", "black_holes"):
            comparison = getattr(mc, k) == getattr(mc_copy, k)
            if type(comparison) is bool:
                assert comparison
            else:
                assert all(comparison)
