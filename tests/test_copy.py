from copy import copy, deepcopy
import unyt as u
from unyt.testing import assert_allclose_units

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

    def test_deepcopy_sg(self, sg):
        """
        Test that dataset arrays get copied on deep copy.
        """
        # lazy load a dataset and a named column
        sg.gas.masses
        sg.gas.hydrogen_ionization_fractions.neutral
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
