from copy import deepcopy
import unyt as u
from unyt.testing import assert_allclose_units

abstol_m = 1e2 * u.solMass
reltol_m = 1.0e-4
abstol_nd = 1.0e-4
reltol_nd = 1.0e-4


class TestCopySWIFTGalaxy:
    def test_deepcopy_sg(self, sg):
        """
        Test that datasets get copied on deep copy.
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
