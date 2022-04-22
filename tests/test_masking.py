import pytest
import numpy as np
import unyt as u
from toysnap import present_particle_types
from swiftgalaxy import MaskCollection

abstol_c = 1 * u.pc  # less than this is ~0
abstol_v = 10 * u.m / u.s  # less than this is ~0
abstol_a = 1.e-4 * u.rad


class TestSortingMasks:

    @pytest.mark.parametrize("particle_name", present_particle_types.values())
    @pytest.mark.parametrize("before_load", (True, False))
    def test_reordering_slice(self, sg, particle_name, before_load):
        mask = np.s_[::-1]
        ids_before = getattr(sg, particle_name).particle_ids
        if before_load:
            getattr(sg, particle_name)._particle_dataset._particle_ids = None
        sg.mask_particles(MaskCollection(**{particle_name: mask}))
        ids = getattr(sg, particle_name).particle_ids
        assert u.array.allclose_units(
            ids_before[mask],
            ids,
            rtol=0,
            atol=0
        )
