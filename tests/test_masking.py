import pytest
import numpy as np
import unyt as u
from toysnap import present_particle_types
from swiftgalaxy import MaskCollection

abstol_c = 1 * u.pc  # less than this is ~0
abstol_v = 10 * u.m / u.s  # less than this is ~0
abstol_a = 1.e-4 * u.rad


class TestMaskingSWIFTGalaxy:

    @pytest.mark.parametrize("particle_name", present_particle_types.values())
    @pytest.mark.parametrize("before_load", (True, False))
    def test_reordering_slice_mask(self, sg, particle_name, before_load):
        """
        Test whether a slice mask that re-orders elements works.
        """
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

    @pytest.mark.parametrize("particle_name", present_particle_types.values())
    @pytest.mark.parametrize("before_load", (True, False))
    def test_reordering_int_mask(self, sg, particle_name, before_load):
        """
        Test whether an integer array mask that re-orders elements and changes
        the array length works.
        """
        ids_before = getattr(sg, particle_name).particle_ids
        mask = np.arange(ids_before.size)
        # randomize order (in-place operation)
        np.random.shuffle(mask)
        # keep half the particles
        mask = mask[:mask.size // 2]
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

    @pytest.mark.parametrize("particle_name", present_particle_types.values())
    @pytest.mark.parametrize("before_load", (True, False))
    def test_bool_mask(self, sg, particle_name, before_load):
        """
        Test whether a boolean array mask works.
        """
        ids_before = getattr(sg, particle_name).particle_ids
        # randomly keep about half of particles
        mask = np.random.rand(ids_before.size) > .5
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


class TestMaskingParticleDatasets:

    @pytest.mark.parametrize("particle_name", present_particle_types.values())
    @pytest.mark.parametrize("before_load", (True, False))
    def test_reordering_slice_mask(self, sg, particle_name, before_load):
        """
        Test whether a slice mask that re-orders elements works.
        """
        mask = np.s_[::-1]
        ids_before = getattr(sg, particle_name).particle_ids
        if before_load:
            getattr(sg, particle_name)._particle_dataset._particle_ids = None
        masked_dataset = getattr(sg, particle_name)[mask]
        ids = masked_dataset.particle_ids
        assert u.array.allclose_units(
            ids_before[mask],
            ids,
            rtol=0,
            atol=0
        )

    @pytest.mark.parametrize("particle_name", present_particle_types.values())
    @pytest.mark.parametrize("before_load", (True, False))
    def test_reordering_int_mask(self, sg, particle_name, before_load):
        """
        Test whether an integer array mask that re-orders elements and changes
        the array length works.
        """
        ids_before = getattr(sg, particle_name).particle_ids
        mask = np.arange(ids_before.size)
        # randomize order (in-place operation)
        np.random.shuffle(mask)
        # keep half the particles
        mask = mask[:mask.size // 2]
        if before_load:
            getattr(sg, particle_name)._particle_dataset._particle_ids = None
        masked_dataset = getattr(sg, particle_name)[mask]
        ids = masked_dataset.particle_ids
        assert u.array.allclose_units(
            ids_before[mask],
            ids,
            rtol=0,
            atol=0
        )

    @pytest.mark.parametrize("particle_name", present_particle_types.values())
    @pytest.mark.parametrize("before_load", (True, False))
    def test_bool_mask(self, sg, particle_name, before_load):
        """
        Test whether a boolean array mask works.
        """
        ids_before = getattr(sg, particle_name).particle_ids
        # randomly keep about half of particles
        mask = np.random.rand(ids_before.size) > .5
        if before_load:
            getattr(sg, particle_name)._particle_dataset._particle_ids = None
        masked_dataset = getattr(sg, particle_name)[mask]
        ids = masked_dataset.particle_ids
        assert u.array.allclose_units(
            ids_before[mask],
            ids,
            rtol=0,
            atol=0
        )


class TestMaskingNamedColumnDatasets:

    pass
