import pytest
import numpy as np
from unyt.testing import assert_allclose_units
from toysnap import present_particle_types
from swiftgalaxy import MaskCollection

abstol_nd = 1.0e-4
reltol_nd = 1.0e-4


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
        assert_allclose_units(ids_before[mask], ids, rtol=0, atol=0)

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
        mask = mask[: mask.size // 2]
        if before_load:
            getattr(sg, particle_name)._particle_dataset._particle_ids = None
        sg.mask_particles(MaskCollection(**{particle_name: mask}))
        ids = getattr(sg, particle_name).particle_ids
        assert_allclose_units(ids_before[mask], ids, rtol=0, atol=0)

    @pytest.mark.parametrize("particle_name", present_particle_types.values())
    @pytest.mark.parametrize("before_load", (True, False))
    def test_bool_mask(self, sg, particle_name, before_load):
        """
        Test whether a boolean array mask works.
        """
        ids_before = getattr(sg, particle_name).particle_ids
        # randomly keep about half of particles
        mask = np.random.rand(ids_before.size) > 0.5
        if before_load:
            getattr(sg, particle_name)._particle_dataset._particle_ids = None
        sg.mask_particles(MaskCollection(**{particle_name: mask}))
        ids = getattr(sg, particle_name).particle_ids
        assert_allclose_units(ids_before[mask], ids, rtol=0, atol=0)

    @pytest.mark.parametrize("before_load", (True, False))
    def test_namedcolumn_masked(self, sg, before_load):
        """
        Test that named columns get masked too.
        """
        neutral_before = sg.gas.hydrogen_ionization_fractions.neutral
        mask = np.random.rand(neutral_before.size) > 0.5
        if before_load:
            sg.gas.hydrogen_ionization_fractions._named_column_dataset._neutral = None
        sg.mask_particles(MaskCollection(**{"gas": mask}))
        neutral = sg.gas.hydrogen_ionization_fractions.neutral
        assert_allclose_units(
            neutral_before[mask], neutral, rtol=reltol_nd, atol=abstol_nd
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
        assert_allclose_units(ids_before[mask], ids, rtol=0, atol=0)

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
        mask = mask[: mask.size // 2]
        if before_load:
            getattr(sg, particle_name)._particle_dataset._particle_ids = None
        masked_dataset = getattr(sg, particle_name)[mask]
        ids = masked_dataset.particle_ids
        assert_allclose_units(ids_before[mask], ids, rtol=0, atol=0)

    @pytest.mark.parametrize("particle_name", present_particle_types.values())
    @pytest.mark.parametrize("before_load", (True, False))
    def test_bool_mask(self, sg, particle_name, before_load):
        """
        Test whether a boolean array mask works.
        """
        ids_before = getattr(sg, particle_name).particle_ids
        # randomly keep about half of particles
        mask = np.random.rand(ids_before.size) > 0.5
        if before_load:
            getattr(sg, particle_name)._particle_dataset._particle_ids = None
        masked_dataset = getattr(sg, particle_name)[mask]
        ids = masked_dataset.particle_ids
        assert_allclose_units(ids_before[mask], ids, rtol=0, atol=0)


class TestMaskingNamedColumnDatasets:
    @pytest.mark.parametrize("before_load", (True, False))
    def test_reordering_slice_mask(self, sg, before_load):
        """
        Test whether a slice mask that re-orders elements works.
        """
        mask = np.s_[::-1]
        fractions_before = sg.gas.hydrogen_ionization_fractions.neutral
        if before_load:
            sg.gas.hydrogen_ionization_fractions._neutral = None
        masked_namedcolumnsdataset = sg.gas.hydrogen_ionization_fractions[mask]
        fractions = masked_namedcolumnsdataset.neutral
        assert_allclose_units(
            fractions_before[mask], fractions, rtol=reltol_nd, atol=abstol_nd
        )

    @pytest.mark.parametrize("before_load", (True, False))
    def test_reordering_int_mask(self, sg, before_load):
        """
        Test whether an integer array mask that re-orders elements and changes
        the array length works.
        """
        fractions_before = sg.gas.hydrogen_ionization_fractions.neutral
        mask = np.arange(fractions_before.size)
        # randomize order (in-place operation)
        np.random.shuffle(mask)
        # keep half the particles
        mask = mask[: mask.size // 2]
        if before_load:
            sg.gas.hydrogen_ionization_fractions._neutral = None
        masked_namedcolumnsdataset = sg.gas.hydrogen_ionization_fractions[mask]
        fractions = masked_namedcolumnsdataset.neutral
        assert_allclose_units(
            fractions_before[mask], fractions, rtol=reltol_nd, atol=abstol_nd
        )

    @pytest.mark.parametrize("before_load", (True, False))
    def test_bool_mask(self, sg, before_load):
        """
        Test whether a boolean array mask works.
        """
        fractions_before = sg.gas.hydrogen_ionization_fractions.neutral
        # randomly keep about half of particles
        mask = np.random.rand(fractions_before.size) > 0.5
        if before_load:
            sg.gas.hydrogen_ionization_fractions._neutral = None
        masked_namedcolumnsdataset = sg.gas.hydrogen_ionization_fractions[mask]
        fractions = masked_namedcolumnsdataset.neutral
        assert_allclose_units(
            fractions_before[mask], fractions, rtol=reltol_nd, atol=abstol_nd
        )
