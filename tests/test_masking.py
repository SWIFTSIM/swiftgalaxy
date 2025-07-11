"""
Tests for applying masks to swiftgalaxy, datasets and named columns.
"""

import pytest
import numpy as np
from unyt.testing import assert_allclose_units
from swiftgalaxy import SWIFTGalaxy, MaskCollection
from swiftgalaxy.demo_data import (
    ToyHF,
    _create_toysnap,
    _remove_toysnap,
    _toysnap_filename,
    _present_particle_types,
    _n_g_all,
    _n_dm_all,
    _n_s_all,
    _n_bh_all,
)

abstol_nd = 1.0e-4
reltol_nd = 1.0e-4


class TestMaskingSWIFTGalaxy:
    @pytest.mark.parametrize("particle_name", _present_particle_types.values())
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

    @pytest.mark.parametrize("particle_name", _present_particle_types.values())
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

    @pytest.mark.parametrize("particle_name", _present_particle_types.values())
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

    def test_mask_without_spatial_mask(self):
        """
        Check that if we have no masks we read everything in the box (and warn about it).
        Then that we can still apply an extra mask, and a second one (there's specific
        logic for applying two consecutively).
        """
        try:
            _create_toysnap()
            sg = SWIFTGalaxy(
                _toysnap_filename,
                None,  # no halo_catalogue is easiest way to get no mask
                transforms_like_coordinates={"coordinates", "extra_coordinates"},
                transforms_like_velocities={"velocities", "extra_velocities"},
            )
            # check that extra mask is blank for all particle types:
            assert sg._extra_mask.gas is None
            assert sg._extra_mask.dark_matter is None
            assert sg._extra_mask.stars is None
            assert sg._extra_mask.black_holes is None
            # check that cell mask is blank for all particle types:
            assert sg._spatial_mask is None
            # check that we read all the particles:
            assert sg.gas.masses.size == _n_g_all
            assert sg.dark_matter.masses.size == _n_dm_all
            assert sg.stars.masses.size == _n_s_all
            assert sg.black_holes.masses.size == _n_bh_all
            # now apply an extra mask
            sg.mask_particles(MaskCollection(gas=np.s_[:1000]))
            assert sg.gas.masses.size == 1000
            # and the second consecutive one
            sg.mask_particles(MaskCollection(gas=np.s_[:100]))
            assert sg.gas.masses.size == 100
        finally:
            _remove_toysnap()


class TestMaskingParticleDatasets:
    @pytest.mark.parametrize("particle_name", _present_particle_types.values())
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

    @pytest.mark.parametrize("particle_name", _present_particle_types.values())
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

    @pytest.mark.parametrize("particle_name", _present_particle_types.values())
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


class TestMultiModeMask:
    """
    Tests particular to masks when handling a halo catalogue with multiple galaxies
    selected.
    """

    def test_mask_multi_invalid(self, sg):
        """
        When a multi-galaxy halo catalogue object is not masked down to a single object,
        attempting to request a mask for a single object should raise.
        """
        cat = ToyHF(index=[0, 1])
        assert cat._multi_galaxy  # confirm this is multi-galaxy mode
        with pytest.raises(
            RuntimeError,
            match="Halo catalogue has multiple galaxies and is not currently masked.",
        ):
            cat._get_extra_mask(sg)

    def test_caesar_mask_catalogue(self, caesar_multi):
        """
        If a caesar catalogue is in multi-galaxy mode and is not currently masked then
        trying to use the helper to select a single row when the catalogue isn't currently
        restricted to a single galaxy should raise.
        """
        with pytest.raises(
            RuntimeError, match="Tried to mask catalogue without mask index!"
        ):
            caesar_multi._mask_catalogue()
