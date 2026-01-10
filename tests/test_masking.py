"""Tests for applying masks to swiftgalaxy, datasets and named columns."""

import pytest
from copy import copy, deepcopy
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
from swiftgalaxy.masks import LazyMask

abstol_nd = 1.0e-4
reltol_nd = 1.0e-4


class TestMaskingSWIFTGalaxy:
    @pytest.mark.parametrize("particle_name", _present_particle_types.values())
    def test_getattr_masking(self, sg, particle_name):
        """Test that we can mask with square bracket notation. Use an order reversing mask."""
        getattr(sg, particle_name).particle_ids
        mask = np.s_[::-1]
        new_sg = sg[MaskCollection(**{particle_name: mask})]
        assert (
            getattr(new_sg, particle_name).particle_ids[::-1]
            == getattr(sg, particle_name).particle_ids
        ).all()

    @pytest.mark.parametrize("particle_name", _present_particle_types.values())
    @pytest.mark.parametrize("before_load", (True, False))
    def test_reordering_slice_mask(self, sg, particle_name, before_load):
        """Test whether a slice mask that re-orders elements works."""
        mask = np.s_[::-1]
        ids_before = getattr(sg, particle_name).particle_ids
        if before_load:
            getattr(sg, particle_name)._particle_dataset._particle_ids = None
            del getattr(sg._extra_mask, particle_name)._mask
            getattr(sg._extra_mask, particle_name)._evaluated = False
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
            del getattr(sg._extra_mask, particle_name)._mask
            getattr(sg._extra_mask, particle_name)._evaluated = False
        sg.mask_particles(MaskCollection(**{particle_name: mask}))
        ids = getattr(sg, particle_name).particle_ids
        assert_allclose_units(ids_before[mask], ids, rtol=0, atol=0)

    @pytest.mark.parametrize("particle_name", _present_particle_types.values())
    @pytest.mark.parametrize("before_load", (True, False))
    def test_bool_mask(self, sg, particle_name, before_load):
        """Test whether a boolean array mask works."""
        ids_before = getattr(sg, particle_name).particle_ids
        # randomly keep about half of particles
        mask = np.random.rand(ids_before.size) > 0.5
        if before_load:
            getattr(sg, particle_name)._particle_dataset._particle_ids = None
            del getattr(sg._extra_mask, particle_name)._mask
            getattr(sg._extra_mask, particle_name)._evaluated = False
        sg.mask_particles(MaskCollection(**{particle_name: mask}))
        ids = getattr(sg, particle_name).particle_ids
        assert_allclose_units(ids_before[mask], ids, rtol=0, atol=0)

    @pytest.mark.parametrize("before_load", (True, False))
    def test_namedcolumn_masked(self, sg, before_load):
        """Test that named columns get masked too."""
        neutral_before = sg.gas.hydrogen_ionization_fractions.neutral
        mask = np.random.rand(neutral_before.size) > 0.5
        if before_load:
            sg.gas.hydrogen_ionization_fractions._named_column_dataset._neutral = None
            del sg._extra_mask.gas._mask
            sg._extra_mask.gas._evaluated = False
        sg.mask_particles(MaskCollection(**{"gas": mask}))
        neutral = sg.gas.hydrogen_ionization_fractions.neutral
        assert_allclose_units(
            neutral_before[mask], neutral, rtol=reltol_nd, atol=abstol_nd
        )

    def test_mask_without_spatial_mask(self, tmp_path_factory):
        """
        Check that if we have no masks we read everything in the box (and warn about it).
        Then that we can still apply an extra mask, and a second one (there's specific
        logic for applying two consecutively).
        """
        toysnap_filename = (
            tmp_path_factory.mktemp(_toysnap_filename.parent) / _toysnap_filename.name
        )
        try:
            _create_toysnap(snapfile=toysnap_filename)
            sg = SWIFTGalaxy(
                toysnap_filename,
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
            _remove_toysnap(snapfile=toysnap_filename)


class TestMaskingParticleDatasets:
    @pytest.mark.parametrize("particle_name", _present_particle_types.values())
    @pytest.mark.parametrize("before_load", (True, False))
    def test_reordering_slice_mask(self, sg, particle_name, before_load):
        """Test whether a slice mask that re-orders elements works."""
        mask = np.s_[::-1]
        ids_before = getattr(sg, particle_name).particle_ids
        if before_load:
            getattr(sg, particle_name)._particle_dataset._particle_ids = None
            del getattr(sg._extra_mask, particle_name)._mask
            getattr(sg._extra_mask, particle_name)._evaluated = False
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
            del getattr(sg._extra_mask, particle_name)._mask
            getattr(sg._extra_mask, particle_name)._evaluated = False
        masked_dataset = getattr(sg, particle_name)[mask]
        ids = masked_dataset.particle_ids
        assert_allclose_units(ids_before[mask], ids, rtol=0, atol=0)

    @pytest.mark.parametrize("particle_name", _present_particle_types.values())
    @pytest.mark.parametrize("before_load", (True, False))
    def test_bool_mask(self, sg, particle_name, before_load):
        """Test whether a boolean array mask works."""
        ids_before = getattr(sg, particle_name).particle_ids
        # randomly keep about half of particles
        mask = np.random.rand(ids_before.size) > 0.5
        if before_load:
            getattr(sg, particle_name)._particle_dataset._particle_ids = None
            del getattr(sg._extra_mask, particle_name)._mask
            getattr(sg._extra_mask, particle_name)._evaluated = False
        masked_dataset = getattr(sg, particle_name)[mask]
        ids = masked_dataset.particle_ids
        assert_allclose_units(ids_before[mask], ids, rtol=0, atol=0)


class TestMaskingNamedColumnDatasets:
    @pytest.mark.parametrize("before_load", (True, False))
    def test_reordering_slice_mask(self, sg, before_load):
        """Test whether a slice mask that re-orders elements works."""
        mask = np.s_[::-1]
        fractions_before = sg.gas.hydrogen_ionization_fractions.neutral
        if before_load:
            sg.gas.hydrogen_ionization_fractions._neutral = None
            del sg._extra_mask.gas._mask
            sg._extra_mask.gas._evaluated = False
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
            del sg._extra_mask.gas._mask
            sg._extra_mask.gas._mask = False
        masked_namedcolumnsdataset = sg.gas.hydrogen_ionization_fractions[mask]
        fractions = masked_namedcolumnsdataset.neutral
        assert_allclose_units(
            fractions_before[mask], fractions, rtol=reltol_nd, atol=abstol_nd
        )

    @pytest.mark.parametrize("before_load", (True, False))
    def test_bool_mask(self, sg, before_load):
        """Test whether a boolean array mask works."""
        fractions_before = sg.gas.hydrogen_ionization_fractions.neutral
        # randomly keep about half of particles
        mask = np.random.rand(fractions_before.size) > 0.5
        if before_load:
            sg.gas.hydrogen_ionization_fractions._neutral = None
            del sg._extra_mask.gas._mask
            sg._extra_mask.gas._evaluated = False
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


class TestLazyMask:
    """Unit tests for the LazyMask class itself."""

    def test_init_lazy(self, lm):
        """Check that initializing in lazy mode works."""
        assert lm._evaluated is False
        assert lm._mask_function is not None  # also implies it exists
        assert not hasattr(lm, "_mask")

    def test_init_concrete(self):
        """Check that initializing with a concrete mask works."""
        m = np.ones(10, dtype=bool)
        lm = LazyMask(mask=m)
        assert lm._evaluated
        assert lm._mask is m
        assert lm._mask_function is None

    def test_init_none(self):
        """Check that initializing with ``None`` works (i.e. no mask)."""
        lm = LazyMask(mask=None)
        assert lm._evaluated
        assert lm._mask is None
        assert not hasattr(lm, "_mask_function")

    def test_trigger_eval(self, lm):
        """Check that accessing mask triggers evaluation if lazy."""
        assert lm._evaluated is False
        assert not hasattr(lm, "_mask")
        assert (lm.mask == lm._mask_function()).all()
        assert lm._evaluated
        assert (lm._mask == lm._mask_function()).all()

    def test_access_not_lazy(self):
        """Check that accessing the mask works for a non-lazy mask."""
        m = np.ones(10, dtype=bool)
        lm = LazyMask(mask=m)
        assert lm._evaluated
        assert (lm.mask == m).all()

    def test_manual_trigger_eval(self, lm):
        """Check that accessing mask triggers evaluation if lazy."""
        assert lm._evaluated is False
        assert not hasattr(lm, "_mask")
        lm._evaluate()
        assert lm._evaluated
        assert (lm.mask == lm._mask_function()).all()
        assert (lm._mask == lm._mask_function()).all()

    def test_trigger_eval_once_only(self):
        """Check that we can't trigger mask evaluation repeatedly."""

        class MF(object):
            """
            A simple class that behaves as a mask generator and counts how many times its
            ``__call__`` is called.
            """

            call_counter: int = 0

            def __call__(self):
                """
                Call the class to behave like a simple mask function.

                Returns
                -------
                out : ndarray
                    A simple mask array.
                """
                self.call_counter += 1
                return np.ones(10, dtype=bool)

        mf = MF()
        lm = LazyMask(mask_function=mf)
        assert lm._evaluated is False
        # trigger a mask evaluation:
        lm.mask
        assert lm._evaluated is True
        assert mf.call_counter == 1
        # we shouldn't be able to trigger or force another mask evaluation:
        lm.mask
        lm._evaluate()
        assert mf.call_counter == 1

    def test_copy(self, lm):
        """Check copying behaviour of ``LazyMask`` objects."""
        # first copy before evaluating
        lm_unevaluated_copy = copy(lm)
        assert lm_unevaluated_copy._evaluated is False
        assert not hasattr(lm_unevaluated_copy, "_mask")
        assert lm_unevaluated_copy._mask_function is lm._mask_function
        # trigger evaluated
        lm._evaluate()
        assert lm._evaluated
        # now copy after evaluating
        lm_evaluated_copy = copy(lm)
        assert lm_evaluated_copy._evaluated
        assert (lm_evaluated_copy._mask == lm._mask_function()).all()
        assert lm_evaluated_copy._mask_function is lm._mask_function
        # and test a non-lazy mask
        m = np.ones(10, dtype=bool)
        nlm = LazyMask(mask=m)
        nlm_copy = copy(nlm)
        assert nlm_copy._evaluated
        assert (nlm_copy._mask == m).all()
        assert nlm_copy._mask_function is None

    def test_deepcopy(self, lm):
        """Check deep copying behaviour of ``LazyMask`` objects."""
        # first copy before evaluating
        lm_unevaluated_copy = deepcopy(lm)
        assert lm_unevaluated_copy._evaluated is False
        assert not hasattr(lm_unevaluated_copy, "_mask")
        assert lm_unevaluated_copy._mask_function is lm._mask_function
        # trigger evaluated
        lm._evaluate()
        assert lm._evaluated
        # now copy after evaluating
        lm_evaluated_copy = deepcopy(lm)
        assert lm_evaluated_copy._evaluated
        assert (lm_evaluated_copy._mask == lm._mask_function()).all()
        assert lm_evaluated_copy._mask_function is lm._mask_function
        # and test a non-lazy mask
        m = np.ones(10, dtype=bool)
        nlm = LazyMask(mask=m)
        nlm_copy = copy(nlm)
        assert nlm_copy._evaluated
        assert (nlm_copy._mask == m).all()
        assert nlm_copy._mask_function is None

    def test_compare_lazymask(self, lm):
        """Check comparison behaviour between two ``LazyMask`` objects."""
        lm2 = copy(lm)
        with pytest.raises(
            ValueError, match="Cannot compare when one or more masks are not evaluated."
        ):
            lm == lm2
        lm._evaluate()
        lm2._evaluate()
        assert lm == lm2
        lmn = LazyMask(mask=None)
        assert lm != lmn

    def test_compare_nonlazymask(self, lm):
        """Check comparison behaviour between a ``LazyMask`` and other objects."""
        m = np.ones(10, dtype=bool)
        with pytest.raises(
            ValueError, match="Cannot compare when one or more masks are not evaluated."
        ):
            lm == m
        lm._evaluate()
        assert lm == m
        assert lm != np.zeros(10, dtype=bool)

    def test_compare_nonemask(self):
        """Check comparison behaviour between null masks."""
        lm = LazyMask(mask=None)
        assert lm == lm
        assert not lm != lm
