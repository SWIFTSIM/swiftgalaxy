"""Tests for applying masks to swiftgalaxy, datasets and named columns."""

import pytest
from copy import copy, deepcopy
import numpy as np
import unyt as u
from unyt.testing import assert_allclose_units
from swiftsimio import cosmo_quantity
from swiftgalaxy import MaskCollection, SWIFTGalaxy
from swiftgalaxy.halo_catalogues import Standalone, Caesar
from swiftgalaxy.demo_data import (
    ToyHF,
    _present_particle_types,
    _n_g_all,
    _n_dm_all,
    _n_s_all,
    _n_bh_all,
)
from swiftgalaxy.masks import LazyMask

abstol_nd = 1.0e-4
reltol_nd = 1.0e-4


def assert_no_data_loaded(sg: SWIFTGalaxy) -> None:
    """
    Iterate over all datasets and asserts that they are ``None``.

    Parameters
    ----------
    sg : ~swiftgalaxy.reader.SWIFTGalaxy
        The :class:`~swiftgalaxy.reader.SWIFTGalaxy` to check for loaded data.

    Raises
    ------
    AssertionError
        If any datasets are not ``None``.
    """
    for ptype in sg.metadata.present_group_names:
        for field_name in getattr(sg, ptype).group_metadata.field_names:
            assert (
                getattr(getattr(sg, ptype)._particle_dataset, f"_{field_name}") is None
            )


class TestMaskingSWIFTGalaxy:
    """Test applying masks to :class:`~swiftgalaxy.reader.SWIFTGalaxy` objects."""

    @pytest.mark.parametrize("particle_name", _present_particle_types.values())
    def test_getattr_masking(self, sg, particle_name):
        """Test masking with square bracket notation. Uses an order reversing mask."""
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
        em_before = sg._extra_mask
        mask = np.s_[::-1]
        ids_before = getattr(sg, particle_name).particle_ids
        if before_load:
            getattr(sg, particle_name)._particle_dataset._particle_ids = None
            sg._extra_mask = em_before
        sg.mask_particles(MaskCollection(**{particle_name: mask}))
        ids = getattr(sg, particle_name).particle_ids
        assert_allclose_units(ids_before[mask], ids, rtol=0, atol=0)

    @pytest.mark.parametrize("particle_name", _present_particle_types.values())
    @pytest.mark.parametrize("before_load", (True, False))
    def test_reordering_int_mask(self, sg, particle_name, before_load):
        """Test an integer array mask that re-orders elements and changes the length."""
        em_before = sg._extra_mask
        ids_before = getattr(sg, particle_name).particle_ids
        mask = np.arange(ids_before.size)
        # randomize order (in-place operation)
        np.random.shuffle(mask)
        # keep half the particles
        mask = mask[: mask.size // 2]
        if before_load:
            getattr(sg, particle_name)._particle_dataset._particle_ids = None
            sg._extra_mask = em_before
        sg.mask_particles(MaskCollection(**{particle_name: mask}))
        ids = getattr(sg, particle_name).particle_ids
        assert_allclose_units(ids_before[mask], ids, rtol=0, atol=0)

    @pytest.mark.parametrize("particle_name", _present_particle_types.values())
    @pytest.mark.parametrize("before_load", (True, False))
    def test_bool_mask(self, sg, particle_name, before_load):
        """Test whether a boolean array mask works."""
        em_before = sg._extra_mask
        ids_before = getattr(sg, particle_name).particle_ids
        # randomly keep about half of particles
        mask = np.random.rand(ids_before.size) > 0.5
        if before_load:
            getattr(sg, particle_name)._particle_dataset._particle_ids = None
            sg._extra_mask = em_before
        sg.mask_particles(MaskCollection(**{particle_name: mask}))
        ids = getattr(sg, particle_name).particle_ids
        assert_allclose_units(ids_before[mask], ids, rtol=0, atol=0)

    @pytest.mark.parametrize("before_load", (True, False))
    def test_data_masked(self, sg, before_load):
        """Test that data get masked."""
        em_before = sg._extra_mask
        masses_before = sg.gas.masses
        mask = np.random.rand(masses_before.size) > 0.5
        if before_load:
            sg.gas._particle_dataset._masses = None
            sg._extra_mask = em_before
        sg.mask_particles(MaskCollection(**{"gas": mask}))
        masses = sg.gas.masses
        assert_allclose_units(
            masses_before[mask], masses, rtol=reltol_nd, atol=abstol_nd
        )

    @pytest.mark.parametrize("before_load", (True, False))
    def test_namedcolumn_masked(self, sg, before_load):
        """Test that named columns get masked too."""
        em_before = sg._extra_mask
        neutral_before = sg.gas.hydrogen_ionization_fractions.neutral
        mask = np.random.rand(neutral_before.size) > 0.5
        if before_load:
            sg.gas.hydrogen_ionization_fractions._named_column_dataset._neutral = None
            sg._extra_mask = em_before
        sg.mask_particles(MaskCollection(**{"gas": mask}))
        neutral = sg.gas.hydrogen_ionization_fractions.neutral
        assert_allclose_units(
            neutral_before[mask], neutral, rtol=reltol_nd, atol=abstol_nd
        )

    def test_mask_without_spatial_mask(self, sg_no_hf):
        """
        Check that if we have no masks we read everything in the box (and warn about it).

        Then that we can still apply an extra mask, and a second one (there's specific
        logic for applying two consecutively).
        """
        # check that extra mask is blank for all particle types:
        assert sg_no_hf._extra_mask.gas.mask is Ellipsis
        assert sg_no_hf._extra_mask.dark_matter.mask is Ellipsis
        assert sg_no_hf._extra_mask.stars.mask is Ellipsis
        assert sg_no_hf._extra_mask.black_holes.mask is Ellipsis
        # check that cell mask is blank for all particle types:
        assert sg_no_hf._spatial_mask is None
        # check that we read all the particles:
        assert sg_no_hf.gas.masses.size == _n_g_all
        assert sg_no_hf.dark_matter.masses.size == _n_dm_all
        assert sg_no_hf.stars.masses.size == _n_s_all
        assert sg_no_hf.black_holes.masses.size == _n_bh_all
        # now apply an extra mask
        sg_no_hf.mask_particles(MaskCollection(gas=np.s_[:1000]))
        assert sg_no_hf.gas.masses.size == 1000
        # and the second consecutive one
        sg_no_hf.mask_particles(MaskCollection(gas=np.s_[:100]))
        assert sg_no_hf.gas.masses.size == 100

    def test_repeated_copy_mask(self, sg_soap):
        """
        Check that we can apply a copying mask operation more than once.

        Regression test for https://github.com/SWIFTSIM/swiftgalaxy/issues/89.
        This had previously caused an ``IndexError``, specifically when using a
        :class:`~swiftgalaxy.halo_catalogues.SOAP` catalogue (because it uses a boolean
        ``"bound_only"`` mask).
        """
        sg_copy1 = sg_soap[
            MaskCollection(
                gas=sg_soap.gas.spherical_coordinates.r
                < cosmo_quantity(
                    3,
                    u.kpc,
                    comoving=True,
                    scale_factor=sg_soap.metadata.scale_factor,
                    scale_exponent=1,
                )
            )
        ]
        sg_copy2 = sg_soap[
            MaskCollection(
                gas=sg_soap.gas.spherical_coordinates.r
                < cosmo_quantity(
                    2,
                    u.kpc,
                    comoving=True,
                    scale_factor=sg_soap.metadata.scale_factor,
                    scale_exponent=1,
                )
            )
        ]
        assert (
            sg_copy1.gas.spherical_coordinates.r
            < cosmo_quantity(
                3,
                u.kpc,
                comoving=True,
                scale_factor=sg_soap.metadata.scale_factor,
                scale_exponent=1,
            )
        ).all()
        assert (
            sg_copy2.gas.spherical_coordinates.r
            < cosmo_quantity(
                2,
                u.kpc,
                comoving=True,
                scale_factor=sg_soap.metadata.scale_factor,
                scale_exponent=1,
            )
        ).all()

    @pytest.mark.parametrize("before_load", (True, False))
    def test_chained_masking(self, sg, before_load):
        """
        Check that we can mask repeatedly.

        Check both the case with (sg) and without (sg_no_hf) a spatial mask.
        """
        em_before = sg._extra_mask
        ids_before_sg = sg.gas.particle_ids
        if before_load:
            sg.gas._particle_dataset._particle_ids = None
            sg._extra_mask = em_before
        sg.mask_particles(MaskCollection(gas=np.s_[::2]))
        sg.mask_particles(MaskCollection(gas=np.s_[::2]))
        assert_allclose_units(ids_before_sg[::2][::2], sg.gas.particle_ids)

    @pytest.mark.parametrize("before_load", (True, False))
    def test_chained_masking_without_spatial(self, sg_no_hf, before_load):
        """
        Check that we can mask repeatedly.

        Check both the case with (sg) and without (sg_no_hf) a spatial mask.
        """
        ids_before_sg_no_hf = sg_no_hf.gas.particle_ids
        if before_load:
            sg_no_hf.gas._particle_dataset._particle_ids = None
        sg_no_hf.mask_particles(MaskCollection(gas=np.s_[::2]))
        sg_no_hf.mask_particles(MaskCollection(gas=np.s_[::2]))
        assert_allclose_units(ids_before_sg_no_hf[::2][::2], sg_no_hf.gas.particle_ids)

    def test_mask_combining_is_lazy(self, sg_soap):
        """Check that no data loading is triggered by lazy masking."""
        assert sg_soap.halo_catalogue.extra_mask == "bound_only"
        # check that setting bound_only mask hasn't triggered any data reads:
        assert_no_data_loaded(sg_soap)
        # combine existing lazy mask with a new mask:
        sg_soap.mask_particles(MaskCollection(gas=np.s_[:100]))
        # check that applying the new mask hasn't triggered any data reads:
        assert_no_data_loaded(sg_soap)
        # also check the copy-masking case
        new_sg = sg_soap[MaskCollection(dark_matter=np.s_[:100])]
        assert_no_data_loaded(new_sg)
        # now check that we can load successfully:
        sg_soap.gas.group_nr_bound
        assert (
            sg_soap.gas.group_nr_bound
            == sg_soap.halo_catalogue.input_halos.halo_catalogue_index
        ).all()
        assert sg_soap.gas.group_nr_bound.size == 100
        # and check we haven't loaded the DM group IDs, just to be sure:
        assert sg_soap.dark_matter._particle_dataset._group_nr_bound is None

    @pytest.mark.parametrize("load_before", (True, False))
    def test_get_bound_only_mask(self, sg_hf, load_before):
        """Check applying a bound_only mask from any catalogue (not Standalone)."""
        if load_before:
            for ptype in sg_hf.metadata.present_group_names:
                getattr(sg_hf, ptype).masses
        if isinstance(sg_hf.halo_catalogue, Standalone):
            with pytest.raises(NotImplementedError):
                sg_hf.get_bound_only_mask()
            return
        sg_hf.mask_particles(sg_hf.get_bound_only_mask())
        if isinstance(sg_hf.halo_catalogue, Caesar) and not load_before:
            with pytest.raises(RuntimeError):
                for ptype in sg_hf.metadata.present_group_names:
                    getattr(sg_hf, ptype).masses
            return
        for ptype in sg_hf.metadata.present_group_names:
            getattr(sg_hf, ptype).masses

    def test_get_bound_only_mask_raises_without_halo_catalogue(self, sg_no_hf):
        """Check that getting a bound_only mask requires a halo catalogue."""
        with pytest.raises(RuntimeError, match="without an associated halo catalogue"):
            sg_no_hf.get_bound_only_mask()

    def test_get_bound_only_mask_is_lazy(self, sg):
        """Check that creating a bound_only mask does not trigger data loading."""
        sg_unbound = SWIFTGalaxy(
            sg.snapshot_filename,
            ToyHF(snapfile=sg.snapshot_filename, extra_mask=None),
            transforms_like_coordinates={"coordinates", "extra_coordinates"},
            transforms_like_velocities={"velocities", "extra_velocities"},
        )
        assert_no_data_loaded(sg_unbound)
        current_bound_only = sg_unbound.get_bound_only_mask()
        assert_no_data_loaded(sg_unbound)
        # evaluate one particle type to ensure lazy mask works
        assert current_bound_only.gas.mask.size > 0

    def test_get_bound_only_mask_compatible_with_current_particles(self, sg):
        """Check that returned mask is directly applicable to currently selected data."""
        sg_unbound = SWIFTGalaxy(
            sg.snapshot_filename,
            ToyHF(snapfile=sg.snapshot_filename, extra_mask=None),
            transforms_like_coordinates={"coordinates", "extra_coordinates"},
            transforms_like_velocities={"velocities", "extra_velocities"},
        )
        sg_bound = SWIFTGalaxy(
            sg.snapshot_filename,
            ToyHF(snapfile=sg.snapshot_filename, extra_mask="bound_only"),
            transforms_like_coordinates={"coordinates", "extra_coordinates"},
            transforms_like_velocities={"velocities", "extra_velocities"},
        )

        sg_unbound.mask_particles(
            MaskCollection(
                gas=np.s_[::3],
                dark_matter=np.s_[::-2],
                stars=np.s_[::2],
                black_holes=np.s_[...],
            )
        )

        current_bound_only = sg_unbound.get_bound_only_mask()
        for ptype in sg_unbound.metadata.present_group_names:
            current_ids = getattr(sg_unbound, ptype).particle_ids
            expected_bound = np.isin(current_ids, getattr(sg_bound, ptype).particle_ids)
            got_mask = getattr(current_bound_only, ptype).mask
            assert got_mask.shape == current_ids.shape
            assert np.array_equal(got_mask, expected_bound)

    def test_get_bound_only_mask_relative_to_current_default(self, sg):
        """Check default bound-only mask is all-True for bound-only SWIFTGalaxy."""
        current_bound_only = sg.get_bound_only_mask()
        for ptype in sg.metadata.present_group_names:
            got_mask = getattr(current_bound_only, ptype).mask
            assert got_mask.shape == getattr(sg, ptype).particle_ids.shape
            assert got_mask.dtype == bool
            assert got_mask.all()

    def test_get_bound_only_after_manual_masking(self, sg):
        """Check that we can get a bound_only mask after applying a manual mask."""
        sg.mask_particles(
            MaskCollection(
                gas=np.s_[::3],
                dark_matter=np.s_[::-2],
                stars=np.s_[::2],
                black_holes=np.s_[...],
            )
        )
        current_bound_only = sg.get_bound_only_mask()
        for ptype in sg.metadata.present_group_names:
            got_mask = getattr(current_bound_only, ptype).mask
            assert got_mask.dtype == bool
            assert got_mask.dtype == bool
            assert got_mask.all()


class TestMaskingParticleDatasets:
    """Test applying masks to particle datasets."""

    @pytest.mark.parametrize("particle_name", _present_particle_types.values())
    @pytest.mark.parametrize("before_load", (True, False))
    def test_reordering_slice_mask(self, sg, particle_name, before_load):
        """Test whether a slice mask that re-orders elements works."""
        em_before = sg._extra_mask
        mask = np.s_[::-1]
        ids_before = getattr(sg, particle_name).particle_ids
        if before_load:
            getattr(sg, particle_name)._particle_dataset._particle_ids = None
            sg._extra_mask = em_before
        masked_dataset = getattr(sg, particle_name)[mask]
        ids = masked_dataset.particle_ids
        assert_allclose_units(ids_before[mask], ids, rtol=0, atol=0)

    @pytest.mark.parametrize("particle_name", _present_particle_types.values())
    @pytest.mark.parametrize("before_load", (True, False))
    def test_reordering_int_mask(self, sg, particle_name, before_load):
        """Test masking with an integer array: re-orders elements and changes length."""
        em_before = sg._extra_mask
        ids_before = getattr(sg, particle_name).particle_ids
        mask = np.arange(ids_before.size)
        # randomize order (in-place operation)
        np.random.shuffle(mask)
        # keep half the particles
        mask = mask[: mask.size // 2]
        if before_load:
            getattr(sg, particle_name)._particle_dataset._particle_ids = None
            sg._extra_mask = em_before
        masked_dataset = getattr(sg, particle_name)[mask]
        ids = masked_dataset.particle_ids
        assert_allclose_units(ids_before[mask], ids, rtol=0, atol=0)

    @pytest.mark.parametrize("particle_name", _present_particle_types.values())
    @pytest.mark.parametrize("before_load", (True, False))
    def test_bool_mask(self, sg, particle_name, before_load):
        """Test whether a boolean array mask works."""
        em_before = sg._extra_mask
        ids_before = getattr(sg, particle_name).particle_ids
        # randomly keep about half of particles
        mask = np.random.rand(ids_before.size) > 0.5
        if before_load:
            getattr(sg, particle_name)._particle_dataset._particle_ids = None
            sg._extra_mask = em_before
        masked_dataset = getattr(sg, particle_name)[mask]
        ids = masked_dataset.particle_ids
        assert_allclose_units(ids_before[mask], ids, rtol=0, atol=0)

    def test_chaining_masks(self, sg):
        """
        Check that we can mask a particle dataset after masking the swiftgalaxy.

        This is a regression test, but with no associated github issue.
        """
        sg.mask_particles(
            MaskCollection(
                gas=sg.gas.spherical_coordinates.r
                < cosmo_quantity(
                    3,
                    u.kpc,
                    comoving=True,
                    scale_factor=sg.metadata.scale_factor,
                    scale_exponent=1,
                )
            )
        )
        # this had previously caused a crash in version <=2.4.1:
        # IndexError: boolean index did not match indexed array along axis 0;
        # size of axis is 5000 but size of corresponding boolean axis is 1480
        sg.gas[
            sg.gas.spherical_coordinates.r
            > cosmo_quantity(
                1,
                u.kpc,
                comoving=True,
                scale_factor=sg.metadata.scale_factor,
                scale_exponent=1,
            )
        ]


class TestMaskingNamedColumnDatasets:
    """Test applying masks to named column datasets."""

    @pytest.mark.parametrize("before_load", (True, False))
    def test_reordering_slice_mask(self, sg, before_load):
        """Test whether a slice mask that re-orders elements works."""
        em_before = sg._extra_mask
        mask = np.s_[::-1]
        fractions_before = sg.gas.hydrogen_ionization_fractions.neutral
        if before_load:
            sg.gas.hydrogen_ionization_fractions._neutral = None
            sg._extra_mask = em_before
        masked_namedcolumnsdataset = sg.gas.hydrogen_ionization_fractions[mask]
        fractions = masked_namedcolumnsdataset.neutral
        assert_allclose_units(
            fractions_before[mask], fractions, rtol=reltol_nd, atol=abstol_nd
        )

    @pytest.mark.parametrize("before_load", (True, False))
    def test_reordering_int_mask(self, sg, before_load):
        """Test masking with an integer array: re-orders and changes the length."""
        em_before = sg._extra_mask
        fractions_before = sg.gas.hydrogen_ionization_fractions.neutral
        mask = np.arange(fractions_before.size)
        # randomize order (in-place operation)
        np.random.shuffle(mask)
        # keep half the particles
        mask = mask[: mask.size // 2]
        if before_load:
            sg.gas.hydrogen_ionization_fractions._neutral = None
            sg._extra_mask = em_before
        masked_namedcolumnsdataset = sg.gas.hydrogen_ionization_fractions[mask]
        fractions = masked_namedcolumnsdataset.neutral
        assert_allclose_units(
            fractions_before[mask], fractions, rtol=reltol_nd, atol=abstol_nd
        )

    @pytest.mark.parametrize("before_load", (True, False))
    def test_bool_mask(self, sg, before_load):
        """Test whether a boolean array mask works."""
        em_before = sg._extra_mask
        fractions_before = sg.gas.hydrogen_ionization_fractions.neutral
        # randomly keep about half of particles
        mask = np.random.rand(fractions_before.size) > 0.5
        if before_load:
            sg.gas.hydrogen_ionization_fractions._neutral = None
            sg._extra_mask = em_before
        masked_namedcolumnsdataset = sg.gas.hydrogen_ionization_fractions[mask]
        fractions = masked_namedcolumnsdataset.neutral
        assert_allclose_units(
            fractions_before[mask], fractions, rtol=reltol_nd, atol=abstol_nd
        )


class TestMultiModeMask:
    """Tests handling a halo catalogue with multiple galaxies selected."""

    def test_mask_multi_invalid(self, sg):
        """
        Check that catalogues raise when accessing single row in multi mode.

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
        Check that caesar catalogues raise when accessing single row in multi mode.

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
            """A mask generator that counts how many times its ``__call__`` is called."""

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

    def test_ensure_combinable_evaluated(self, sg):
        """Test that making a LazyMask 'combinable' results in an integer index array."""
        lm = LazyMask(np.s_[:10])
        assert not isinstance(lm.mask, np.ndarray)
        assert not lm._combinable
        lm._ensure_combinable(sg=sg, mask_type="gas")
        assert lm._evaluated
        assert isinstance(lm.mask, np.ndarray)
        assert lm.mask.dtype == int
        assert lm._combinable
        assert len(lm.mask) == 10

    def test_ensure_combinable_unevaluated(self, sg):
        """Test that making a LazyMask 'combinable' results in an integer index array."""
        lm = LazyMask(mask_function=lambda: np.s_[:10])
        assert not lm._combinable
        lm._ensure_combinable(sg=sg, mask_type="gas")
        assert lm._combinable
        assert not lm._evaluated
        assert isinstance(lm.mask, np.ndarray)  # triggers evaluation
        assert lm.mask.dtype == int
        assert len(lm.mask) == 10

    def test_combined_with(self, sg):
        """Check that combining two masks results in 'chaining together' the masks."""
        lm1 = LazyMask(mask=np.s_[::2])
        lm2 = LazyMask(mask=np.s_[::2])
        combined_lm = lm1._combined_with(lm2, sg=sg, mask_type="gas")
        assert isinstance(combined_lm.mask, np.ndarray)
        assert combined_lm.mask.dtype == int
        assert sg._spatial_mask is not None
        # expect length // 4 since we did [::2] twice:
        assert (
            combined_lm.mask.size
            == np.sum(sg._spatial_mask.get_masked_counts_offsets()[0]["gas"]) // 4
        )


class TestMaskCollection:
    """Tests for the MaskCollection class."""

    def test_warning_for_unexpected_field_in_combining_mask_collections(self, sg):
        """Check that trying to combine with a collection with extra fields warns."""
        mc1 = MaskCollection(gas=Ellipsis)
        mc2 = MaskCollection(gas=Ellipsis, dark_matter=Ellipsis)
        with pytest.warns(UserWarning, match="Unexpected fields"):
            mc1.combined_with(mc2, sg=sg)

    def test_blank_from_mask_types(self):
        """Test that a set of ``Ellipsis`` masks with desired names is created."""
        mask_types = ("a", "b", "c")
        mc = MaskCollection._blank_from_mask_types(mask_types)
        assert len(mc._masks) == len(mask_types)
        assert set(mc._masks.keys()) == set(mask_types)
        for k in mask_types:
            assert getattr(mc, k) == LazyMask(mask=Ellipsis)

    def test_from_mask_types_and_values(self):
        """Test that a set of masks with all desired names and mask values is created."""
        mask_types = ("a", "b", "c")
        values = {"a": np.s_[:10], "c": np.array([True, False], dtype=bool)}
        mc = MaskCollection._from_mask_types_and_values(mask_types, values)
        assert len(mc._masks) == len(mask_types)
        assert set(mc._masks.keys()) == set(mask_types)
        for k in mask_types:
            if k in values:
                assert getattr(mc, k) == LazyMask(mask=values[k])
            else:
                assert getattr(mc, k) == LazyMask(mask=Ellipsis)

    def test_mask_not_found(self):
        """Test that AttributeError is raised when a non-existant mask is requested."""
        mc = MaskCollection(a=np.s_[:10])
        assert "b" not in mc._masks.keys()
        with pytest.raises(
            AttributeError, match="'MaskCollection' has no attribute 'b'"
        ):
            mc.b

    def test_combined_with(self, sg):
        """Check that combining two masks results in 'chaining together' the masks."""
        mc1 = MaskCollection(gas=np.s_[:10], dark_matter=np.s_[::2])
        mc2 = MaskCollection(gas=np.s_[:5], dark_matter=np.s_[::2])
        combined_mc = mc1.combined_with(mc2, sg=sg)
        assert isinstance(combined_mc.gas.mask, np.ndarray)
        assert combined_mc.gas.mask.dtype == int
        assert combined_mc.gas.mask.size == 5
        assert isinstance(combined_mc.dark_matter.mask, np.ndarray)
        assert combined_mc.dark_matter.mask.dtype == int
        assert sg._spatial_mask is not None
        # expect length // 4 since we did [::2] twice:
        assert (
            combined_mc.dark_matter.mask.size
            == np.sum(sg._spatial_mask.get_masked_counts_offsets()[0]["dark_matter"])
            // 4
        )
