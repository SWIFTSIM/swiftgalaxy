"""Test the tools for iterating over :class:`~swiftgalaxy.reader.SWIFTGalaxy`s."""

import pytest
import re
import pickle
import multiprocessing
from scipy.spatial.transform import Rotation
from pathlib import Path
import numpy as np
import unyt as u
from unyt.testing import assert_allclose_units
from swiftgalaxy.demo_data import (
    _toysnap_filename,
    ToyHF,
    _present_particle_types,
    _toysoap_virtual_snapshot_filename,
    _toysoap_membership_filebase,
    _toysoap_filename,
    _toyvr_filebase,
    _toycaesar_filename,
    _create_toysnap,
    _remove_toysnap,
    _create_toysoap,
    _create_toyvr,
    _create_toycaesar,
    _remove_toysoap,
    _remove_toyvr,
    _remove_toycaesar,
)
from conftest import hfs
from swiftsimio.objects import cosmo_array
from swiftgalaxy.reader import SWIFTGalaxy
from swiftgalaxy.iterator import (
    SWIFTGalaxies,
    _CoordinateFrameSpec,
    _worker_context,
)
from swiftgalaxy.halo_catalogues import Standalone, SOAP, Velociraptor, Caesar


class TestSWIFTGalaxies:
    """Test the :mod:`swiftgalaxy` iteration helper class."""

    def test_eval_sparse_optimized_solution(self, toysnap):
        """
        Check that the sparse solution is chosen when optimal.

        It should match expectations for a case that we can work out by hand.
        """
        # place a single target in the centre of each cell
        # this should make sparse iteration optimal
        # at a cost of 2 cell reads
        sgs = SWIFTGalaxies(
            toysnap["toysnap_filename"],
            Standalone(
                centre=cosmo_array(
                    [[2.5, 5.0, 5.0], [7.5, 5.0, 5.0]],
                    u.Mpc,
                    comoving=True,
                    scale_factor=1.0,
                    scale_exponent=1,
                ),
                velocity_centre=cosmo_array(
                    [[0, 0, 0] * 2],
                    u.km / u.s,
                    comoving=True,
                    scale_factor=1.0,
                    scale_exponent=0,
                ),
                spatial_offsets=cosmo_array(
                    [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]],
                    u.Mpc,
                    comoving=True,
                    scale_factor=1.0,
                    scale_exponent=1,
                ),
            ),
        )
        sparse_solution = sgs._sparse_optimized_solution
        assert_allclose_units(
            sparse_solution["regions"],
            cosmo_array(
                np.array(
                    [
                        [[0.01, 0.99], [0.01, 0.99], [0.01, 0.99]],
                        [[1.01, 1.99], [0.01, 0.99], [0.01, 0.99]],
                    ]
                )
                * np.array([[[5], [10], [10]]]),
                u.Mpc,
                comoving=True,
                scale_factor=1.0,
                scale_exponent=1,
            ),
            atol=0.0001 * u.Mpc,
        )
        for ss, expected in zip(
            sparse_solution["region_target_indices"], [np.array([0]), np.array([1])]
        ):
            assert np.allclose(ss, expected, atol=0)
        assert sparse_solution["cost"] == 2
        for k in sgs._solution.keys():
            if isinstance(sgs._solution[k], np.integer):
                assert sgs._solution[k] == sparse_solution[k]
            elif isinstance(sgs._solution[k], list):
                for i in range(len(sgs._solution[k])):
                    assert np.allclose(
                        sgs._solution[k][i], sparse_solution[k][i], atol=0
                    )
            else:
                assert_allclose_units(
                    sgs._solution[k], sparse_solution[k], atol=0.001 * u.Mpc
                )

    def test_eval_dense_optimized_solution(self, toysnap):
        """
        Check that the dense solution is chosen when optimal.

        It should match expectations for a case that we can work out by hand.
        """
        # Place a single target in the centre of each cell
        # and ones straddling many faces, vertices and corners
        # this should make dense iteration optimal
        # at a cost of 2-4 cell reads.
        # * Technically this solution is inefficient because
        # the cell regions are repeated, because the code is
        # not clever enough to recognize copies wrapped through
        # the periodic bounday. But having so few cells is
        # not an expected case for "real" simulations.
        # The dense solution could start to include some
        # copies if the target region is larger than ~0.5 the
        # box size in any dimension.
        sgs = SWIFTGalaxies(
            toysnap["toysnap_filename"],
            Standalone(
                centre=cosmo_array(
                    [
                        [2.5, 5.0, 5.0],
                        [5.0, 5.0, 5.0],
                        [7.5, 5.0, 5.0],
                        [5.0, 0.0, 0.0],
                        [5.0, 0.0, 9.9],
                        [5.0, 9.9, 0.0],
                        [5.0, 9.9, 9.9],
                        [9.9, 0.0, 9.9],
                        [9.9, 9.9, 0.0],
                        [9.9, 9.9, 9.9],
                    ],
                    u.Mpc,
                    comoving=True,
                    scale_factor=1.0,
                    scale_exponent=1,
                ),
                velocity_centre=cosmo_array(
                    [[0, 0, 0] * 10],
                    u.km / u.s,
                    comoving=True,
                    scale_factor=1.0,
                    scale_exponent=0,
                ),
                spatial_offsets=cosmo_array(
                    [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]],
                    u.Mpc,
                    comoving=True,
                    scale_factor=1.0,
                    scale_exponent=1,
                ),
            ),
            # force this - with our 2-cell test snapshot we can't contrive for this
            # approach to be automatically determined to be optimal:
            optimize_iteration="dense",
        )
        dense_solution = sgs._dense_optimized_solution
        assert_allclose_units(
            dense_solution["regions"],
            cosmo_array(
                np.array(
                    [
                        [[0.5, 4.5], [3.0, 7.0], [3.0, 7.0]],
                        [[3.0, 11.9], [-2.0, 11.9], [-2.0, 11.9]],
                    ]
                ),
                u.Mpc,
                comoving=True,
                scale_factor=1.0,
                scale_exponent=1,
            ),
            atol=0.001 * u.Mpc,
        )
        assert len(dense_solution["region_target_indices"]) == 2
        for ds, expected in zip(
            dense_solution["region_target_indices"],
            [
                np.array([0]),
                np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
            ],
        ):
            assert np.allclose(ds, expected, atol=0)
        # expect cost_min, cost_max = 9, 35
        # so expect cost = (9 + 35) / 2 = 22
        assert dense_solution["cost"] == 22
        for k in sgs._solution.keys():
            if isinstance(sgs._solution[k], int):
                assert sgs._solution[k] == dense_solution[k]
            elif isinstance(sgs._solution[k], list):
                for i in range(len(sgs._solution[k])):
                    assert np.allclose(
                        sgs._solution[k][i], dense_solution[k][i], atol=0
                    )
            else:
                assert_allclose_units(
                    sgs._solution[k], dense_solution[k], atol=0.001 * u.Mpc
                )

    def test_iteration_order(self, sgs):
        """Check that the iteration order is that computed for the two optimizations."""
        # force dense solution
        sgs._solution = sgs._dense_optimized_solution
        assert np.allclose(
            sgs.iteration_order,
            np.concatenate(sgs._dense_optimized_solution["region_target_indices"]),
            atol=0,
        )
        # force sparse solution
        sgs._solution = sgs._sparse_optimized_solution
        assert np.allclose(
            sgs.iteration_order,
            np.concatenate(sgs._sparse_optimized_solution["region_target_indices"]),
            atol=0,
        )

    def test_iterate(self, sgs):
        """
        Check that we iterate over the right number of SWIFTGalaxy objects.

        Each should behave like a SWIFTGalaxy created on its own for each target.
        """
        count = 0
        for sg_from_sgs in sgs:
            sg = SWIFTGalaxy(
                sgs.snapshot_filename,
                ToyHF(
                    snapfile=sgs.snapshot_filename,
                    index=sg_from_sgs.halo_catalogue.index,
                ),
            )
            for ptype in _present_particle_types.values():
                getattr(sg._extra_mask, ptype)._make_combinable(sg=sg, mask_type=ptype)
                assert np.all(
                    getattr(sg_from_sgs._extra_mask, ptype).mask
                    == getattr(sg._extra_mask, ptype).mask
                )
            count += 1
        assert count == len(sgs.halo_catalogue.index)

    def test_warn_on_no_preload(self, toysnap):
        """Check that we warn users if they specify anything to pre-load (deprecated)."""
        with pytest.warns(DeprecationWarning, match="Preloading is no longer required"):
            SWIFTGalaxies(
                toysnap["toysnap_filename"],
                ToyHF(toysnap["toysnap_filename"], index=[0, 1]),
                preload={"gas.coordinates"},
            )

    def test_exception_on_repeated_targets(self, toysnap):
        """
        Check that duplicate targets are forbidden.

        Due to especially swiftsimio's masking behaviour having duplicate targets in the
        list for a SWIFTGalaxies causes all kinds of problems, so make sure we raise an
        exception if a user tries to do this.
        """
        with pytest.raises(ValueError, match="must not contain duplicates"):
            SWIFTGalaxies(
                toysnap["toysnap_filename"],
                ToyHF(toysnap["toysnap_filename"], index=[0, 0]),
            )

    def test_map(self, tmp_path_factory, hf_multi):
        """
        Check that map method returns results in the same order as the input target list.

        We're careful in this test to make sure that the iteration order is different from
        the input list order.
        """
        if isinstance(hf_multi, SOAP):
            tp = hf_multi.soap_file.parent
        elif isinstance(hf_multi, Caesar):
            tp = hf_multi.caesar_file.parent
        elif isinstance(hf_multi, Velociraptor):
            tp = Path(hf_multi.velociraptor_files["properties"]).parent
        else:
            tp = tmp_path_factory.mktemp(_toysnap_filename.parent)
            _create_toysnap(snapfile=tp / _toysnap_filename.name)
        toysnap_filename = tp / _toysnap_filename.name
        sgs = SWIFTGalaxies(
            (
                tp / _toysoap_virtual_snapshot_filename.name
                if isinstance(hf_multi, SOAP)
                else toysnap_filename
            ),
            hf_multi,
        )

        def f(sg):
            if isinstance(hf_multi, Standalone):
                return int(
                    np.argwhere(
                        np.all(
                            sg.halo_catalogue.centre == sg.halo_catalogue._centre,
                            axis=1,
                        )
                    ).squeeze()
                )
            # _index_attr has leading underscore, access through property with [1:]
            return getattr(sg.halo_catalogue, sg.halo_catalogue._index_attr[1:])

        if (np.diff(sgs.iteration_order) == 1).all():
            # if we iterate in order success is trivial, ensure that we don't:
            sgs._solution["regions"] = sgs._solution["regions"][::-1]
            sgs._solution["region_target_indices"] = sgs._solution[
                "region_target_indices"
            ][::-1]
        # double-check that success won't be trivial:
        assert not (np.diff(sgs.iteration_order) == 1).all()
        # check that map returns results ordered in input order
        if isinstance(hf_multi, Standalone):
            assert sgs.map(f) == [0, 1]
            return
        assert sgs.map(f) == getattr(
            sgs.halo_catalogue, sgs.halo_catalogue._index_attr[1:]
        )
        if isinstance(hf_multi, Standalone):
            _remove_toysnap(snapfile=toysnap_filename)

    def test_arbitrary_index_ordering(
        self, tmp_path_factory, hf_multi_forwards_and_backwards
    ):
        """
        Check that SWIFTGalaxies gives consistent results for any order of target objects.

        Especially important for velociraptor where some logic had to be added to avoid
        hdf5 complaining about an unsorted list of indices to read from file.
        """
        hf_multi, hf_multi_backwards = hf_multi_forwards_and_backwards
        if isinstance(hf_multi, SOAP):
            tp = hf_multi.soap_file.parent
        elif isinstance(hf_multi, Caesar):
            tp = hf_multi.caesar_file.parent
        elif isinstance(hf_multi, Velociraptor):
            tp = Path(hf_multi.velociraptor_files["properties"]).parent
        else:
            tp = tmp_path_factory.mktemp(_toysnap_filename.parent)
            _create_toysnap(snapfile=tp / _toysnap_filename.name)
        toysnap_filename = tp / _toysnap_filename.name

        def f(sg):
            return sg.halo_catalogue.centre

        sgs_forwards = SWIFTGalaxies(
            (
                tp / _toysoap_virtual_snapshot_filename.name
                if isinstance(hf_multi, SOAP)
                else toysnap_filename
            ),
            hf_multi,
        )
        map_forwards = sgs_forwards.map(f)
        sgs_backwards = SWIFTGalaxies(
            (
                tp / _toysoap_virtual_snapshot_filename.name
                if isinstance(hf_multi, SOAP)
                else toysnap_filename
            ),
            hf_multi_backwards,
        )
        map_backwards = sgs_backwards.map(f)
        assert np.allclose(map_forwards, map_backwards[::-1])
        if isinstance(hf_multi, Standalone):
            _remove_toysnap(snapfile=toysnap_filename)

    def test_args_kwargs_to_map(self, sgs):
        """Make sure that we can pass extra args & kwargs to a function given to map."""
        extra_arg = [("foo",), ("bar",)]
        extra_kwarg = [dict(extra_kwarg="spam"), dict(extra_kwarg="eggs")]

        def f(sg, extra_arg, extra_kwarg=None):
            return extra_arg, extra_kwarg

        result = sgs.map(f, args=extra_arg, kwargs=extra_kwarg)
        assert result == [("foo", "spam"), ("bar", "eggs")]

    def test_soap_target_order_consistency(self, toysoap_with_virtual_snapshot):
        """
        Test that SOAP doesn't sneakily reorder the galaxy catalogue.

        SOAP implicitly sorts the target mask (when getting things from the catalogue
        we get them masked in the order that they appear in the catalogue, rather
        than the order that they appear in the mask - say we ask for items [1, 0]
        as a mask, we get back those two items in order [0, 1]). This test checks
        that we get a SWIFTGalaxy from a SWIFTGalaxies that matches the one
        constructed directly when the targets given to the SWIFTGalaxies are not
        in the order that they appear in the catalogue.
        """
        soaps = [
            SOAP(
                soap_file=toysoap_with_virtual_snapshot["toysoap_filename"],
                soap_index=0,
            ),
            SOAP(
                soap_file=toysoap_with_virtual_snapshot["toysoap_filename"],
                soap_index=1,
            ),
        ]
        with pytest.warns(
            UserWarning, match="`constrain_indices` selects indices in order"
        ):
            soap_both = SOAP(
                soap_file=toysoap_with_virtual_snapshot["toysoap_filename"],
                soap_index=[1, 0],
            )
        sgs_individual = [
            SWIFTGalaxy(
                toysoap_with_virtual_snapshot["toysoap_virtual_snapshot_filename"], soap
            )
            for soap in soaps
        ]
        sgs = SWIFTGalaxies(
            toysoap_with_virtual_snapshot["toysoap_virtual_snapshot_filename"],
            soap_both,
        )
        for i, sg in enumerate(sgs):
            sg_single = sgs_individual[sgs.iteration_order[i]]
            assert_allclose_units(
                sg_single.halo_catalogue._region_centre,
                sg.halo_catalogue._region_centre,
            )
            assert_allclose_units(
                sg_single.halo_catalogue.centre,
                sg.halo_catalogue.centre,
            )
            assert_allclose_units(
                sg_single.halo_catalogue.velocity_centre,
                sg.halo_catalogue.velocity_centre,
            )
            assert_allclose_units(
                sg_single.halo_catalogue.spherical_overdensity_200_crit.centre_of_mass,
                sg.halo_catalogue.spherical_overdensity_200_crit.centre_of_mass,
            )

    @pytest.mark.parametrize("hf_type", hfs)
    def test_halo_catalogue_with_non_list_indices(self, hf_type, toysnap_withfof):
        """
        Check that we can initialize a halo_catalogue in multi-galaxy mode.

        Here we check some ordered containers that are not a list.
        """
        toysnap_filename = toysnap_withfof["toysnap_filename"]
        tp = toysnap_filename.parent
        if hf_type == "soap":
            pytest.importorskip("SOAP.compression")
            membership_filebase = tp / _toysoap_membership_filebase.name
            toysoap_filename = tp / _toysoap_filename.name
            toysoap_virtual_snapshot_filename = (
                tp / _toysoap_virtual_snapshot_filename.name
            )
            _create_toysoap(
                filename=toysoap_filename,
                membership_filebase=membership_filebase,
                create_virtual_snapshot=True,
                create_virtual_snapshot_from=toysnap_filename,
                virtual_snapshot_filename=toysoap_virtual_snapshot_filename,
            )
        elif hf_type == "vr":
            pytest.importorskip("velociraptor")
            toyvr_filebase = tp / _toyvr_filebase.name
            _create_toyvr(filebase=toyvr_filebase)
        elif "caesar" in hf_type:
            pytest.importorskip("caesar")
            toycaesar_filename = tp / _toycaesar_filename.name
            _create_toycaesar(filename=toycaesar_filename)
        elif hf_type == "sa":
            return  # doesn't take an index list, nothing to test
        else:
            raise NotImplementedError  # a new halo_catalogue that we're not testing
        for init_indices in (
            np.array([0, 1], dtype=int),
            u.unyt_array([0, 1], u.dimensionless, dtype=int),
        ):
            if hf_type == "soap":
                sgs = SWIFTGalaxies(
                    toysoap_virtual_snapshot_filename,
                    SOAP(toysoap_filename, soap_index=init_indices),
                )
            elif hf_type == "vr":
                sgs = SWIFTGalaxies(
                    toysnap_filename,
                    Velociraptor(
                        velociraptor_filebase=toyvr_filebase, halo_index=init_indices
                    ),
                )
            elif hf_type == "caesar_galaxy":
                sgs = SWIFTGalaxies(
                    toysnap_filename,
                    Caesar(
                        toycaesar_filename,
                        group_type="galaxy",
                        group_index=init_indices,
                    ),
                )
            elif hf_type == "caesar_halo":
                sgs = SWIFTGalaxies(
                    toysnap_filename,
                    Caesar(
                        toycaesar_filename, group_type="halo", group_index=init_indices
                    ),
                )
            for sg in sgs:
                pass  # just go through the iteration
        if hf_type == "soap":
            _remove_toysoap(
                filename=toysoap_filename,
                membership_filebase=membership_filebase,
                virtual_snapshot_filename=toysoap_virtual_snapshot_filename,
            )
        elif hf_type == "vr":
            _remove_toyvr(filebase=toyvr_filebase)
        elif "caesar" in hf_type:
            _remove_toycaesar(filename=toycaesar_filename)

    def test_zero_targets(self, tmp_path_factory, hf_multi_zerotarget):
        """
        Make sure we don't crash with zero targets.

        Instead iterate over zero elements.
        """
        if isinstance(hf_multi_zerotarget, SOAP):
            tp = hf_multi_zerotarget.soap_file.parent
        elif isinstance(hf_multi_zerotarget, Caesar):
            tp = hf_multi_zerotarget.caesar_file.parent
        elif isinstance(hf_multi_zerotarget, Velociraptor):
            tp = Path(hf_multi_zerotarget.velociraptor_files["properties"]).parent
        else:
            tp = tmp_path_factory.mktemp(_toysnap_filename.parent)
            _create_toysnap(snapfile=tp / _toysnap_filename.name)
        toysnap_filename = tp / _toysnap_filename.name
        sgs = SWIFTGalaxies(
            (
                tp / _toysoap_virtual_snapshot_filename.name
                if isinstance(hf_multi_zerotarget, SOAP)
                else toysnap_filename
            ),
            hf_multi_zerotarget,
        )
        for sg in sgs:  # should not crash by iterating
            # but should not reach this:
            raise RuntimeError("Supposed to iterate over 0 elements!")

        def f(sg):
            raise RuntimeError  # we should never call this

        map_result = sgs.map(f)
        assert len(map_result) == 0

    def test_one_target(self, tmp_path_factory, hf_multi_onetarget):
        """Make sure that we don't crash with a single target. Instead iterate over it."""
        if isinstance(hf_multi_onetarget, SOAP):
            tp = hf_multi_onetarget.soap_file.parent
        elif isinstance(hf_multi_onetarget, Caesar):
            tp = hf_multi_onetarget.caesar_file.parent
        elif isinstance(hf_multi_onetarget, Velociraptor):
            tp = Path(hf_multi_onetarget.velociraptor_files["properties"]).parent
        else:
            tp = tmp_path_factory.mktemp(_toysnap_filename.parent)
            _create_toysnap(snapfile=tp / _toysnap_filename.name)
        toysnap_filename = tp / _toysnap_filename.name
        sgs = SWIFTGalaxies(
            (
                tp / _toysoap_virtual_snapshot_filename.name
                if isinstance(hf_multi_onetarget, SOAP)
                else toysnap_filename
            ),
            hf_multi_onetarget,
        )

        def f(sg):
            if isinstance(hf_multi_onetarget, Standalone):
                return int(
                    np.argwhere(
                        np.all(
                            sg.halo_catalogue.centre == sg.halo_catalogue._centre,
                            axis=1,
                        )
                    ).squeeze()
                )
            # _index_attr has leading underscore, access through property with [1:]
            return getattr(sg.halo_catalogue, sg.halo_catalogue._index_attr[1:])

        count = 0
        for sg in sgs:  # should iterate over the one target
            count += 1
            if sg.halo_catalogue._index_attr is not None:
                assert f(sg) == 1
            else:
                assert f(sg) == 0  # standalone gets position in own catalogue: 0
        assert count == 1

        map_result = sgs.map(f)
        assert len(map_result) == 1
        if sg.halo_catalogue._index_attr is not None:
            assert map_result == [1]
        else:
            assert map_result == [0]  # standalone gets position in own catalogue: 0

    def test_catalogue_not_iterable(self, toysnap):
        """Check that trying to use a non-iterable catalogue raises."""
        with pytest.raises(
            ValueError, match="halo_catalogue target list is not iterable"
        ):
            SWIFTGalaxies(
                toysnap["toysnap_filename"],
                ToyHF(snapfile=toysnap["toysnap_filename"], index=0),
            )

    def test_invalid_iteration_mode(self, toysnap):
        """Check that giving an invalid iteration mode raises."""
        with pytest.raises(ValueError, match="optimize_iteration must be one of"):
            SWIFTGalaxies(
                toysnap["toysnap_filename"],
                ToyHF(snapfile=toysnap["toysnap_filename"], index=[0, 1]),
                optimize_iteration="not_implemented",
            )

    def test_coordinate_frame_from_and_auto_recentre_invalid(self, toysnap):
        """Check inheriting coordinate frame and auto-recentering are incompatible."""
        sg = SWIFTGalaxy(
            toysnap["toysnap_filename"],
            ToyHF(snapfile=toysnap["toysnap_filename"], index=0),
        )
        sgs = SWIFTGalaxies(
            toysnap["toysnap_filename"],
            ToyHF(snapfile=toysnap["toysnap_filename"], index=[0, 1]),
            auto_recentre=True,
            coordinate_frame_from=sg,
        )
        with pytest.raises(
            ValueError, match="Cannot use coordinate_frame_from with auto_recentre"
        ):
            for sg_i in sgs:
                pass

    def test_coordinate_frame_from_in_iteration(self, toysnap):
        """Check that we can borrow a coordinate frame when iterating."""
        sg = SWIFTGalaxy(
            toysnap["toysnap_filename"],
            ToyHF(snapfile=toysnap["toysnap_filename"], index=0),
        )
        translation = cosmo_array(
            [1, 0, 0],
            u.Mpc,
            comoving=True,
            scale_factor=sg.metadata.a,
            scale_exponent=1.0,
        )
        sg.translate(translation)
        sgs = SWIFTGalaxies(
            toysnap["toysnap_filename"],
            ToyHF(snapfile=toysnap["toysnap_filename"], index=[0, 1]),
            auto_recentre=False,
            coordinate_frame_from=sg,
        )
        for sg_i in sgs:
            assert np.allclose(sg.halo_catalogue.centre - translation, sg_i.centre)

    def test_internal_units_mismatch_in_coordinate_frame_from(self, toysnap):
        """Check that incompatible internal units raises."""
        sg = SWIFTGalaxy(
            toysnap["toysnap_filename"],
            ToyHF(snapfile=toysnap["toysnap_filename"], index=0),
        )
        sg.metadata.units.length = 1 * u.kpc
        sgs = SWIFTGalaxies(
            toysnap["toysnap_filename"],
            ToyHF(snapfile=toysnap["toysnap_filename"], index=[0, 1]),
            auto_recentre=False,
            coordinate_frame_from=sg,
        )
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Internal units (length and time) of coordinate_frame_from don't match."
            ),
        ):
            for sg_i in sgs:
                pass

    def test_auto_recentre_off(self, toysnap):
        """Check that we can switch of auto-recentering in iteration."""
        sgs = SWIFTGalaxies(
            toysnap["toysnap_filename"],
            ToyHF(snapfile=toysnap["toysnap_filename"], index=[0, 1]),
            auto_recentre=False,
        )
        for sg in sgs:
            assert np.allclose(sg.centre, np.zeros(3))

    def test_read_caching(self, toysnap):
        """Check that we can re-use raw data read for a region through _data_server."""
        server = SWIFTGalaxy(
            toysnap["toysnap_filename"],
            ToyHF(snapfile=toysnap["toysnap_filename"], index=0, extra_mask=None),
            auto_recentre=False,
        )
        client1 = SWIFTGalaxy(
            toysnap["toysnap_filename"],
            ToyHF(snapfile=toysnap["toysnap_filename"], index=0),
            _data_server=server,
        )
        client2 = SWIFTGalaxy(
            toysnap["toysnap_filename"],
            ToyHF(snapfile=toysnap["toysnap_filename"], index=0),
            _data_server=server,
        )
        assert client1._data_server is server
        assert client1.gas._internal_dataset._masses is None
        assert server.gas._internal_dataset._masses is None
        client1.gas.masses  # trigger read
        assert client1.gas._internal_dataset._masses is not None
        assert server.gas._internal_dataset._masses is not None
        assert (
            client1.gas._internal_dataset._masses.shape
            != server.gas._internal_dataset._masses.shape
        )
        assert not np.all(server.gas._internal_dataset._masses == 0)
        # set cached data to new values (0's) so that we can make sure that we used it
        server.gas._internal_dataset._masses[...] = 0
        assert client2._data_server is server
        assert client2.gas._internal_dataset._masses is None
        assert server.gas._internal_dataset._masses is not None
        client2.gas.masses  # trigger read: should just mask/transform cached data
        assert client2.gas._internal_dataset._masses is not None
        assert (
            client2.gas._internal_dataset._masses.shape
            != server.gas._internal_dataset._masses.shape
        )
        assert np.all(client2.gas._internal_dataset._masses == 0)


def _n_dm(sg):
    """
    Count dark matter particles.

    Defined at module level so that it can be pickled and sent to worker processes.

    Parameters
    ----------
    sg : :class:`~swiftgalaxy.reader.SWIFTGalaxy`
        The galaxy to count particles for.

    Returns
    -------
    :obj:`int`
        Number of dark matter particles.
    """
    return int(sg.dark_matter.coordinates.shape[0])


def _scaled_n_dm(sg, factor, offset=0.0):
    """
    Count dark matter particles, with extra arguments, to check argument routing.

    Parameters
    ----------
    sg : :class:`~swiftgalaxy.reader.SWIFTGalaxy`
        The galaxy to count particles for.

    factor : :obj:`float`
        Multiplied into the result.

    offset : :obj:`float` (optional), default: ``0.0``
        Added to the result.

    Returns
    -------
    :obj:`float`
        The scaled particle count.
    """
    return _n_dm(sg) * factor + offset


def _assorted_types(sg):
    """
    Return a mixture of container types, to check that results survive transport.

    Parameters
    ----------
    sg : :class:`~swiftgalaxy.reader.SWIFTGalaxy`
        The galaxy to summarise.

    Returns
    -------
    :obj:`dict`
        A dictionary containing a tuple, an array and ``None``.
    """
    return {
        "pair": (_n_dm(sg), "label"),
        "array": np.arange(3) + _n_dm(sg),
        "nothing": None,
    }


def _sum_x(sg):
    """
    Sum of x coordinates over all present particle types.

    Sensitive to the coordinate frame, but unlike a mean it stays finite when a
    particle type is empty. Caesar galaxies, for instance, contain no dark matter at
    all, and a mean over those would be NaN in both the serial and parallel results,
    which compare unequal and would mask a real disagreement.

    Parameters
    ----------
    sg : :class:`~swiftgalaxy.reader.SWIFTGalaxy`
        The galaxy to summarise.

    Returns
    -------
    :obj:`float`
        Summed x coordinate, in the galaxy's length units.
    """
    total = 0.0
    for particle_type in ("dark_matter", "stars", "gas"):
        if not hasattr(sg, particle_type):
            continue
        xyz = getattr(sg, particle_type).coordinates
        if xyz.shape[0]:
            total += float(np.sum(xyz.to_value(xyz.units)[:, 0]))
    return total


def _always_raises(sg):
    """
    Raise an error, to check that worker exceptions reach the caller.

    Parameters
    ----------
    sg : :class:`~swiftgalaxy.reader.SWIFTGalaxy`
        Ignored.

    Raises
    ------
    RuntimeError
        Always.
    """
    raise RuntimeError("failure inside worker")


def _snapshot_dir(hf_multi, tmp_path_factory):
    """
    Locate the directory holding the toy snapshot for a halo catalogue.

    Parameters
    ----------
    hf_multi : :class:`~swiftgalaxy.halo_catalogues._HaloCatalogue`
        A halo catalogue containing several targets.

    tmp_path_factory : :class:`pytest.TempPathFactory`
        Factory for temporary directories.

    Returns
    -------
    :class:`pathlib.Path`
        The directory containing the snapshot.
    """
    if isinstance(hf_multi, SOAP):
        return hf_multi.soap_file.parent
    elif isinstance(hf_multi, Caesar):
        return hf_multi.caesar_file.parent
    elif isinstance(hf_multi, Velociraptor):
        return Path(hf_multi.velociraptor_files["properties"]).parent
    tp = tmp_path_factory.mktemp(_toysnap_filename.parent)
    _create_toysnap(snapfile=tp / _toysnap_filename.name)
    return tp


def _snapshot_file(hf_multi, tmp_path_factory):
    """
    Locate the snapshot file to open for a halo catalogue.

    Parameters
    ----------
    hf_multi : :class:`~swiftgalaxy.halo_catalogues._HaloCatalogue`
        A halo catalogue containing several targets.

    tmp_path_factory : :class:`pytest.TempPathFactory`
        Factory for temporary directories.

    Returns
    -------
    :class:`pathlib.Path`
        The snapshot (or virtual snapshot) file.
    """
    tp = _snapshot_dir(hf_multi, tmp_path_factory)
    if isinstance(hf_multi, SOAP):
        return tp / _toysoap_virtual_snapshot_filename.name
    return tp / _toysnap_filename.name


def _sgs_for(hf_multi, tmp_path_factory, **kwargs):
    """
    Build a :class:`~swiftgalaxy.iterator.SWIFTGalaxies` for a multi-target catalogue.

    Parameters
    ----------
    hf_multi : :class:`~swiftgalaxy.halo_catalogues._HaloCatalogue`
        A halo catalogue containing several targets.

    tmp_path_factory : :class:`pytest.TempPathFactory`
        Factory for temporary directories.

    **kwargs : :obj:`dict`
        Extra keyword arguments for
        :class:`~swiftgalaxy.iterator.SWIFTGalaxies`.

    Returns
    -------
    :class:`~swiftgalaxy.iterator.SWIFTGalaxies`
        Iterator over the catalogue's targets.
    """
    return SWIFTGalaxies(_snapshot_file(hf_multi, tmp_path_factory), hf_multi, **kwargs)


class TestParallelMapEquivalence:
    """Check that ``map(nproc>1)`` agrees with serial evaluation."""

    @pytest.mark.parametrize("nproc", [2, 3])
    def test_parallel_matches_serial(self, tmp_path_factory, hf_multi, nproc):
        """Check that parallel results equal serial results for several worker counts."""
        sgs = _sgs_for(hf_multi, tmp_path_factory)
        assert sgs.map(_n_dm, nproc=nproc) == sgs.map(_n_dm)

    def test_nproc_one_matches_default(self, tmp_path_factory, hf_multi):
        """Check that an explicit ``nproc=1`` is the same as omitting it."""
        sgs = _sgs_for(hf_multi, tmp_path_factory)
        assert sgs.map(_n_dm, nproc=1) == sgs.map(_n_dm)

    def test_more_workers_than_regions(self, tmp_path_factory, hf_multi):
        """Check that asking for more workers than there is work is harmless."""
        sgs = _sgs_for(hf_multi, tmp_path_factory)
        assert sgs.map(_n_dm, nproc=6) == sgs.map(_n_dm)

    def test_repeated_calls_are_stable(self, tmp_path_factory, hf_multi):
        """Check that mapping twice in a row gives the same answer both times."""
        sgs = _sgs_for(hf_multi, tmp_path_factory)
        first = sgs.map(_n_dm, nproc=2)
        assert sgs.map(_n_dm, nproc=2) == first

    def test_serial_iteration_still_works_after_parallel(
        self, tmp_path_factory, hf_multi
    ):
        """Check that a parallel map leaves the object usable for serial iteration."""
        sgs = _sgs_for(hf_multi, tmp_path_factory)
        sgs.map(_n_dm, nproc=2)
        assert [_n_dm(sg) for sg in sgs] == [
            sgs.map(_n_dm)[i] for i in sgs.iteration_order
        ]


class TestParallelMapOrdering:
    """Check that results follow the user's input order, not the iteration order."""

    def test_input_order_preserved(
        self, tmp_path_factory, hf_multi_forwards_and_backwards
    ):
        """Check that reversing the target list reverses the results."""
        hf_forwards, hf_backwards = hf_multi_forwards_and_backwards
        forwards = _sgs_for(hf_forwards, tmp_path_factory).map(_n_dm, nproc=2)
        backwards = _sgs_for(hf_backwards, tmp_path_factory).map(_n_dm, nproc=2)
        assert forwards == backwards[::-1]

    def test_order_independent_of_iteration_order(self, tmp_path_factory, hf_multi):
        """Check that results are ordered by input even when iteration is reordered."""
        sgs = _sgs_for(hf_multi, tmp_path_factory)
        serial = sgs.map(_n_dm)
        parallel = sgs.map(_n_dm, nproc=3)
        assert parallel == serial
        assert len(parallel) == len(sgs.iteration_order)


class TestParallelMapArguments:
    """Check that extra arguments reach the galaxy that they belong to."""

    def test_args_and_kwargs(self, tmp_path_factory, hf_multi):
        """Check that args and kwargs are routed per-target in parallel."""
        sgs = _sgs_for(hf_multi, tmp_path_factory)
        ntargets = len(sgs.iteration_order)
        args = [(i + 1,) for i in range(ntargets)]
        kwargs = [dict(offset=10.0 * i) for i in range(ntargets)]
        assert sgs.map(_scaled_n_dm, args=args, kwargs=kwargs, nproc=2) == sgs.map(
            _scaled_n_dm, args=args, kwargs=kwargs
        )

    def test_args_only(self, tmp_path_factory, hf_multi):
        """Check that positional arguments alone are routed correctly."""
        sgs = _sgs_for(hf_multi, tmp_path_factory)
        args = [(i + 1,) for i in range(len(sgs.iteration_order))]
        assert sgs.map(_scaled_n_dm, args=args, nproc=2) == sgs.map(
            _scaled_n_dm, args=args
        )

    def test_kwargs_only(self, tmp_path_factory, hf_multi):
        """Check that keyword arguments alone are routed correctly."""
        sgs = _sgs_for(hf_multi, tmp_path_factory)
        ntargets = len(sgs.iteration_order)
        args = [(1.0,)] * ntargets
        kwargs = [dict(offset=100.0 * i) for i in range(ntargets)]
        assert sgs.map(_scaled_n_dm, args=args, kwargs=kwargs, nproc=2) == sgs.map(
            _scaled_n_dm, args=args, kwargs=kwargs
        )


class TestParallelMapReturnValues:
    """Check that non-trivial return values survive the trip back from a worker."""

    def test_container_return_values(self, tmp_path_factory, hf_multi):
        """Check that dicts, tuples, arrays and ``None`` come back intact."""
        sgs = _sgs_for(hf_multi, tmp_path_factory)
        serial = sgs.map(_assorted_types)
        parallel = sgs.map(_assorted_types, nproc=2)
        assert len(serial) == len(parallel)
        for expected, got in zip(serial, parallel):
            assert expected["pair"] == got["pair"]
            assert expected["nothing"] is None and got["nothing"] is None
            assert np.array_equal(expected["array"], got["array"])


class TestParallelMapEdgeCases:
    """Check target lists of unusual length, and invalid arguments."""

    def test_single_target(self, tmp_path_factory, hf_multi_onetarget):
        """Check that a one-target catalogue works in parallel."""
        sgs = _sgs_for(hf_multi_onetarget, tmp_path_factory)
        parallel = sgs.map(_n_dm, nproc=2)
        assert len(parallel) == 1
        assert parallel == sgs.map(_n_dm)

    def test_zero_targets(self, tmp_path_factory, hf_multi_zerotarget):
        """Check that an empty target list gives an empty result, not an error."""
        sgs = _sgs_for(hf_multi_zerotarget, tmp_path_factory)
        assert sgs.map(_n_dm, nproc=2) == []

    @pytest.mark.parametrize("nproc", [0, -1])
    def test_invalid_nproc(self, tmp_path_factory, hf_multi, nproc):
        """Check that a nonsensical number of processes is rejected."""
        sgs = _sgs_for(hf_multi, tmp_path_factory)
        with pytest.raises(ValueError, match="nproc"):
            sgs.map(_n_dm, nproc=nproc)

    def test_worker_exception_propagates(self, tmp_path_factory, hf_multi):
        """Check that an error raised inside a worker reaches the caller."""
        sgs = _sgs_for(hf_multi, tmp_path_factory)
        with pytest.raises(RuntimeError, match="failure inside worker"):
            sgs.map(_always_raises, nproc=2)

    def test_unpicklable_function_rejected(self, tmp_path_factory, hf_multi):
        """Check that a lambda cannot be smuggled into a worker process."""
        sgs = _sgs_for(hf_multi, tmp_path_factory)
        with pytest.raises(Exception):
            sgs.map(lambda sg: _n_dm(sg), nproc=2)


class TestParallelMapCoordinateFrame:
    """Check that a copied coordinate frame survives transport to a worker."""

    def _reference_galaxy(self, hf_multi, tmp_path_factory):
        """
        Build a galaxy with a non-trivial coordinate frame to copy from.

        Parameters
        ----------
        hf_multi : :class:`~swiftgalaxy.halo_catalogues._HaloCatalogue`
            Used only to locate the snapshot file.

        tmp_path_factory : :class:`pytest.TempPathFactory`
            Factory for temporary directories.

        Returns
        -------
        :class:`~swiftgalaxy.reader.SWIFTGalaxy`
            A galaxy that has been rotated and translated.
        """
        snap = _snapshot_file(hf_multi, tmp_path_factory)
        ref = SWIFTGalaxy(
            snap,
            Standalone(
                centre=cosmo_array(
                    [2.0, 2.0, 2.0],
                    u.Mpc,
                    comoving=True,
                    scale_factor=1.0,
                    scale_exponent=1,
                ),
                velocity_centre=cosmo_array(
                    [0.0, 0.0, 0.0],
                    u.km / u.s,
                    comoving=True,
                    scale_factor=1.0,
                    scale_exponent=0,
                ),
                spatial_offsets=cosmo_array(
                    [[-1.0, 1.0]] * 3,
                    u.Mpc,
                    comoving=True,
                    scale_factor=1.0,
                    scale_exponent=1,
                ),
            ),
        )
        ref.rotate(Rotation.from_euler("z", 37, degrees=True))
        ref.translate(
            cosmo_array(
                [0.31, -0.12, 0.05],
                u.Mpc,
                comoving=True,
                scale_factor=1.0,
                scale_exponent=1,
            )
        )
        return ref

    def test_coordinate_frame_from_matches_serial(self, tmp_path_factory, hf_multi):
        """Check that a copied coordinate frame gives the same answer in parallel."""
        ref = self._reference_galaxy(hf_multi, tmp_path_factory)
        sgs = _sgs_for(
            hf_multi, tmp_path_factory, auto_recentre=False, coordinate_frame_from=ref
        )
        assert np.allclose(sgs.map(_sum_x, nproc=2), sgs.map(_sum_x))

    def test_coordinate_frame_actually_applied(self, tmp_path_factory, hf_multi):
        """Check that copying a frame changes the answer, so the test above has teeth."""
        ref = self._reference_galaxy(hf_multi, tmp_path_factory)
        framed = _sgs_for(
            hf_multi, tmp_path_factory, auto_recentre=False, coordinate_frame_from=ref
        ).map(_sum_x, nproc=2)
        unframed = _sgs_for(hf_multi, tmp_path_factory, auto_recentre=False).map(
            _sum_x, nproc=2
        )
        assert not np.allclose(framed, unframed)

    @pytest.mark.parametrize("nproc", [1, 2])
    def test_frame_with_auto_recentre_raises(self, tmp_path_factory, hf_multi, nproc):
        """Check that combining a copied frame with auto_recentre is still rejected."""
        ref = self._reference_galaxy(hf_multi, tmp_path_factory)
        sgs = _sgs_for(
            hf_multi, tmp_path_factory, auto_recentre=True, coordinate_frame_from=ref
        )
        with pytest.raises(ValueError, match="coordinate_frame_from"):
            sgs.map(_sum_x, nproc=nproc)


class TestCoordinateFrameSpec:
    """Unit tests for the picklable stand-in for ``coordinate_frame_from``."""

    def test_spec_is_picklable_and_small(self, tmp_path_factory, hf_multi):
        """Check the spec pickles, unlike the galaxy, and does not carry a registry."""
        ref = TestParallelMapCoordinateFrame()._reference_galaxy(
            hf_multi, tmp_path_factory
        )
        with pytest.raises(TypeError, match="h5py"):
            pickle.dumps(ref)
        blob = pickle.dumps(_CoordinateFrameSpec.from_swift_galaxy(ref))
        # a unyt quantity would drag a unit registry of tens of kilobytes
        assert len(blob) < 5000

    def test_spec_round_trip_preserves_transforms(self, tmp_path_factory, hf_multi):
        """Check that the transforms are unchanged by pickling."""
        ref = TestParallelMapCoordinateFrame()._reference_galaxy(
            hf_multi, tmp_path_factory
        )
        spec = _CoordinateFrameSpec.from_swift_galaxy(ref)
        back = pickle.loads(pickle.dumps(spec))
        assert np.allclose(
            back.coordinate_transform.as_matrix(), spec.coordinate_transform.as_matrix()
        )
        assert np.allclose(
            back.velocity_transform.as_matrix(), spec.velocity_transform.as_matrix()
        )
        assert (back.length_unit, back.time_unit) == (spec.length_unit, spec.time_unit)

    def test_unit_mismatch_raises(self, tmp_path_factory, hf_multi):
        """Check that mismatched internal units are still detected."""
        ref = TestParallelMapCoordinateFrame()._reference_galaxy(
            hf_multi, tmp_path_factory
        )
        spec = _CoordinateFrameSpec.from_swift_galaxy(ref)._replace(
            length_unit="not_the_same_unit"
        )
        with pytest.raises(ValueError, match="don't match"):
            spec.apply_to(ref)


class TestSubsetSpec:
    """Unit tests for describing a halo catalogue as picklable plain data."""

    def test_spec_is_picklable(self, hf_multi):
        """Check that the spec pickles even though the catalogue itself does not."""
        spec = hf_multi._subset_spec([0])
        pickle.dumps(spec)

    def test_rebuilds_an_equivalent_catalogue(self, hf_multi):
        """Check that rebuilding from the spec selects the requested targets."""
        catalogue_class, kwargs = hf_multi._subset_spec([0])
        rebuilt = catalogue_class(**kwargs)
        assert isinstance(rebuilt, type(hf_multi))
        if hf_multi._index_attr is not None:
            assert len(np.atleast_1d(getattr(rebuilt, hf_multi._index_attr))) == 1
        else:  # Standalone carries its centres by value instead of an index
            assert np.atleast_2d(rebuilt._centre).shape[0] == 1

    def test_selects_the_requested_targets(self, hf_multi):
        """Check that a subset picks out the targets asked for, in order."""
        catalogue_class, kwargs = hf_multi._subset_spec([1, 0])
        rebuilt = catalogue_class(**kwargs)
        if hf_multi._index_attr is not None:
            full = np.atleast_1d(np.asarray(getattr(hf_multi, hf_multi._index_attr)))
            got = np.atleast_1d(np.asarray(getattr(rebuilt, hf_multi._index_attr)))
            assert list(got) == [full[1], full[0]]
        else:
            full = np.atleast_2d(hf_multi._centre)
            got = np.atleast_2d(rebuilt._centre)
            assert np.allclose(got[0], full[1]) and np.allclose(got[1], full[0])


class TestWorkerStartMethod:
    """Check the process start method chosen for workers."""

    def test_avoids_fork(self):
        """Check that we do not fork, which is unsafe with open HDF5 handles."""
        context = _worker_context()
        if "forkserver" in multiprocessing.get_all_start_methods():
            assert context.get_start_method() == "forkserver"
        else:
            assert context.get_start_method() in ("spawn", "forkserver")


def _own_index(sg):
    """
    Report the catalogue's own identifier for the galaxy being processed.

    Confirms that a worker rebuilt its catalogue around the target it was actually
    given, rather than silently processing a different one.

    Parameters
    ----------
    sg : :class:`~swiftgalaxy.reader.SWIFTGalaxy`
        The galaxy being processed.

    Returns
    -------
    :obj:`int`
        The catalogue index of this galaxy.
    """
    catalogue = sg.halo_catalogue
    return int(np.atleast_1d(getattr(catalogue, catalogue._index_attr[1:]))[0])


def _own_centre_x(sg):
    """
    Report the x coordinate of the centre the worker's catalogue was built around.

    :class:`~swiftgalaxy.halo_catalogues.Standalone` has no catalogue index, and a
    worker's rebuilt catalogue holds only its own target, so identity is checked by
    centre rather than by position in a list.

    Parameters
    ----------
    sg : :class:`~swiftgalaxy.reader.SWIFTGalaxy`
        The galaxy being processed.

    Returns
    -------
    :obj:`float`
        The x coordinate of this galaxy's centre.
    """
    centre = np.atleast_2d(sg.halo_catalogue.centre)
    return float(centre[0, 0].to_value(centre.units))


def _catalogue_class_name(sg):
    """
    Report the class of the halo catalogue seen inside the worker.

    Parameters
    ----------
    sg : :class:`~swiftgalaxy.reader.SWIFTGalaxy`
        The galaxy being processed.

    Returns
    -------
    :obj:`str`
        Name of the halo catalogue class.
    """
    return type(sg.halo_catalogue).__name__


class TestParallelMapSOAP:
    """Parallel iteration specifics for the SOAP backend."""

    def test_targets_not_mixed_up(self, sgs_soap):
        """Check that each worker processes the SOAP target it was given."""
        assert sgs_soap.map(_own_index, nproc=2) == [0, 1]

    def test_catalogue_available_in_worker(self, sgs_soap):
        """Check that the rebuilt catalogue is a SOAP catalogue in the worker."""
        assert sgs_soap.map(_catalogue_class_name, nproc=2) == ["SOAP", "SOAP"]

    def test_results_match_serial(self, sgs_soap):
        """Check that parallel and serial agree for SOAP."""
        assert sgs_soap.map(_n_dm, nproc=2) == sgs_soap.map(_n_dm)

    def test_centre_types_survive_reconstruction(self, soap_multi):
        """Check that SOAP's two centre-type options are carried to the worker."""
        soap_class, kwargs = soap_multi._subset_spec([0])
        assert kwargs["centre_type"] == soap_multi.centre_type
        assert kwargs["velocity_centre_type"] == soap_multi.velocity_centre_type
        assert Path(kwargs["soap_file"]) == Path(soap_multi.soap_file)
        assert soap_class is SOAP


class TestParallelMapVelociraptor:
    """Parallel iteration specifics for the Velociraptor backend."""

    def test_targets_not_mixed_up(self, sgs_vr):
        """Check that each worker processes the Velociraptor target it was given."""
        assert sgs_vr.map(_own_index, nproc=2) == [0, 1]

    def test_catalogue_available_in_worker(self, sgs_vr):
        """Check that the rebuilt catalogue is a Velociraptor catalogue."""
        assert sgs_vr.map(_catalogue_class_name, nproc=2) == [
            "Velociraptor",
            "Velociraptor",
        ]

    def test_results_match_serial(self, sgs_vr):
        """Check that parallel and serial agree for Velociraptor."""
        assert sgs_vr.map(_n_dm, nproc=2) == sgs_vr.map(_n_dm)

    def test_both_catalogue_files_survive_reconstruction(self, vr_multi):
        """Check that both of Velociraptor's file paths reach the worker.

        Velociraptor is the only backend built from a pair of files, and it can be
        constructed from a filebase instead, so the rebuilt catalogue has to carry the
        resolved pair rather than the filebase it was given.
        """
        vr_class, kwargs = vr_multi._subset_spec([0])
        assert set(kwargs["velociraptor_files"]) == {"properties", "catalog_groups"}
        assert kwargs["velociraptor_files"] == vr_multi.velociraptor_files
        assert isinstance(vr_class(**kwargs), Velociraptor)


class TestParallelMapCaesar:
    """Parallel iteration specifics for the Caesar backend."""

    def test_targets_not_mixed_up(self, sgs_caesar):
        """Check that each worker processes the Caesar target it was given."""
        assert sgs_caesar.map(_own_index, nproc=2) == [0, 1]

    def test_catalogue_available_in_worker(self, sgs_caesar):
        """Check that the rebuilt catalogue is a Caesar catalogue."""
        assert sgs_caesar.map(_catalogue_class_name, nproc=2) == ["Caesar", "Caesar"]

    def test_results_match_serial(self, sgs_caesar):
        """Check that parallel and serial agree for both Caesar group types."""
        assert sgs_caesar.map(_n_dm, nproc=2) == sgs_caesar.map(_n_dm)

    def test_group_type_reaches_the_worker(self, sgs_caesar):
        """Check that halo/galaxy selection actually changes what a worker sees.

        Caesar galaxies contain no dark matter while haloes do, so the particle counts
        distinguish the two. If ``group_type`` were lost when the catalogue was rebuilt,
        workers would quietly analyse the wrong kind of object and this would catch it.
        """
        counts = sgs_caesar.map(_n_dm, nproc=2)
        if sgs_caesar.halo_catalogue.group_type == "galaxy":
            assert set(counts) == {0}
        else:
            assert max(counts) > 0

    def test_group_type_survives_reconstruction(self, caesar_multi):
        """Check that the group type is present in the reconstruction spec."""
        caesar_class, kwargs = caesar_multi._subset_spec([0])
        assert kwargs["group_type"] == caesar_multi.group_type
        assert caesar_class(**kwargs).group_type == caesar_multi.group_type


class TestParallelMapStandalone:
    """Parallel iteration specifics for the Standalone backend."""

    def test_targets_not_mixed_up(self, sgs_sa):
        """Check that each worker processes the centre it was given, in input order."""
        centres = np.atleast_2d(sgs_sa.halo_catalogue._centre)
        expected = [float(c.to_value(centres.units)) for c in centres[:, 0]]
        assert np.allclose(sgs_sa.map(_own_centre_x, nproc=2), expected)

    def test_results_match_serial(self, sgs_sa):
        """Check that parallel and serial agree for Standalone."""
        assert sgs_sa.map(_n_dm, nproc=2) == sgs_sa.map(_n_dm)

    def test_centres_passed_by_value(self, sa_multi):
        """Check that centres are sliced and sent, there being no file to reopen."""
        _, kwargs = sa_multi._subset_spec([1])
        assert "centre" in kwargs and "velocity_centre" in kwargs
        assert np.allclose(
            np.atleast_2d(kwargs["centre"])[0], np.atleast_2d(sa_multi._centre)[1]
        )
        assert np.allclose(
            np.atleast_2d(kwargs["velocity_centre"])[0],
            np.atleast_2d(sa_multi._velocity_centre)[1],
        )

    def test_centres_keep_their_units(self, sa_multi):
        """Check that the sliced centres remain cosmo_arrays with units."""
        _, kwargs = sa_multi._subset_spec([0])
        assert isinstance(kwargs["centre"], cosmo_array)
        assert kwargs["centre"].units == np.atleast_2d(sa_multi._centre).units
