import pytest
import re
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
from swiftgalaxy.iterator import SWIFTGalaxies
from swiftgalaxy.halo_catalogues import Standalone, SOAP, Velociraptor, Caesar
from swiftsimio import mask


class TestSWIFTGalaxies:

    def test_eval_sparse_optimized_solution(self, toysnap):
        """
        Check that the sparse solution is chosen when optimal and matches expectations
        for a case that we can work out by hand.
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
            preload={  # just to keep warnings quiet
                "gas.particle_ids",
                "dark_matter.particle_ids",
                "stars.particle_ids",
                "black_holes.particle_ids",
            },
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
        Check that the dense solution is chosen when optimal and matches expectations
        for a case that we can work out by hand.
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
            preload={  # just to keep warnings quiet
                "gas.particle_ids",
                "dark_matter.particle_ids",
                "stars.particle_ids",
                "black_holes.particle_ids",
            },
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
                print(k, sgs._solution[k], dense_solution[k])
                assert_allclose_units(
                    sgs._solution[k], dense_solution[k], atol=0.001 * u.Mpc
                )

    def test_iteration_order(self, sgs):
        """
        Check that the iteration order agrees with that computed in the two iteration
        optimizations.
        """
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
        Check that we iterate over the right number of SWIFTGalaxy objects and that
        they behave like a SWIFTGalaxy created on its own for each target.
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
                assert np.all(
                    getattr(sg_from_sgs._extra_mask, ptype).mask
                    == getattr(sg._extra_mask, ptype).mask
                )
            count += 1
        assert count == len(sgs.halo_catalogue.index)

    @pytest.mark.parametrize("extra_mask", ["bound_only", None])
    def test_preload(self, tmp_path_factory, hf_multi, extra_mask):
        """
        Make sure that data that we ask to have pre-loaded is actually pre-loaded.
        """
        if isinstance(hf_multi, SOAP):
            tp = hf_multi.soap_file.parent
        elif isinstance(hf_multi, Caesar):
            tp = hf_multi.caesar_file.parent
        elif isinstance(hf_multi, Velociraptor):
            tp = Path(hf_multi.velociraptor_files["properties"]).parent
        else:
            tp = tmp_path_factory.mktemp(_toysnap_filename.parent)
        toysnap_filename = tp / _toysnap_filename.name
        _create_toysnap(snapfile=toysnap_filename, withfof=isinstance(hf_multi, SOAP))
        hf_multi.extra_mask = extra_mask
        sgs = SWIFTGalaxies(
            toysnap_filename,
            hf_multi,
            preload={
                "gas.particle_ids",
                "dark_matter.particle_ids",
                "stars.particle_ids",
                "black_holes.particle_ids",
            },
        )
        region_mask = mask(toysnap_filename)
        region_mask.constrain_spatial(sgs._solution["regions"][0])
        sgs._start_server(region_mask)
        sgs._preload()
        for preload_field in sgs.preload:
            ptype, field = preload_field.split(".")
            assert (
                getattr(getattr(sgs._server, ptype)._particle_dataset, f"_{field}")
                is not None
            )
        _remove_toysnap(snapfile=toysnap_filename)

    def test_warn_on_no_preload(self, toysnap):
        """
        Check that we warn users if they don't specify anything to pre-load since this
        probably indicates that they're using the SWIFTGalaxies class inefficiently.
        """
        with pytest.warns(RuntimeWarning, match="No data specified to preload"):
            SWIFTGalaxies(
                toysnap["toysnap_filename"],
                ToyHF(toysnap["toysnap_filename"], index=[0, 1]),
            )

    def test_warn_on_read_not_preloaded(self, sgs):
        """
        Check that we warn users when data is loaded while iterating over a SWIFTGalaxies
        since this probably indicates that they're using the class inefficiently.
        """
        assert "gas.coordinates" not in sgs.preload
        for sg in sgs:
            with pytest.warns(RuntimeWarning, match="should it be preloaded"):
                sg.gas.coordinates

    def test_exception_on_repeated_targets(self, toysnap):
        """
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
        Check that the map method returns results in the same order as the input target
        list. We're careful in this test to make sure that the iteration order is
        different from the input list order.
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
            preload={  # just to keep warnings quiet
                "gas.particle_ids",
                "dark_matter.particle_ids",
                "stars.particle_ids",
                "black_holes.particle_ids",
            },
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
            preload={  # just to keep warnings quiet
                "gas.particle_ids",
                "dark_matter.particle_ids",
                "stars.particle_ids",
                "black_holes.particle_ids",
            },
        )
        map_forwards = sgs_forwards.map(f)
        sgs_backwards = SWIFTGalaxies(
            (
                tp / _toysoap_virtual_snapshot_filename.name
                if isinstance(hf_multi, SOAP)
                else toysnap_filename
            ),
            hf_multi_backwards,
            preload={  # just to keep warnings quiet
                "gas.particle_ids",
                "dark_matter.particle_ids",
                "stars.particle_ids",
                "black_holes.particle_ids",
            },
        )
        map_backwards = sgs_backwards.map(f)
        assert np.allclose(map_forwards, map_backwards[::-1])
        if isinstance(hf_multi, Standalone):
            _remove_toysnap(snapfile=toysnap_filename)

    def test_args_kwargs_to_map(self, sgs):
        """
        Make sure that we can pass extra args & kwargs to a function given to map.
        """
        extra_arg = [("foo",), ("bar",)]
        extra_kwarg = [dict(extra_kwarg="spam"), dict(extra_kwarg="eggs")]

        def f(sg, extra_arg, extra_kwarg=None):
            return extra_arg, extra_kwarg

        result = sgs.map(f, args=extra_arg, kwargs=extra_kwarg)
        assert result == [("foo", "spam"), ("bar", "eggs")]

    def test_soap_target_order_consistency(self, toysoap_with_virtual_snapshot):
        """
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
            preload={"black_holes.masses"},  # just keep warnings quiet
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
        Check if we can initialize a halo_catalogue in multi-galaxy mode with
        ordered containers that are not a list.
        """
        toysnap_filename = toysnap_withfof["toysnap_filename"]
        tp = toysnap_filename.parent
        if hf_type == "soap":
            pytest.importorskip("compression")
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
                    preload={"gas.masses"},  # just to keep warnings quiet
                )
            elif hf_type == "vr":
                sgs = SWIFTGalaxies(
                    toysnap_filename,
                    Velociraptor(
                        velociraptor_filebase=toyvr_filebase, halo_index=init_indices
                    ),
                    preload={"gas.masses"},
                )
            elif hf_type == "caesar_galaxy":
                sgs = SWIFTGalaxies(
                    toysnap_filename,
                    Caesar(
                        toycaesar_filename,
                        group_type="galaxy",
                        group_index=init_indices,
                    ),
                    preload={"gas.masses"},
                )
            elif hf_type == "caesar_halo":
                sgs = SWIFTGalaxies(
                    toysnap_filename,
                    Caesar(
                        toycaesar_filename, group_type="halo", group_index=init_indices
                    ),
                    preload={"gas.masses"},
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
        Make sure we don't crash with zero targets. Instead iterate over zero elements.
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
            preload={  # just to keep warnings quiet
                "gas.particle_ids",
                "dark_matter.particle_ids",
                "stars.particle_ids",
                "black_holes.particle_ids",
            },
        )
        for sg in sgs:  # should not crash by iterating
            # but should not reach this:
            raise RuntimeError("Supposed to iterate over 0 elements!")

        def f(sg):
            raise RuntimeError  # we should never call this

        map_result = sgs.map(f)
        assert len(map_result) == 0

    def test_one_target(self, tmp_path_factory, hf_multi_onetarget):
        """
        Make sure that we don't crash with a single target. Instead iterate over it.
        """
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
            preload={  # just to keep warnings quiet
                "gas.particle_ids",
                "dark_matter.particle_ids",
                "stars.particle_ids",
                "black_holes.particle_ids",
            },
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
        """
        Check that trying to use a non-iterable catalogue raises.
        """
        with pytest.raises(
            ValueError, match="halo_catalogue target list is not iterable"
        ):
            SWIFTGalaxies(
                toysnap["toysnap_filename"],
                ToyHF(snapfile=toysnap["toysnap_filename"], index=0),
            )

    def test_invalid_iteration_mode(self, toysnap):
        """
        Check that giving an invalid iteration mode raises.
        """
        with pytest.raises(ValueError, match="optimize_iteration must be one of"):
            SWIFTGalaxies(
                toysnap["toysnap_filename"],
                ToyHF(snapfile=toysnap["toysnap_filename"], index=[0, 1]),
                preload=("gas.coordinates",),  # just keep warning quiet
                optimize_iteration="not_implemented",
            )

    def test_coordinate_frame_from_and_auto_recentre_invalid(self, toysnap):
        """
        Check that inheriting a coordinate frame and auto-recentering are incompatible.
        """
        sg = SWIFTGalaxy(
            toysnap["toysnap_filename"],
            ToyHF(snapfile=toysnap["toysnap_filename"], index=0),
        )
        sgs = SWIFTGalaxies(
            toysnap["toysnap_filename"],
            ToyHF(snapfile=toysnap["toysnap_filename"], index=[0, 1]),
            preload=("gas.coordinates",),  # just keep warning quiet
            auto_recentre=True,
            coordinate_frame_from=sg,
        )
        with pytest.raises(
            ValueError, match="Cannot use coordinate_frame_from with auto_recentre"
        ):
            for sg_i in sgs:
                pass

    def test_coordinate_frame_from_in_iteration(self, toysnap):
        """
        Check that we can borrow a coordinate frame when iterating.
        """
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
            preload=("gas.coordinates",),  # just keep warning quiet
            auto_recentre=False,
            coordinate_frame_from=sg,
        )
        for sg_i in sgs:
            assert np.allclose(sg.halo_catalogue.centre - translation, sg_i.centre)

    def test_internal_units_mismatch_in_coordinate_frame_from(self, toysnap):
        """
        Check that incompatible internal units raises.
        """
        sg = SWIFTGalaxy(
            toysnap["toysnap_filename"],
            ToyHF(snapfile=toysnap["toysnap_filename"], index=0),
        )
        sg.metadata.units.length = 1 * u.kpc
        sgs = SWIFTGalaxies(
            toysnap["toysnap_filename"],
            ToyHF(snapfile=toysnap["toysnap_filename"], index=[0, 1]),
            preload=("gas.coordinates",),  # just keep warning quiet
            auto_recentre=False,
            coordinate_frame_from=sg,
        )
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Internal units (length and time) of coordinate_frame_from don't "
                "match."
            ),
        ):
            for sg_i in sgs:
                pass

    def test_auto_recentre_off(self, toysnap):
        """
        Check that we can switch of auto-recentering in iteration.
        """
        sgs = SWIFTGalaxies(
            toysnap["toysnap_filename"],
            ToyHF(snapfile=toysnap["toysnap_filename"], index=[0, 1]),
            preload=("gas.coordinates",),  # just keep warning quiet
            auto_recentre=False,
        )
        for sg in sgs:
            assert np.allclose(sg.centre, np.zeros(3))
