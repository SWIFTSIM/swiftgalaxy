import pytest
import numpy as np
import unyt as u
from unyt.testing import assert_allclose_units
from toysnap import (
    toysnap_filename,
    ToyHF,
    present_particle_types,
    toysoap_virtual_snapshot_filename,
    toysoap_filename,
)
from swiftsimio.objects import cosmo_array, cosmo_factor, a
from swiftgalaxy.reader import SWIFTGalaxy
from swiftgalaxy.iterator import SWIFTGalaxies
from swiftgalaxy.halo_catalogues import Standalone, SOAP
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
            toysnap_filename,
            Standalone(
                centre=cosmo_array(
                    [[2.5, 5.0, 5.0], [7.5, 5.0, 5.0]],
                    u.Mpc,
                    comoving=True,
                    cosmo_factor=cosmo_factor(a**1, 1.0),
                ),
                velocity_centre=cosmo_array(
                    [[0, 0, 0] * 2],
                    u.km / u.s,
                    comoving=True,
                    cosmo_factor=cosmo_factor(a**0, 1.0),
                ),
                spatial_offsets=cosmo_array(
                    [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]],
                    u.Mpc,
                    comoving=True,
                    cosmo_factor=cosmo_factor(a**1, 1.0),
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
                cosmo_factor=cosmo_factor(a**1, 1.0),
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
            toysnap_filename,
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
                    cosmo_factor=cosmo_factor(a**1, 1.0),
                ),
                velocity_centre=cosmo_array(
                    [[0, 0, 0] * 10],
                    u.km / u.s,
                    comoving=True,
                    cosmo_factor=cosmo_factor(a**0, 1.0),
                ),
                spatial_offsets=cosmo_array(
                    [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]],
                    u.Mpc,
                    comoving=True,
                    cosmo_factor=cosmo_factor(a**1, 1.0),
                ),
            ),
            preload={  # just to keep warnings quiet
                "gas.particle_ids",
                "dark_matter.particle_ids",
                "stars.particle_ids",
                "black_holes.particle_ids",
            },
        )
        dense_solution = sgs._dense_optimized_solution
        assert_allclose_units(
            dense_solution["regions"],
            cosmo_array(
                np.array(
                    [
                        [[0.005, 4.995], [0.01, 9.99], [0.01, 9.99]],
                        [[5.005, 9.995], [0.01, 9.99], [0.01, 9.99]],
                        [[2.505, 7.495], [5.01, 14.99], [5.01, 14.99]],
                        [[7.505, 12.495], [5.01, 14.99], [5.01, 14.99]],
                    ]
                ),
                u.Mpc,
                comoving=True,
                cosmo_factor=cosmo_factor(a**1, 1.0),
            ),
            atol=0.001 * u.Mpc,
        )
        assert len(dense_solution["region_target_indices"]) == 4
        for ds, expected in zip(
            dense_solution["region_target_indices"],
            [
                np.array([0]),
                np.array([1, 2, 3, 4, 5, 7, 8]),
                np.array([6]),
                np.array([9]),
            ],
        ):
            assert np.allclose(ds, expected, atol=0)
        assert dense_solution["cost_min"] == 4
        assert dense_solution["cost_max"] == 32
        for k in sgs._solution.keys():
            if isinstance(sgs._solution[k], np.integer):
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
                toysnap_filename,
                ToyHF(
                    snapfile=toysnap_filename, index=sg_from_sgs.halo_catalogue.index
                ),
            )
            for ptype in present_particle_types.values():
                assert np.all(
                    getattr(sg_from_sgs._extra_mask, ptype)
                    == getattr(sg._extra_mask, ptype)
                )
            count += 1
        assert count == len(sgs.halo_catalogue.index)

    @pytest.mark.parametrize("extra_mask", ["bound_only", None])
    def test_preload(self, toysnap_withfof, hf_multi, extra_mask):
        """
        Make sure that data that we ask to have pre-loaded is actually pre-loaded.
        """
        hf_multi.extra_mask = extra_mask
        sgs = SWIFTGalaxies(
            (
                toysoap_virtual_snapshot_filename
                if isinstance(hf_multi, SOAP)
                else toysnap_filename
            ),
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
        for preload_field in sgs.halo_catalogue._get_preload_fields(sgs._server):
            ptype, field = preload_field.split(".")
            assert (
                getattr(getattr(sgs._server, ptype)._particle_dataset, f"_{field}")
                is not None
            )

    def test_warn_on_no_preload(self, toysnap):
        """
        Check that we warn users if they don't specify anything to pre-load since this
        probably indicates that they're using the SWIFTGalaxies class inefficiently.
        """
        with pytest.warns(RuntimeWarning, match="No data specified to preload"):
            SWIFTGalaxies(
                toysnap_filename,
                ToyHF(index=[0, 1]),
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
                toysnap_filename,
                ToyHF(index=[0, 0]),
            )

    def test_map(self, toysnap_withfof, hf_multi):
        """
        Check that the map method returns results in the same order as the input target
        list. We're careful in this test to make sure that the iteration order is
        different from the input list order.
        """
        sgs = SWIFTGalaxies(
            (
                toysoap_virtual_snapshot_filename
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

    def test_arbitrary_index_ordering(self, toysnap_withfof, hf_multi):
        """
        Check that SWIFTGalaxies gives consistent results for any order of target objects.

        Especially important for velociraptor where some logic had to be added to avoid
        hdf5 complaining about an unsorted list of indices to read from file.
        """

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

        sgs_forwards = SWIFTGalaxies(
            (
                toysoap_virtual_snapshot_filename
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
        if isinstance(hf_multi, Standalone):
            hf_multi._centre = hf_multi._centre[::-1]
            hf_multi._velocity_centre = hf_multi._velocity_centre[::-1]
        else:
            setattr(
                hf_multi,
                hf_multi._index_attr,
                getattr(hf_multi, hf_multi._index_attr)[::-1],
            )
        sgs_backwards = SWIFTGalaxies(
            (
                toysoap_virtual_snapshot_filename
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
        map_backwards = sgs_backwards.map(f)
        if isinstance(hf_multi, Standalone):
            # because the reversed catalogue compares to the reversed catalogue in f,
            # we get the reverse-of-the-reverse in map_backwards, which should match
            # map_forwards
            assert map_forwards == map_backwards
            return
        assert map_forwards == map_backwards[::-1]

    def test_args_kwargs_to_map(self, sgs):
        """
        Make sure that we can pass extra args & kwargs to a function given to map.
        """
        extra_arg, extra_kwarg = "foo", "bar"

        def f(sg, extra_arg, extra_kwarg=None):
            return extra_arg, extra_kwarg

        result = sgs.map(f, (extra_arg,), dict(extra_kwarg=extra_kwarg))
        assert result == [(extra_arg, extra_kwarg)] * sgs.halo_catalogue.count

    def test_soap_target_order_consistency(
        self, toysnap_withfof, toysoap_with_virtual_snapshot
    ):
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
            SOAP(soap_file=toysoap_filename, soap_index=0),
            SOAP(soap_file=toysoap_filename, soap_index=1),
        ]
        soap_both = SOAP(soap_file=toysoap_filename, soap_index=[1, 0])
        sgs_individual = [
            SWIFTGalaxy(toysoap_virtual_snapshot_filename, soap) for soap in soaps
        ]
        sgs = SWIFTGalaxies(
            toysoap_virtual_snapshot_filename,
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
