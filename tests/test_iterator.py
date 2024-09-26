import pytest
import numpy as np
import unyt as u
from unyt.testing import assert_allclose_units
from toysnap import (
    toysnap_filename,
    ToyHF,
    present_particle_types,
    toysoap_virtual_snapshot_filename,
)
from swiftsimio.objects import cosmo_array, cosmo_factor, a
from swiftgalaxy.reader import SWIFTGalaxy
from swiftgalaxy.iterator import SWIFTGalaxies
from swiftgalaxy.halo_catalogues import Standalone, SOAP
from swiftsimio import mask


class TestSWIFTGalaxies:

    def test_eval_sparse_optimized_solution(self, toysnap):
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
        hf_multi.extra_mask = extra_mask
        sgs = SWIFTGalaxies(
            (
                toysoap_virtual_snapshot_filename
                if isinstance(hf_multi, SOAP)
                else toysnap_filename
            ),
            hf_multi,
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
