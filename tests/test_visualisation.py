"""Tests of integration of swiftsimio visualisation tools with swiftgalaxy."""

import pytest
import numpy as np
import unyt as u
from swiftsimio import cosmo_array, cosmo_quantity
from swiftsimio.visualisation.projection import project_gas
from swiftsimio.visualisation.slice import slice_gas
from swiftsimio.visualisation.volume_render import render_gas
from swiftgalaxy import SWIFTGalaxy


@pytest.mark.parametrize("periodic", [False, True])
class TestRecenteredVisualisation:
    """Tests for visualisation functions on swiftgalaxies with recentered coordinates."""

    @pytest.mark.parametrize("z_cut", [False, True])
    def test_recentered_projection(self, sg_autorecentre_off, z_cut, periodic):
        """We should be able to make the same projections whether we recentered or not."""
        sg = SWIFTGalaxy(
            sg_autorecentre_off.snapshot_filename, sg_autorecentre_off.halo_catalogue
        )
        z_limits = [1.9, 2.1] if z_cut else []
        recentered_z_limits = [-0.1, 0.1] if z_cut else []
        kwargs = {
            "resolution": 4,
            "project": "masses",
            "parallel": True,
            "periodic": periodic,
        }
        ref_img = project_gas(
            sg_autorecentre_off,
            region=cosmo_array(
                [1.5, 2.5, 1.5, 2.5] + z_limits,
                u.Mpc,
                comoving=True,
                scale_factor=1,
                scale_exponent=1,
            ),
            **kwargs,
        )
        recentered_img = project_gas(
            sg,
            region=cosmo_array(
                [-0.5, 0.5, -0.5, 0.5] + recentered_z_limits,
                u.Mpc,
                comoving=True,
                scale_factor=1,
                scale_exponent=1,
            ),
            **kwargs,
        )
        assert (ref_img > 0).any()
        assert np.allclose(ref_img, recentered_img)

    def test_recentered_slice(self, sg_autorecentre_off, periodic):
        """We should be able to make the same slice whether we recentered or not."""
        sg = SWIFTGalaxy(
            sg_autorecentre_off.snapshot_filename, sg_autorecentre_off.halo_catalogue
        )
        z_slice = cosmo_quantity(
            1.95, u.Mpc, comoving=True, scale_factor=1, scale_exponent=1
        )
        recentered_z_slice = cosmo_quantity(
            -0.05, u.Mpc, comoving=True, scale_factor=1, scale_exponent=1
        )
        kwargs = {
            "resolution": 4,
            "project": "masses",
            "parallel": True,
            "periodic": periodic,
        }
        ref_img = slice_gas(
            sg_autorecentre_off,
            z_slice=z_slice,
            region=cosmo_array(
                [1.5, 2.5, 1.5, 2.5],
                u.Mpc,
                comoving=True,
                scale_factor=1,
                scale_exponent=1,
            ),
            **kwargs,
        )
        recentered_img = slice_gas(
            sg,
            z_slice=recentered_z_slice,
            region=cosmo_array(
                [-0.5, 0.5, -0.5, 0.5],
                u.Mpc,
                comoving=True,
                scale_factor=1,
                scale_exponent=1,
            ),
            **kwargs,
        )
        assert (ref_img > 0).any()
        assert np.allclose(ref_img, recentered_img)

    def test_recentered_volume_render(self, sg_autorecentre_off, periodic):
        """We should be able to make the same rendering whether we recentered or not."""
        sg = SWIFTGalaxy(
            sg_autorecentre_off.snapshot_filename, sg_autorecentre_off.halo_catalogue
        )
        kwargs = {
            "resolution": 4,
            "project": "masses",
            "parallel": True,
            "periodic": periodic,
        }
        ref_img = render_gas(
            sg_autorecentre_off,
            region=cosmo_array(
                [1.5, 2.5, 1.5, 2.5, 1.5, 2.5],
                u.Mpc,
                comoving=True,
                scale_factor=1,
                scale_exponent=1,
            ),
            **kwargs,
        )
        recentered_img = render_gas(
            sg,
            region=cosmo_array(
                [-0.5, 0.5, -0.5, 0.5, -0.5, 0.5],
                u.Mpc,
                comoving=True,
                scale_factor=1,
                scale_exponent=1,
            ),
            **kwargs,
        )
        assert (ref_img > 0).any()
        assert np.allclose(ref_img, recentered_img)
