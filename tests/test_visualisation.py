import pytest
import numpy as np
import unyt as u
from swiftsimio import cosmo_array
from swiftsimio.visualisation.projection import project_pixel_grid


@pytest.mark.parametrize("z_slice", [False, True])
def test_recentered_visualisation(sg, sg_autorecentre_off, z_slice):
    """
    We should be able to make the same visualisations whether we recentered or not.
    """
    z_limits = [1.9, 2.1] if z_slice else []
    recentered_z_limits = [-0.1, 0.1] if z_slice else []
    kwargs = {
        "resolution": 4,
        "project": "masses",
        "parallel": True,
        "periodic": True,
    }
    ref_img = project_pixel_grid(
        sg_autorecentre_off.gas,
        region=cosmo_array(
            [1.5, 2.5, 1.5, 2.5] + z_limits,
            u.Mpc,
            comoving=True,
            scale_factor=1,
            scale_exponent=1,
        ),
        **kwargs,
    )
    recentered_img = project_pixel_grid(
        sg.gas,
        region=cosmo_array(
            [-0.5, 0.5, -0.5, 0.5] + recentered_z_limits,
            u.Mpc,
            comoving=True,
            scale_factor=1,
            scale_exponent=1,
        ),
        **kwargs,
    )
    assert np.allclose(ref_img, recentered_img)
