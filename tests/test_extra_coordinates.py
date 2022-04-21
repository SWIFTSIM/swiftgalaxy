import pytest
import numpy as np
import unyt as u
from toysnap import present_particle_types

reltol = 1.01  # allow some wiggle room for floating point roundoff
abstol_c = 1 * u.pc  # less than this is ~0
abstol_v = 10 * u.m / u.s  # less than this is ~0


class TestCartesianCoordinates:

    @pytest.mark.parametrize("particle_name", present_particle_types.values())
    @pytest.mark.parametrize(
        "coordinate_name, mask",
        (
            ('x', np.s_[:, 0]),
            ('y', np.s_[:, 1]),
            ('z', np.s_[:, 2]),
            ('xyz', np.s_[...])
        )
    )
    @pytest.mark.parametrize(
        "coordinate_type, tol",
        (
            ('coordinates', abstol_c),
            ('velocities', abstol_v),
        )
    )
    def test_cartesian_coordinates(self, sg, particle_name, coordinate_name,
                                   mask, coordinate_type, tol):
        coordinate = getattr(
            getattr(sg, particle_name),
            coordinate_type
        )[mask]
        cartesian_coordinate = getattr(
            getattr(
                getattr(sg, particle_name),
                f'cartesian_{coordinate_type}'
            ),
            coordinate_name
        )
        assert(np.abs(cartesian_coordinate - coordinate) <= tol).all()


class TestSphericalCoordinates:

    @pytest.mark.parametrize("particle_name", present_particle_types.values())
    @pytest.mark.parametrize("alias", ('r', 'radius'))
    @pytest.mark.filterwarnings(
        "ignore:invalid value encountered in true_divide"
    )  # comes from r=0 particle, handled in definition of theta
    def test_spherical_r(self, sg, particle_name, alias):
        spherical_r = getattr(
            getattr(sg, particle_name).spherical_coordinates,
            alias
        )
        xyz = getattr(sg, particle_name).coordinates
        r_from_cartesian = np.sqrt(np.sum(np.power(xyz, 2), axis=1))
        assert (spherical_r - r_from_cartesian < abstol_c).all()

    @pytest.mark.parametrize("particle_name", present_particle_types.values())
    @pytest.mark.parametrize("alias", ('r', 'radius'))
    @pytest.mark.filterwarnings(
        "ignore:invalid value encountered in true_divide"
    )  # comes from r=0 particle, handled in definition of theta
    def test_spherical_velocity_r(self, sg, particle_name, alias):
        spherical_v_r = getattr(
            getattr(sg, particle_name).spherical_velocities,
            alias
        )
        xyz = getattr(sg, particle_name).coordinates
        vxyz = getattr(sg, particle_name).velocities
        r_from_cartesian = np.sqrt(np.sum(np.power(xyz, 2), axis=1))
        theta_from_cartesian = np.where(
            r_from_cartesian == 0,
            0,
            np.arcsin(xyz[:, 2] / r_from_cartesian)
        ) * u.rad
        phi_from_cartesian = np.arctan2(xyz[:, 1], xyz[:, 0])
        phi_from_cartesian = np.where(
            phi_from_cartesian < 0,
            phi_from_cartesian + 2 * np.pi,
            phi_from_cartesian
        )
        v_r_from_cartesian = \
            np.cos(theta_from_cartesian) * np.cos(phi_from_cartesian) \
            * vxyz[:, 0] \
            + np.cos(theta_from_cartesian) * np.sin(phi_from_cartesian) \
            * vxyz[:, 1] \
            + np.sin(theta_from_cartesian) \
            * vxyz[:, 2]
        print(theta_from_cartesian)
        assert (spherical_v_r - v_r_from_cartesian < abstol_v).all()
