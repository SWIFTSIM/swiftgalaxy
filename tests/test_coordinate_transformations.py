import pytest
import numpy as np
import unyt as u
from swiftsimio.objects import cosmo_array
from scipy.spatial.transform import Rotation
from toysnap import present_particle_types

reltol = 1.01  # allow some wiggle room for floating point roundoff
abstol_c = 10 * u.pc  # less than this is ~0
abstol_v = 10 * u.m / u.s  # less than this is ~0

expected_xy = {
    'gas': reltol * 10 * u.kpc,
    'dark_matter': reltol * 100 * u.kpc,
    'stars': reltol * 5 * u.kpc,
    'black_holes': abstol_c
}
expected_z = {
    'gas': reltol * 1 * u.kpc,
    'dark_matter': reltol * 100 * u.kpc,
    'stars': reltol * 500 * u.pc,
    'black_holes': abstol_c
}
expected_vxy = {
    'gas': reltol * 100 * u.km / u.s,
    'dark_matter': reltol * 100 * u.km / u.s,
    'stars': reltol * 50 * u.km / u.s,
    'black_holes': abstol_v
}
expected_vz = {
    'gas': reltol * 10 * u.km / u.s,
    'dark_matter': reltol * 100 * u.km / u.s,
    'stars': reltol * 10 * u.km / u.s,
    'black_holes': abstol_v
}


class TestAutoCoordinateTransformations:

    @pytest.mark.parametrize(
        "particle_name, expected_xy, expected_z",
        [(k, expected_xy[k], expected_z[k])
         for k in present_particle_types.values()]
    )
    def test_auto_recentering(self, sg, particle_name, expected_xy,
                              expected_z):
        # the galaxy particles should be around (0, 0, 0)
        # (not around (2, 2, 2) Mpc like they are in box coords)
        getattr(sg, particle_name).coordinates
        xyz = getattr(sg, particle_name)._particle_dataset._coordinates
        assert (np.abs(xyz[:, :2]) <= expected_xy).all()
        assert (np.abs(xyz[:, 2]) <= expected_z).all()

    @pytest.mark.parametrize(
        "particle_name, expected_xy, expected_z",
        [(k, expected_vxy[k], expected_vz[k])
         for k in present_particle_types.values()]
    )
    def test_auto_recentering_velocity(self, sg, particle_name, expected_xy,
                                       expected_z):
        # the galaxy velocities should be around (0, 0, 0)
        # (the velocity centre is (200, 200, 200))
        getattr(sg, particle_name).velocities
        vxyz = getattr(sg, particle_name)._particle_dataset._velocities
        assert (np.abs(vxyz[:, :2]) <= expected_xy).all()
        assert (np.abs(vxyz[:, 2]) <= expected_z).all()

    @pytest.mark.parametrize(
        "particle_name, expected_xy, expected_z",
        [(k, expected_xy[k], expected_z[k])
         for k in present_particle_types.values()]
    )
    def test_auto_recentering_extra(self, sg, particle_name, expected_xy,
                                    expected_z):
        # check that another array flagged to transform like coordinates
        # actually auto-recentres
        getattr(sg, particle_name).extra_coordinates
        xyz = getattr(sg, particle_name)._particle_dataset._extra_coordinates
        assert (np.abs(xyz[:, :2]) <= expected_xy).all()
        assert (np.abs(xyz[:, 2]) <= expected_z).all()

    @pytest.mark.parametrize(
        "particle_name, expected_vxy, expected_vz",
        [(k, expected_vxy[k], expected_vz[k])
         for k in present_particle_types.values()]
    )
    def test_auto_recentering_velocity_extra(self, sg, particle_name,
                                             expected_vxy, expected_vz):
        # check that another array flagged to transform like velocities
        # actually auto-recentres
        getattr(sg, particle_name).extra_velocities
        vxyz = getattr(sg, particle_name)._particle_dataset._extra_velocities
        assert (np.abs(vxyz[:, :2]) <= expected_vxy).all()
        assert (np.abs(vxyz[:, 2]) <= expected_vz).all()

    @pytest.mark.parametrize(
        "particle_name, expected_xy, expected_z",
        [(k, expected_xy[k], expected_z[k])
         for k in present_particle_types.values()]
    )
    def test_auto_recentering_custom_name(self, sg_custom_names, particle_name,
                                          expected_xy, expected_z):
        # recentering should still work if we set up a custom coordinates name
        custom_name = sg_custom_names.coordinates_dataset_name
        getattr(
            getattr(sg_custom_names, particle_name),
            custom_name
        )
        xyz = getattr(
            getattr(
                sg_custom_names,
                particle_name
            )._particle_dataset,
            f'_{custom_name}'
        )
        assert (np.abs(xyz[:, :2]) <= expected_xy).all()
        assert (np.abs(xyz[:, 2]) <= expected_z).all()

    @pytest.mark.parametrize(
        "particle_name, expected_vxy, expected_vz",
        [(k, expected_vxy[k], expected_vz[k])
         for k in present_particle_types.values()]
    )
    def test_auto_recentering_velocity_custom_name(self, sg_custom_names,
                                                   particle_name, expected_vxy,
                                                   expected_vz):
        # recentering should still work if we set up a custom velocities name
        custom_name = sg_custom_names.velocities_dataset_name
        getattr(
            getattr(
                sg_custom_names,
                particle_name
            ),
            custom_name
        )
        vxyz = getattr(
            getattr(
                sg_custom_names,
                particle_name
            )._particle_dataset,
            f'_{custom_name}'
        )
        assert (np.abs(vxyz[:, :2]) <= expected_vxy).all()
        assert (np.abs(vxyz[:, 2]) <= expected_vz).all()

    @pytest.mark.parametrize(
        "particle_name, expected_xy, expected_z",
        [(k, expected_xy[k], expected_z[k])
         for k in present_particle_types.values()]
    )
    def test_auto_recentering_off(self, sg_autorecentre_off, particle_name,
                                  expected_xy, expected_z):
        # positions should still be offcentre
        getattr(sg_autorecentre_off, particle_name).coordinates
        xyz = getattr(
            sg_autorecentre_off,
            particle_name
        )._particle_dataset._coordinates
        assert (np.abs(xyz[:, :2] - 2 * u.Mpc) <= expected_xy).all()
        assert (np.abs(xyz[:, 2] - 2 * u.Mpc) <= expected_z).all()

    @pytest.mark.parametrize(
        "particle_name, expected_vxy, expected_vz",
        [(k, expected_vxy[k], expected_vz[k])
         for k in present_particle_types.values()]
    )
    def test_auto_recentering_off_velocity(self, sg_autorecentre_off,
                                           particle_name, expected_vxy,
                                           expected_vz):
        # velocities should still be offcentre
        getattr(sg_autorecentre_off, particle_name).velocities
        vxyz = getattr(
            sg_autorecentre_off,
            particle_name
        )._particle_dataset._velocities
        assert (np.abs(vxyz[:, :2] - 200 * u.km / u.s) <= expected_vxy).all()
        assert (np.abs(vxyz[:, 2] - 200 * u.km / u.s) <= expected_vz).all()


class TestManualCoordinateTransformations:

    @pytest.mark.parametrize(
        "coordinate_name",
        ('coordinates', 'extra_coordinates')
    )
    @pytest.mark.parametrize("particle_name", present_particle_types.values())
    def test_manual_recentring(self, sg, particle_name, coordinate_name):
        # check that recentring places new centre where expected
        getattr(getattr(sg, particle_name), coordinate_name)
        xyz_before = getattr(
            getattr(sg, particle_name)._particle_dataset,
            f'_{coordinate_name}'
        )
        sg.recentre(cosmo_array([1, 1, 1], u.Mpc))
        xyz = getattr(
            getattr(sg, particle_name)._particle_dataset,
            f'_{coordinate_name}'
        )
        assert ((xyz_before - cosmo_array([1, 1, 1], u.Mpc)) - xyz
                <= abstol_c).all()

    @pytest.mark.parametrize(
        "velocity_name",
        ('velocities', 'extra_velocities')
    )
    @pytest.mark.parametrize("particle_name", present_particle_types.values())
    def test_manual_recentring_velocity(self, sg, particle_name,
                                        velocity_name):
        # check that velocity recentring places new velocity centre correctly
        getattr(getattr(sg, particle_name), velocity_name)
        vxyz_before = getattr(
            getattr(sg, particle_name)._particle_dataset,
            f'_{velocity_name}'
        )
        sg.recentre_velocity(cosmo_array([100, 100, 100], u.km / u.s))
        vxyz = getattr(
            getattr(sg, particle_name)._particle_dataset,
            f'_{velocity_name}'
        )
        assert (
            (vxyz_before - cosmo_array([100, 100, 100], u.km / u.s)) - vxyz
            <= abstol_v
        ).all()

    @pytest.mark.parametrize(
        "coordinate_name",
        ('coordinates', 'extra_coordinates')
    )
    @pytest.mark.parametrize("particle_name", present_particle_types.values())
    def test_translate(self, sg, particle_name, coordinate_name):
        # check that translation translates to expected location
        getattr(getattr(sg, particle_name), coordinate_name)
        xyz_before = getattr(
            getattr(sg, particle_name)._particle_dataset,
            f'_{coordinate_name}'
        )
        sg.translate(cosmo_array([1, 1, 1], u.Mpc))
        xyz = getattr(
            getattr(sg, particle_name)._particle_dataset,
            f'_{coordinate_name}'
        )
        assert ((xyz_before + cosmo_array([1, 1, 1], u.Mpc)) - xyz
                <= abstol_c).all()

    @pytest.mark.parametrize(
        "velocity_name",
        ('velocities', 'extra_velocities')
    )
    @pytest.mark.parametrize("particle_name", present_particle_types.values())
    def test_boost(self, sg, particle_name, velocity_name):
        # check that boost boosts to expected reference velocity
        getattr(getattr(sg, particle_name), velocity_name)
        vxyz_before = getattr(
            getattr(sg, particle_name)._particle_dataset,
            f'_{velocity_name}'
        )
        sg.boost(cosmo_array([100, 100, 100], u.km / u.s))
        vxyz = getattr(
            getattr(sg, particle_name)._particle_dataset,
            f'_{velocity_name}'
        )
        assert (
            (vxyz_before + cosmo_array([100, 100, 100], u.km / u.s)) - vxyz
            <= abstol_v
        ).all()

    @pytest.mark.parametrize(
        "coordinate_name, velocity_name",
        (
            ('coordinates', 'velocities'),
            ('extra_coordinates', 'extra_velocities')
        )
    )
    @pytest.mark.parametrize("particle_name", present_particle_types.values())
    def test_rotation(self, sg, particle_name, coordinate_name, velocity_name):
        # check that an arbitrary rotation rotates positions and velocities
        alpha = np.pi / 7
        beta = 5 * np.pi / 3
        gamma = 13 * np.pi / 8
        rot = np.array([
            [
                np.cos(beta) * np.cos(gamma),
                np.sin(alpha) * np.sin(beta) * np.cos(gamma)
                - np.cos(alpha) * np.sin(gamma),
                np.cos(alpha) * np.sin(beta) * np.cos(gamma)
                + np.sin(alpha) * np.sin(gamma)
            ],
            [
                np.cos(beta) * np.sin(gamma),
                np.sin(alpha) * np.sin(beta) * np.sin(gamma)
                + np.cos(alpha) * np.cos(gamma),
                np.cos(alpha) * np.sin(beta) * np.sin(gamma)
                - np.sin(alpha) * np.cos(gamma)
            ],
            [
                -np.sin(beta),
                np.sin(alpha) * np.cos(beta),
                np.cos(alpha) * np.cos(beta)
            ]
        ])
        getattr(getattr(sg, particle_name), coordinate_name)
        getattr(getattr(sg, particle_name), velocity_name)
        xyz_before = getattr(
            getattr(sg, particle_name)._particle_dataset,
            f'_{coordinate_name}'
        )
        vxyz_before = getattr(
            getattr(sg, particle_name)._particle_dataset,
            f'_{velocity_name}'
        )
        sg.rotate(Rotation.from_matrix(rot))
        xyz = getattr(
            getattr(sg, particle_name)._particle_dataset,
            f'_{coordinate_name}'
        )
        vxyz = getattr(
            getattr(sg, particle_name)._particle_dataset,
            f'_{velocity_name}'
        )
        assert (xyz_before.dot(rot) - xyz <= abstol_c).all()
        assert (vxyz_before.dot(rot) - vxyz <= abstol_v).all()

    @pytest.mark.parametrize(
        "coordinate_name",
        ('coordinates', 'extra_coordinates')
    )
    @pytest.mark.parametrize(
        "particle_name",
        present_particle_types.values()
    )
    def test_box_wrap(self, sg, particle_name, coordinate_name):
        # check that translating by a box length wraps back to previous state
        boxsize = sg.metadata.boxsize
        getattr(getattr(sg, particle_name), coordinate_name)
        xyz_before = getattr(
            getattr(sg, particle_name)._particle_dataset,
            f'_{coordinate_name}'
        )
        sg.translate(boxsize)
        xyz = getattr(
            getattr(sg, particle_name)._particle_dataset,
            f'_{coordinate_name}'
        )
        assert (xyz_before - xyz <= abstol_c).all()
