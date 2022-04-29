import pytest
import numpy as np
import unyt as u
from unyt.testing import assert_allclose_units
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

# make an arbitrary rotation matrix for testing
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


class TestAutoCoordinateTransformations:

    @pytest.mark.parametrize(
        "particle_name, expected_xy, expected_z",
        [(k, expected_xy[k], expected_z[k])
         for k in present_particle_types.values()]
    )
    def test_auto_recentering(self, sg, particle_name, expected_xy,
                              expected_z):
        """
        The galaxy particles should be around (0, 0, 0),
        not around (2, 2, 2) Mpc like they are in box coords.
        """
        xyz = getattr(sg, particle_name).coordinates
        assert (np.abs(xyz[:, :2]) <= expected_xy).all()
        assert (np.abs(xyz[:, 2]) <= expected_z).all()

    @pytest.mark.parametrize(
        "particle_name, expected_xy, expected_z",
        [(k, expected_vxy[k], expected_vz[k])
         for k in present_particle_types.values()]
    )
    def test_auto_recentering_velocity(self, sg, particle_name, expected_xy,
                                       expected_z):
        """
        The galaxy velocities should be around (0, 0, 0),
        the velocity centre is (200, 200, 200).
        """
        vxyz = getattr(sg, particle_name).velocities
        assert (np.abs(vxyz[:, :2]) <= expected_xy).all()
        assert (np.abs(vxyz[:, 2]) <= expected_z).all()

    @pytest.mark.parametrize(
        "particle_name, expected_xy, expected_z",
        [(k, expected_xy[k], expected_z[k])
         for k in present_particle_types.values()]
    )
    def test_auto_recentering_extra(self, sg, particle_name, expected_xy,
                                    expected_z):
        """
        Check that another array flagged to transform like coordinates
        actually auto-recentres.
        """
        xyz = getattr(sg, particle_name).extra_coordinates
        assert (np.abs(xyz[:, :2]) <= expected_xy).all()
        assert (np.abs(xyz[:, 2]) <= expected_z).all()

    @pytest.mark.parametrize(
        "particle_name, expected_vxy, expected_vz",
        [(k, expected_vxy[k], expected_vz[k])
         for k in present_particle_types.values()]
    )
    def test_auto_recentering_velocity_extra(self, sg, particle_name,
                                             expected_vxy, expected_vz):
        """
        Check that another array flagged to transform like velocities
        actually auto-recentres.
        """
        vxyz = getattr(sg, particle_name).extra_velocities
        assert (np.abs(vxyz[:, :2]) <= expected_vxy).all()
        assert (np.abs(vxyz[:, 2]) <= expected_vz).all()

    @pytest.mark.parametrize(
        "particle_name, expected_xy, expected_z",
        [(k, expected_xy[k], expected_z[k])
         for k in present_particle_types.values()]
    )
    def test_auto_recentering_custom_name(self, sg_custom_names, particle_name,
                                          expected_xy, expected_z):
        """
        Recentering should still work if we set up a custom coordinates name.
        """
        custom_name = sg_custom_names.coordinates_dataset_name
        xyz = getattr(
            getattr(sg_custom_names, particle_name),
            f'{custom_name}'
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
        """
        Recentering should still work if we set up a custom velocities name.
        """
        custom_name = sg_custom_names.velocities_dataset_name
        vxyz = getattr(
            getattr(sg_custom_names, particle_name),
            f'{custom_name}'
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
        """
        Positions should still be offcentre.
        """
        xyz = getattr(sg_autorecentre_off, particle_name).coordinates
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
        """
        Velocities should still be offcentre.
        """
        vxyz = getattr(sg_autorecentre_off, particle_name).velocities
        assert (np.abs(vxyz[:, :2] - 200 * u.km / u.s) <= expected_vxy).all()
        assert (np.abs(vxyz[:, 2] - 200 * u.km / u.s) <= expected_vz).all()


class TestManualCoordinateTransformations:

    @pytest.mark.parametrize(
        "coordinate_name",
        ('coordinates', 'extra_coordinates')
    )
    @pytest.mark.parametrize("particle_name", present_particle_types.values())
    @pytest.mark.parametrize("before_load", (True, False))
    def test_manual_recentring(self, sg, particle_name, coordinate_name,
                               before_load):
        """
        Check that recentring places new centre where expected.
        """
        xyz_before = getattr(getattr(sg, particle_name), f'{coordinate_name}')
        if before_load:
            setattr(
                getattr(sg, particle_name)._particle_dataset,
                f'_{coordinate_name}',
                None
            )
        sg.recentre(cosmo_array([1, 1, 1], u.Mpc))
        xyz = getattr(getattr(sg, particle_name), f'{coordinate_name}')
        assert_allclose_units(
            xyz_before - cosmo_array([1, 1, 1], u.Mpc),
            xyz,
            rtol=1.e-4,
            atol=abstol_c
        )

    @pytest.mark.parametrize(
        "velocity_name",
        ('velocities', 'extra_velocities')
    )
    @pytest.mark.parametrize("particle_name", present_particle_types.values())
    @pytest.mark.parametrize("before_load", (True, False))
    def test_manual_recentring_velocity(self, sg, particle_name,
                                        velocity_name, before_load):
        """
        Check that velocity recentring places new velocity centre correctly.
        """
        vxyz_before = getattr(getattr(sg, particle_name), f'{velocity_name}')
        if before_load:
            setattr(
                getattr(sg, particle_name)._particle_dataset,
                f'_{velocity_name}',
                None
            )
        sg.recentre_velocity(cosmo_array([100, 100, 100], u.km / u.s))
        vxyz = getattr(getattr(sg, particle_name), f'{velocity_name}')
        assert_allclose_units(
            vxyz_before - cosmo_array([100, 100, 100], u.km / u.s),
            vxyz,
            rtol=1.e-4,
            atol=abstol_v
        )

    @pytest.mark.parametrize(
        "coordinate_name",
        ('coordinates', 'extra_coordinates')
    )
    @pytest.mark.parametrize("particle_name", present_particle_types.values())
    @pytest.mark.parametrize("before_load", (True, False))
    def test_translate(self, sg, particle_name, coordinate_name, before_load):
        """
        Check that translation translates to expected location.
        """
        xyz_before = getattr(getattr(sg, particle_name), f'{coordinate_name}')
        if before_load:
            setattr(
                getattr(sg, particle_name)._particle_dataset,
                f'_{coordinate_name}',
                None
            )
        sg.translate(cosmo_array([1, 1, 1], u.Mpc))
        xyz = getattr(getattr(sg, particle_name), f'{coordinate_name}')
        assert_allclose_units(
            xyz_before + cosmo_array([1, 1, 1], u.Mpc),
            xyz,
            rtol=1.e-4,
            atol=abstol_c
        )

    @pytest.mark.parametrize(
        "velocity_name",
        ('velocities', 'extra_velocities')
    )
    @pytest.mark.parametrize("particle_name", present_particle_types.values())
    @pytest.mark.parametrize("before_load", (True, False))
    def test_boost(self, sg, particle_name, velocity_name, before_load):
        """
        Check that boost boosts to expected reference velocity.
        """
        vxyz_before = getattr(getattr(sg, particle_name), f'{velocity_name}')
        if before_load:
            setattr(
                getattr(sg, particle_name)._particle_dataset,
                f'{velocity_name}',
                None
            )
        sg.boost(cosmo_array([100, 100, 100], u.km / u.s))
        vxyz = getattr(getattr(sg, particle_name), f'{velocity_name}')
        assert_allclose_units(
            vxyz_before + cosmo_array([100, 100, 100], u.km / u.s),
            vxyz,
            rtol=1.e-4,
            atol=abstol_v
        )

    @pytest.mark.parametrize(
        "coordinate_name, velocity_name",
        (
            ('coordinates', 'velocities'),
            ('extra_coordinates', 'extra_velocities')
        )
    )
    @pytest.mark.parametrize("particle_name", present_particle_types.values())
    @pytest.mark.parametrize("before_load", (True, False))
    def test_rotation(self, sg, particle_name, coordinate_name, velocity_name,
                      before_load):
        """
        Check that an arbitrary rotation rotates positions and velocities.
        """
        xyz_before = getattr(getattr(sg, particle_name), f'{coordinate_name}')
        vxyz_before = getattr(getattr(sg, particle_name), f'{velocity_name}')
        if before_load:
            setattr(
                getattr(sg, particle_name)._particle_dataset,
                f'_{coordinate_name}',
                None
            )
            setattr(
                getattr(sg, particle_name)._particle_dataset,
                f'_{velocity_name}',
                None
            )
        sg.rotate(Rotation.from_matrix(rot))
        xyz = getattr(getattr(sg, particle_name), f'{coordinate_name}')
        vxyz = getattr(getattr(sg, particle_name), f'{velocity_name}')
        assert_allclose_units(
            xyz_before.dot(rot),
            xyz,
            rtol=1.e-4,
            atol=abstol_c
        )
        assert_allclose_units(
            vxyz_before.dot(rot),
            vxyz,
            rtol=1.e-4,
            atol=abstol_v
        )

    @pytest.mark.parametrize(
        "coordinate_name",
        ('coordinates', 'extra_coordinates')
    )
    @pytest.mark.parametrize(
        "particle_name",
        present_particle_types.values()
    )
    def test_box_wrap(self, sg, particle_name, coordinate_name):
        """
        Check that translating by a box length wraps back to previous state.
        """
        boxsize = sg.metadata.boxsize
        xyz_before = getattr(getattr(sg, particle_name), f'{coordinate_name}')
        sg.translate(boxsize)
        xyz = getattr(getattr(sg, particle_name), f'{coordinate_name}')
        assert_allclose_units(
            xyz_before,
            xyz,
            rtol=1.e-4,
            atol=abstol_c
        )


class TestSequentialTransformations:

    @pytest.mark.parametrize("before_load", (True, False))
    def test_translate_then_rotate(self, sg, before_load):
        """
        Check that sequential transformations work correctly. Combining
        rotation and translation checks the implementation of the 4x4
        transformation matrix.
        """
        xyz_before = sg.gas.coordinates
        if before_load:
            sg.gas._coordinates = None
        sg.translate(cosmo_array([1, 1, 1], u.Mpc))
        sg.rotate(Rotation.from_matrix(rot))
        xyz = sg.gas.coordinates
        assert_allclose_units(
            (xyz_before + cosmo_array([1, 1, 1], u.Mpc)).dot(rot),
            xyz,
            rtol=1.e-4,
            atol=abstol_c
        )

    @pytest.mark.parametrize("before_load", (True, False))
    def test_rotate_then_translate(self, sg, before_load):
        """
        Check that sequential transformations work correctly. Combining
        rotation and translation checks the implementation of the 4x4
        transformation matrix.
        """
        xyz_before = sg.gas.coordinates
        if before_load:
            sg.gas._coordinates = None
        sg.rotate(Rotation.from_matrix(rot))
        sg.translate(cosmo_array([1, 1, 1], u.Mpc))
        xyz = sg.gas.coordinates
        assert_allclose_units(
            xyz_before.dot(rot) + cosmo_array([1, 1, 1], u.Mpc),
            xyz,
            rtol=1.e-4,
            atol=abstol_c
        )

    @pytest.mark.parametrize("before_load", (True, False))
    def test_boost_then_rotate(self, sg, before_load):
        """
        Check that sequential transformations work correctly. Combining
        rotation and translation checks the implementation of the 4x4
        transformation matrix.
        """
        vxyz_before = sg.gas.velocities
        if before_load:
            sg.gas._velocities = None
        sg.boost(cosmo_array([100, 100, 100], u.km / u.s))
        sg.rotate(Rotation.from_matrix(rot))
        vxyz = sg.gas.velocities
        assert_allclose_units(
            (vxyz_before + cosmo_array([100, 100, 100], u.km / u.s)).dot(rot),
            vxyz,
            rtol=1.e-4,
            atol=abstol_v
        )

    @pytest.mark.parametrize("before_load", (True, False))
    def test_rotate_then_boost(self, sg, before_load):
        """
        Check that sequential transformations work correctly. Combining
        rotation and translation checks the implementation of the 4x4
        transformation matrix.
        """
        vxyz_before = sg.gas.velocities
        if before_load:
            sg.gas._velocities = None
        sg.rotate(Rotation.from_matrix(rot))
        sg.boost(cosmo_array([100, 100, 100], u.km / u.s))
        vxyz = sg.gas.velocities
        assert_allclose_units(
            vxyz_before.dot(rot) + cosmo_array([100, 100, 100], u.km / u.s),
            vxyz,
            rtol=1.e-4,
            atol=abstol_v
        )
