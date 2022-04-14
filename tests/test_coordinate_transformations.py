import numpy as np
import unyt as u
from swiftsimio.objects import cosmo_array
from scipy.spatial.transform import Rotation

tol = 1.01  # allow some wiggle room for floating point roundoff


class TestCoordinateTransformations:

    def test_auto_recentering(self, sg):
        # the galaxy particles should be around (0, 0, 0)
        # (not around (2, 2, 2) like they are in box coords)
        for particle_name in sg.metadata.present_particle_names:
            getattr(sg, particle_name).coordinates
            xyz = getattr(sg, particle_name)._particle_dataset._coordinates
            if particle_name == 'gas':
                assert (np.abs(xyz[:, :2]) <= tol * 10 * u.kpc).all()
                assert (np.abs(xyz[:, 2]) <= tol * 1 * u.kpc).all()
            elif particle_name == 'dark_matter':
                assert (np.abs(xyz) < 100 * u.kpc).all()
            elif particle_name == 'stars':
                assert (np.abs(xyz[:, :2]) <= tol * 5 * u.kpc).all()
                assert (np.abs(xyz[:, 2]) <= tol * 500 * u.pc).all()
            elif particle_name == 'black_holes':
                assert (np.abs(xyz) <= 10 * u.pc).all()  # ~0 pc

    def test_auto_recentering_velocity(self, sg):
        # the galaxy velocities should be around (0, 0, 0)
        # (the velocity centre is (200, 200, 200))
        for particle_name in sg.metadata.present_particle_names:
            getattr(sg, particle_name).velocities
            vxyz = getattr(sg, particle_name)._particle_dataset._velocities
            if particle_name == 'gas':
                assert (np.abs(vxyz[:, :2]) <= tol * 100 * u.km / u.s).all()
                assert (np.abs(vxyz[:, 2]) <= tol * 10 * u.km / u.s).all()
            elif particle_name == 'dark_matter':
                assert (np.abs(vxyz) < 100 * u.km / u.s).all()
            elif particle_name == 'stars':
                assert (np.abs(vxyz[:, :2]) <= tol * 50 * u.km / u.s).all()
                assert (np.abs(vxyz[:, 2]) <= tol * 10 * u.km / u.s).all()
            elif particle_name == 'black_holes':
                assert (np.abs(vxyz) <= 50 * u.m / u.s).all()  # ~0 km/s

    def test_auto_recentering_extra(self, sg):
        # check that another array flagged to transform like coordinates
        # actually auto-recentres
        for particle_name in sg.metadata.present_particle_names:
            getattr(sg, particle_name).extra_coordinates
            xyz = \
                getattr(sg, particle_name)._particle_dataset._extra_coordinates
            if particle_name == 'gas':
                assert (np.abs(xyz[:, :2]) <= tol * 10 * u.kpc).all()
                assert (np.abs(xyz[:, 2]) <= tol * 1 * u.kpc).all()
            elif particle_name == 'dark_matter':
                assert (np.abs(xyz) < 100 * u.kpc).all()
            elif particle_name == 'stars':
                assert (np.abs(xyz[:, :2]) <= tol * 5 * u.kpc).all()
                assert (np.abs(xyz[:, 2]) <= tol * 500 * u.pc).all()
            elif particle_name == 'black_holes':
                assert (np.abs(xyz) <= 10 * u.pc).all()  # ~0 pc

    def test_auto_recentering_velocity_extra(self, sg):
        # check that another array flagged to transform like velocities
        # actually auto-recentres
        for particle_name in sg.metadata.present_particle_names:
            getattr(sg, particle_name).extra_velocities
            vxyz = \
                getattr(sg, particle_name)._particle_dataset._extra_velocities
            if particle_name == 'gas':
                assert (np.abs(vxyz[:, :2]) <= tol * 100 * u.km / u.s).all()
                assert (np.abs(vxyz[:, 2]) <= tol * 10 * u.km / u.s).all()
            elif particle_name == 'dark_matter':
                assert (np.abs(vxyz) < 100 * u.km / u.s).all()
            elif particle_name == 'stars':
                assert (np.abs(vxyz[:, :2]) <= tol * 50 * u.km / u.s).all()
                assert (np.abs(vxyz[:, 2]) <= tol * 10 * u.km / u.s).all()
            elif particle_name == 'black_holes':
                assert (np.abs(vxyz) <= 50 * u.m / u.s).all()  # ~0 km/s

    def test_auto_recentering_custom_name(self, sg_custom_names):
        # recentering should still work if we set up a custom coordinates name
        custom_name = sg_custom_names.coordinates_dataset_name
        for particle_name in sg_custom_names.metadata.present_particle_names:
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
            if particle_name == 'gas':
                assert (np.abs(xyz[:, :2]) <= tol * 10 * u.kpc).all()
                assert (np.abs(xyz[:, 2]) <= tol * 1 * u.kpc).all()
            elif particle_name == 'dark_matter':
                assert (np.abs(xyz) < 100 * u.kpc).all()
            elif particle_name == 'stars':
                assert (np.abs(xyz[:, :2]) <= tol * 5 * u.kpc).all()
                assert (np.abs(xyz[:, 2]) <= tol * 500 * u.pc).all()
            elif particle_name == 'black_holes':
                assert (np.abs(xyz) <= 10 * u.pc).all()  # ~0 pc

    def test_auto_recentering_velocity_custom_name(self, sg_custom_names):
        # recentering should still work if we set up a custom velocities name
        custom_name = sg_custom_names.velocities_dataset_name
        for particle_name in sg_custom_names.metadata.present_particle_names:
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
            if particle_name == 'gas':
                assert (np.abs(vxyz[:, :2]) <= tol * 100 * u.km / u.s).all()
                assert (np.abs(vxyz[:, 2]) <= tol * 10 * u.km / u.s).all()
            elif particle_name == 'dark_matter':
                assert (np.abs(vxyz) < 100 * u.km / u.s).all()
            elif particle_name == 'stars':
                assert (np.abs(vxyz[:, :2]) <= tol * 50 * u.km / u.s).all()
                assert (np.abs(vxyz[:, 2]) <= tol * 10 * u.km / u.s).all()
            elif particle_name == 'black_holes':
                assert (np.abs(vxyz) <= 50 * u.m / u.s).all()  # ~0 km/s

    def test_auto_recentering_off(self, sg_autorecentre_off):
        # both positions and velocities should still be offcentre
        # positions:
        for particle_name in (
                sg_autorecentre_off.metadata.present_particle_names):
            getattr(sg_autorecentre_off, particle_name).coordinates
            xyz = getattr(
                sg_autorecentre_off,
                particle_name
            )._particle_dataset._coordinates
            if particle_name == 'gas':
                assert (np.abs(xyz[:, :2] - 2 * u.Mpc)
                        <= tol * 10 * u.kpc).all()
                assert (np.abs(xyz[:, 2] - 2 * u.Mpc) <= tol * 1 * u.kpc).all()
            elif particle_name == 'dark_matter':
                assert (np.abs(xyz - 2 * u.Mpc) < 100 * u.kpc).all()
            elif particle_name == 'stars':
                assert (np.abs(xyz[:, :2] - 2 * u.Mpc)
                        <= tol * 5 * u.kpc).all()
                assert (np.abs(xyz[:, 2] - 2 * u.Mpc)
                        <= tol * 500 * u.pc).all()
            elif particle_name == 'black_holes':
                assert (np.abs(xyz - 2 * u.Mpc)
                        <= 10 * u.pc).all()  # ~0 pc
            # velocities:a
            getattr(sg_autorecentre_off, particle_name).velocities
            vxyz = getattr(
                sg_autorecentre_off,
                particle_name
            )._particle_dataset._velocities
            if particle_name == 'gas':
                assert (np.abs(vxyz[:, :2] - 200 * u.km / u.s)
                        <= tol * 100 * u.km / u.s).all()
                assert (np.abs(vxyz[:, 2] - 200 * u.km / u.s)
                        <= tol * 10 * u.km / u.s).all()
            elif particle_name == 'dark_matter':
                assert (np.abs(vxyz - 200 * u.km / u.s)
                        <= tol * 100 * u.km / u.s).all()
            elif particle_name == 'stars':
                assert (np.abs(vxyz[:, :2] - 200 * u.km / u.s)
                        <= tol * 50 * u.km / u.s).all()
                assert (np.abs(vxyz[:, 2] - 200 * u.km / u.s)
                        <= tol * 10 * u.km / u.s).all()
            elif particle_name == 'black_holes':
                assert (np.abs(vxyz - 200 * u.km / u.s)
                        <= 50 * u.m / u.s).all()  # ~0 km/s

    def test_manual_recentring(self, sg):
        # check that recentring places new centre where expected
        for particle_name in sg.metadata.present_particle_names:
            getattr(sg, particle_name).coordinates
            getattr(sg, particle_name).extra_coordinates
            xyz_before = \
                getattr(sg, particle_name)._particle_dataset._coordinates
            x_xyz_before = \
                getattr(sg, particle_name)._particle_dataset._extra_coordinates
            sg.recentre(cosmo_array([1, 1, 1], u.Mpc))
            xyz = getattr(sg, particle_name)._particle_dataset._coordinates
            x_xyz = \
                getattr(sg, particle_name)._particle_dataset._extra_coordinates
            assert ((xyz_before - cosmo_array([1, 1, 1], u.Mpc)) - xyz
                    <= 10 * u.pc).all()  # ~0 pc
            assert ((x_xyz_before - cosmo_array([1, 1, 1], u.Mpc)) - x_xyz
                    <= 10 * u.pc).all()  # ~0 pc
            # reset for next particle_name
            sg.recentre(cosmo_array([-1, -1, -1], u.Mpc))

    def test_manual_recentring_velocity(self, sg):
        # check that velocity recentring places new velocity centre correctly
        for particle_name in sg.metadata.present_particle_names:
            getattr(sg, particle_name).velocities
            getattr(sg, particle_name).extra_velocities
            vxyz_before = \
                getattr(sg, particle_name)._particle_dataset._velocities
            x_vxyz_before = \
                getattr(sg, particle_name)._particle_dataset._extra_velocities
            sg.recentre_velocity(cosmo_array([100, 100, 100], u.km / u.s))
            vxyz = getattr(sg, particle_name)._particle_dataset._velocities
            x_vxyz = \
                getattr(sg, particle_name)._particle_dataset._extra_velocities
            assert (
                (vxyz_before - cosmo_array([100, 100, 100], u.km / u.s)) - vxyz
                <= 50 * u.m / u.s
            ).all()  # ~0 km/s
            assert (
                (x_vxyz_before - cosmo_array([100, 100, 100], u.km / u.s))
                - x_vxyz
                <= 50 * u.m / u.s
            ).all()  # ~0 km/s
            # reset for next particle_name
            sg.recentre_velocity(cosmo_array([-100, -100, -100], u.km / u.s))

    def test_translate(self, sg):
        # check that translation translates to expected location
        for particle_name in sg.metadata.present_particle_names:
            getattr(sg, particle_name).coordinates
            getattr(sg, particle_name).extra_coordinates
            xyz_before = \
                getattr(sg, particle_name)._particle_dataset._coordinates
            x_xyz_before = \
                getattr(sg, particle_name)._particle_dataset._extra_coordinates
            sg.translate(cosmo_array([1, 1, 1], u.Mpc))
            xyz = getattr(sg, particle_name)._particle_dataset._coordinates
            x_xyz = \
                getattr(sg, particle_name)._particle_dataset._extra_coordinates
            assert ((xyz_before + cosmo_array([1, 1, 1], u.Mpc)) - xyz
                    <= 10 * u.pc).all()  # ~0 pc
            assert ((x_xyz_before + cosmo_array([1, 1, 1], u.Mpc)) - x_xyz
                    <= 10 * u.pc).all()  # ~0 pc
            # reset for next particle_name
            sg.translate(cosmo_array([-1, -1, -1], u.Mpc))

    def test_boost(self, sg):
        # check that boost boosts to expected reference velocity
        for particle_name in sg.metadata.present_particle_names:
            getattr(sg, particle_name).velocities
            getattr(sg, particle_name).extra_velocities
            vxyz_before = \
                getattr(sg, particle_name)._particle_dataset._velocities
            x_vxyz_before = \
                getattr(sg, particle_name)._particle_dataset._extra_velocities
            sg.boost(cosmo_array([100, 100, 100], u.km / u.s))
            vxyz = getattr(sg, particle_name)._particle_dataset._velocities
            x_vxyz = \
                getattr(sg, particle_name)._particle_dataset._extra_velocities
            assert (
                (vxyz_before + cosmo_array([100, 100, 100], u.km / u.s)) - vxyz
                <= 50 * u.m / u.s
            ).all()  # ~0 km/s
            assert (
                (x_vxyz_before + cosmo_array([100, 100, 100], u.km / u.s))
                - x_vxyz
                <= 50 * u.m / u.s
            ).all()  # ~0 km/s
            # reset for next particle_name
            sg.boost(cosmo_array([-100, -100, -100], u.km / u.s))

    def test_rotation(self, sg):
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
        for particle_name in sg.metadata.present_particle_names:
            getattr(sg, particle_name).coordinates
            getattr(sg, particle_name).velocities
            getattr(sg, particle_name).extra_coordinates
            getattr(sg, particle_name).extra_velocities
            xyz_before = \
                getattr(sg, particle_name)._particle_dataset._coordinates
            vxyz_before = \
                getattr(sg, particle_name)._particle_dataset._velocities
            x_xyz_before = \
                getattr(sg, particle_name)._particle_dataset._extra_coordinates
            x_vxyz_before = \
                getattr(sg, particle_name)._particle_dataset._extra_velocities
            sg.rotate(Rotation.from_matrix(rot))
            xyz = getattr(sg, particle_name)._particle_dataset._coordinates
            vxyz = getattr(sg, particle_name)._particle_dataset._velocities
            x_xyz = \
                getattr(sg, particle_name)._particle_dataset._extra_coordinates
            x_vxyz = \
                getattr(sg, particle_name)._particle_dataset._extra_velocities
            assert (xyz_before.dot(rot) - xyz <= 10 * u.pc).all()
            assert (vxyz_before.dot(rot) - vxyz <= 50 * u.m / u.s).all()
            assert (x_xyz_before.dot(rot) - x_xyz <= 10 * u.pc).all()
            assert (x_vxyz_before.dot(rot) - x_vxyz <= 50 * u.m / u.s).all()
            # reset for next particle_name
            sg.rotate(Rotation.from_matrix(rot.T))

    def test_box_wrap(self, sg):
        # check that translating by a box length wraps back to previous state
        boxsize = sg.metadata.boxsize
        for particle_name in sg.metadata.present_particle_names:
            getattr(sg, particle_name).coordinates
            getattr(sg, particle_name).extra_coordinates
            xyz_before = \
                getattr(sg, particle_name)._particle_dataset._coordinates
            x_xyz_before = \
                getattr(sg, particle_name)._particle_dataset._extra_coordinates
            sg.translate(boxsize)
            xyz = getattr(sg, particle_name)._particle_dataset._coordinates
            x_xyz = \
                getattr(sg, particle_name)._particle_dataset._extra_coordinates
            assert (xyz_before - xyz <= 10 * u.pc).all()  # ~0 pc
            assert (x_xyz_before - x_xyz <= 10 * u.pc).all()  # ~0 pc
            # reset for next particle_name
            sg.translate(-boxsize)
