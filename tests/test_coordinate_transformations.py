import pytest
import numpy as np
import unyt as u
from unyt.testing import assert_allclose_units
from swiftsimio.objects import cosmo_array, cosmo_factor, a, cosmo_quantity
from scipy.spatial.transform import Rotation
from toysnap import present_particle_types, toysnap_filename, ToyHF
from swiftgalaxy import SWIFTGalaxy
from swiftgalaxy.reader import _apply_translation, _apply_4transform

reltol = 1.01  # allow some wiggle room for floating point roundoff
abstol_c = cosmo_quantity(
    10, u.pc, comoving=True, cosmo_factor=cosmo_factor(a**1, 1.0)
)  # less than this is ~0
abstol_v = cosmo_quantity(
    10, u.m / u.s, comoving=True, cosmo_factor=cosmo_factor(a**0, 1.0)
)  # less than this is ~0

expected_xy = {
    "gas": cosmo_array(
        reltol * 10,
        units=u.kpc,
        comoving=True,
        cosmo_factor=cosmo_factor(a**1, scale_factor=1.0),
    ),
    "dark_matter": cosmo_array(
        reltol * 100,
        units=u.kpc,
        comoving=True,
        cosmo_factor=cosmo_factor(a**1, scale_factor=1.0),
    ),
    "stars": cosmo_array(
        reltol * 5,
        units=u.kpc,
        comoving=True,
        cosmo_factor=cosmo_factor(a**1, scale_factor=1.0),
    ),
    "black_holes": cosmo_array(
        abstol_c, comoving=True, cosmo_factor=cosmo_factor(a**1, scale_factor=1.0)
    ),
}
expected_z = {
    "gas": cosmo_array(
        reltol,
        units=u.kpc,
        comoving=True,
        cosmo_factor=cosmo_factor(a**1, scale_factor=1.0),
    ),
    "dark_matter": cosmo_array(
        reltol * 100,
        units=u.kpc,
        comoving=True,
        cosmo_factor=cosmo_factor(a**1, scale_factor=1.0),
    ),
    "stars": cosmo_array(
        reltol * 500,
        units=u.pc,
        comoving=True,
        cosmo_factor=cosmo_factor(a**1, scale_factor=1.0),
    ),
    "black_holes": cosmo_array(
        abstol_c, comoving=True, cosmo_factor=cosmo_factor(a**1, scale_factor=1.0)
    ),
}
expected_vxy = {
    "gas": cosmo_array(
        reltol * 100,
        units=u.km / u.s,
        comoving=True,
        cosmo_factor=cosmo_factor(a**0, scale_factor=1.0),
    ),
    "dark_matter": cosmo_array(
        reltol * 100,
        units=u.km / u.s,
        comoving=True,
        cosmo_factor=cosmo_factor(a**0, scale_factor=1.0),
    ),
    "stars": cosmo_array(
        reltol * 50,
        units=u.km / u.s,
        comoving=True,
        cosmo_factor=cosmo_factor(a**0, scale_factor=1.0),
    ),
    "black_holes": cosmo_array(
        abstol_v, comoving=True, cosmo_factor=cosmo_factor(a**0, scale_factor=1.0)
    ),
}
expected_vz = {
    "gas": cosmo_array(
        reltol * 10,
        units=u.km / u.s,
        comoving=True,
        cosmo_factor=cosmo_factor(a**0, scale_factor=1.0),
    ),
    "dark_matter": cosmo_array(
        reltol * 100,
        units=u.km / u.s,
        comoving=True,
        cosmo_factor=cosmo_factor(a**0, scale_factor=1.0),
    ),
    "stars": cosmo_array(
        reltol * 10,
        units=u.km / u.s,
        comoving=True,
        cosmo_factor=cosmo_factor(a**0, scale_factor=1.0),
    ),
    "black_holes": cosmo_array(
        abstol_v, comoving=True, cosmo_factor=cosmo_factor(a**0, scale_factor=1.0)
    ),
}

# make an arbitrary rotation matrix for testing
alpha = np.pi / 7
beta = 5 * np.pi / 3
gamma = 13 * np.pi / 8
rot = np.array(
    [
        [
            np.cos(beta) * np.cos(gamma),
            np.sin(alpha) * np.sin(beta) * np.cos(gamma)
            - np.cos(alpha) * np.sin(gamma),
            np.cos(alpha) * np.sin(beta) * np.cos(gamma)
            + np.sin(alpha) * np.sin(gamma),
        ],
        [
            np.cos(beta) * np.sin(gamma),
            np.sin(alpha) * np.sin(beta) * np.sin(gamma)
            + np.cos(alpha) * np.cos(gamma),
            np.cos(alpha) * np.sin(beta) * np.sin(gamma)
            - np.sin(alpha) * np.cos(gamma),
        ],
        [-np.sin(beta), np.sin(alpha) * np.cos(beta), np.cos(alpha) * np.cos(beta)],
    ]
)


class TestAutoCoordinateTransformations:
    @pytest.mark.parametrize(
        "particle_name, expected_xy, expected_z",
        [(k, expected_xy[k], expected_z[k]) for k in present_particle_types.values()],
    )
    def test_auto_recentering(self, sg, particle_name, expected_xy, expected_z):
        """
        The galaxy particles should be around (0, 0, 0),
        not around (2, 2, 2) Mpc like they are in box coords.
        """
        xyz = getattr(sg, particle_name).coordinates
        assert (np.abs(xyz[:, :2]) <= expected_xy).all()
        assert (np.abs(xyz[:, 2]) <= expected_z).all()

    @pytest.mark.parametrize(
        "particle_name, expected_vxy, expected_vz",
        [(k, expected_vxy[k], expected_vz[k]) for k in present_particle_types.values()],
    )
    def test_auto_recentering_velocity(
        self, sg, particle_name, expected_vxy, expected_vz
    ):
        """
        The galaxy velocities should be around (0, 0, 0),
        the velocity centre is (200, 200, 200).
        """
        vxyz = getattr(sg, particle_name).velocities
        assert (np.abs(vxyz[:, :2]) <= expected_vxy).all()
        assert (np.abs(vxyz[:, 2]) <= expected_vz).all()

    @pytest.mark.parametrize(
        "particle_name, expected_xy, expected_z",
        [(k, expected_xy[k], expected_z[k]) for k in present_particle_types.values()],
    )
    def test_auto_recentering_extra(self, sg, particle_name, expected_xy, expected_z):
        """
        Check that another array flagged to transform like coordinates
        actually auto-recentres.
        """
        xyz = getattr(sg, particle_name).extra_coordinates
        assert (np.abs(xyz[:, :2]) <= expected_xy).all()
        assert (np.abs(xyz[:, 2]) <= expected_z).all()

    @pytest.mark.parametrize(
        "particle_name, expected_vxy, expected_vz",
        [(k, expected_vxy[k], expected_vz[k]) for k in present_particle_types.values()],
    )
    def test_auto_recentering_velocity_extra(
        self, sg, particle_name, expected_vxy, expected_vz
    ):
        """
        Check that another array flagged to transform like velocities
        actually auto-recentres.
        """
        vxyz = getattr(sg, particle_name).extra_velocities
        assert (np.abs(vxyz[:, :2]) <= expected_vxy).all()
        assert (np.abs(vxyz[:, 2]) <= expected_vz).all()

    @pytest.mark.parametrize(
        "particle_name, expected_xy, expected_z",
        [(k, expected_xy[k], expected_z[k]) for k in present_particle_types.values()],
    )
    def test_auto_recentering_custom_name(
        self, sg_custom_names, particle_name, expected_xy, expected_z
    ):
        """
        Recentering should still work if we set up a custom coordinates name.
        """
        custom_name = sg_custom_names.coordinates_dataset_name
        xyz = getattr(getattr(sg_custom_names, particle_name), f"{custom_name}")
        assert (np.abs(xyz[:, :2]) <= expected_xy).all()
        assert (np.abs(xyz[:, 2]) <= expected_z).all()

    @pytest.mark.parametrize(
        "particle_name, expected_vxy, expected_vz",
        [(k, expected_vxy[k], expected_vz[k]) for k in present_particle_types.values()],
    )
    def test_auto_recentering_velocity_custom_name(
        self, sg_custom_names, particle_name, expected_vxy, expected_vz
    ):
        """
        Recentering should still work if we set up a custom velocities name.
        """
        custom_name = sg_custom_names.velocities_dataset_name
        vxyz = getattr(getattr(sg_custom_names, particle_name), f"{custom_name}")
        assert (np.abs(vxyz[:, :2]) <= expected_vxy).all()
        assert (np.abs(vxyz[:, 2]) <= expected_vz).all()

    @pytest.mark.parametrize(
        "particle_name, expected_xy, expected_z",
        [(k, expected_xy[k], expected_z[k]) for k in present_particle_types.values()],
    )
    def test_auto_recentering_off(
        self, sg_autorecentre_off, particle_name, expected_xy, expected_z
    ):
        """
        Positions should still be offcentre.
        """
        xyz = getattr(sg_autorecentre_off, particle_name).coordinates
        assert (
            np.abs(
                xyz[:, :2]
                - cosmo_quantity(
                    2, u.Mpc, comoving=True, cosmo_factor=cosmo_factor(a**1, 1.0)
                )
            )
            <= expected_xy
        ).all()
        assert (
            np.abs(
                xyz[:, 2]
                - cosmo_quantity(
                    2, u.Mpc, comoving=True, cosmo_factor=cosmo_factor(a**1, 1.0)
                )
            )
            <= expected_z
        ).all()

    @pytest.mark.parametrize(
        "particle_name, expected_vxy, expected_vz",
        [(k, expected_vxy[k], expected_vz[k]) for k in present_particle_types.values()],
    )
    def test_auto_recentering_off_velocity(
        self, sg_autorecentre_off, particle_name, expected_vxy, expected_vz
    ):
        """
        Velocities should still be offcentre.
        """
        vxyz = getattr(sg_autorecentre_off, particle_name).velocities
        assert (
            np.abs(
                vxyz[:, :2]
                - cosmo_quantity(
                    200,
                    u.km / u.s,
                    comoving=True,
                    cosmo_factor=cosmo_factor(a**0, 1.0),
                )
            )
            <= expected_vxy
        ).all()
        assert (
            np.abs(
                vxyz[:, 2]
                - cosmo_quantity(
                    200,
                    u.km / u.s,
                    comoving=True,
                    cosmo_factor=cosmo_factor(a**0, 1.0),
                )
            )
            <= expected_vz
        ).all()


class TestManualCoordinateTransformations:
    @pytest.mark.parametrize("coordinate_name", ("coordinates", "extra_coordinates"))
    @pytest.mark.parametrize("particle_name", present_particle_types.values())
    @pytest.mark.parametrize("before_load", (True, False))
    def test_manual_recentring(self, sg, particle_name, coordinate_name, before_load):
        """
        Check that recentring places new centre where expected.
        """
        xyz_before = getattr(getattr(sg, particle_name), f"{coordinate_name}")
        if before_load:
            setattr(
                getattr(sg, particle_name)._particle_dataset,
                f"_{coordinate_name}",
                None,
            )
        new_centre = cosmo_array(
            [1, 1, 1],
            units=u.Mpc,
            comoving=True,
            cosmo_factor=cosmo_factor(a**1, scale_factor=1.0),
        )
        sg.recentre(new_centre)
        xyz = getattr(getattr(sg, particle_name), f"{coordinate_name}")
        assert_allclose_units(xyz_before - new_centre, xyz, rtol=1.0e-4, atol=abstol_c)

    @pytest.mark.parametrize("velocity_name", ("velocities", "extra_velocities"))
    @pytest.mark.parametrize("particle_name", present_particle_types.values())
    @pytest.mark.parametrize("before_load", (True, False))
    def test_manual_recentring_velocity(
        self, sg, particle_name, velocity_name, before_load
    ):
        """
        Check that velocity recentring places new velocity centre correctly.
        """
        vxyz_before = getattr(getattr(sg, particle_name), f"{velocity_name}")
        if before_load:
            setattr(
                getattr(sg, particle_name)._particle_dataset, f"_{velocity_name}", None
            )
        new_centre = cosmo_array(
            [100, 100, 100],
            units=u.km / u.s,
            comoving=True,
            cosmo_factor=cosmo_factor(a**0, scale_factor=1.0),
        )
        sg.recentre_velocity(new_centre)
        vxyz = getattr(getattr(sg, particle_name), f"{velocity_name}")
        assert_allclose_units(
            vxyz_before - new_centre, vxyz, rtol=1.0e-4, atol=abstol_v
        )

    @pytest.mark.parametrize("coordinate_name", ("coordinates", "extra_coordinates"))
    @pytest.mark.parametrize("particle_name", present_particle_types.values())
    @pytest.mark.parametrize("before_load", (True, False))
    def test_translate(self, sg, particle_name, coordinate_name, before_load):
        """
        Check that translation translates to expected location.
        """
        xyz_before = getattr(getattr(sg, particle_name), f"{coordinate_name}")
        if before_load:
            setattr(
                getattr(sg, particle_name)._particle_dataset,
                f"_{coordinate_name}",
                None,
            )
        translation = cosmo_array(
            [1, 1, 1],
            units=u.Mpc,
            comoving=True,
            cosmo_factor=cosmo_factor(a**1, scale_factor=1.0),
        )
        sg.translate(translation)
        xyz = getattr(getattr(sg, particle_name), f"{coordinate_name}")
        assert_allclose_units(xyz_before + translation, xyz, rtol=1.0e-4, atol=abstol_c)

    @pytest.mark.parametrize("velocity_name", ("velocities", "extra_velocities"))
    @pytest.mark.parametrize("particle_name", present_particle_types.values())
    @pytest.mark.parametrize("before_load", (True, False))
    def test_boost(self, sg, particle_name, velocity_name, before_load):
        """
        Check that boost boosts to expected reference velocity.
        """
        vxyz_before = getattr(getattr(sg, particle_name), f"{velocity_name}")
        if before_load:
            setattr(
                getattr(sg, particle_name)._particle_dataset, f"{velocity_name}", None
            )
        boost = cosmo_array(
            [100, 100, 100],
            u.km / u.s,
            comoving=True,
            cosmo_factor=cosmo_factor(a**0, scale_factor=1.0),
        )
        sg.boost(boost)
        vxyz = getattr(getattr(sg, particle_name), f"{velocity_name}")
        assert_allclose_units(vxyz_before + boost, vxyz, rtol=1.0e-4, atol=abstol_v)

    @pytest.mark.parametrize(
        "coordinate_name, velocity_name",
        (("coordinates", "velocities"), ("extra_coordinates", "extra_velocities")),
    )
    @pytest.mark.parametrize("particle_name", present_particle_types.values())
    @pytest.mark.parametrize("before_load", (True, False))
    def test_rotation(
        self, sg, particle_name, coordinate_name, velocity_name, before_load
    ):
        """
        Check that an arbitrary rotation rotates positions and velocities.
        """
        xyz_before = getattr(getattr(sg, particle_name), f"{coordinate_name}")
        vxyz_before = getattr(getattr(sg, particle_name), f"{velocity_name}")
        if before_load:
            setattr(
                getattr(sg, particle_name)._particle_dataset,
                f"_{coordinate_name}",
                None,
            )
            setattr(
                getattr(sg, particle_name)._particle_dataset, f"_{velocity_name}", None
            )
        sg.rotate(Rotation.from_matrix(rot))
        xyz = getattr(getattr(sg, particle_name), f"{coordinate_name}")
        vxyz = getattr(getattr(sg, particle_name), f"{velocity_name}")
        assert_allclose_units(xyz_before.dot(rot), xyz, rtol=1.0e-4, atol=abstol_c)
        assert_allclose_units(vxyz_before.dot(rot), vxyz, rtol=1.0e-4, atol=abstol_v)

    @pytest.mark.parametrize("coordinate_name", ("coordinates", "extra_coordinates"))
    @pytest.mark.parametrize("particle_name", present_particle_types.values())
    def test_box_wrap(self, sg, particle_name, coordinate_name):
        """
        Check that translating by two box lengths wraps back to previous state.
        """
        xyz_before = getattr(getattr(sg, particle_name), f"{coordinate_name}")
        sg.translate(-2 * sg.metadata.boxsize)  # -2x box size
        xyz = getattr(getattr(sg, particle_name), f"{coordinate_name}")
        assert_allclose_units(xyz_before, xyz, rtol=1.0e-4, atol=abstol_c)


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
        translation = cosmo_array(
            [1, 1, 1],
            units=u.Mpc,
            comoving=True,
            cosmo_factor=cosmo_factor(a**1, scale_factor=1.0),
        )
        sg.translate(translation)
        sg.rotate(Rotation.from_matrix(rot))
        xyz = sg.gas.coordinates
        assert_allclose_units(
            (xyz_before + translation).dot(rot), xyz, rtol=1.0e-4, atol=abstol_c
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
        translation = cosmo_array(
            [1, 1, 1],
            units=u.Mpc,
            comoving=True,
            cosmo_factor=cosmo_factor(a**1, scale_factor=1.0),
        )
        sg.translate(translation)
        xyz = sg.gas.coordinates
        assert_allclose_units(
            xyz_before.dot(rot) + translation, xyz, rtol=1.0e-4, atol=abstol_c
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
        boost = cosmo_array(
            [100, 100, 100],
            units=u.km / u.s,
            comoving=True,
            cosmo_factor=cosmo_factor(a**0, scale_factor=1.0),
        )
        sg.boost(boost)
        sg.rotate(Rotation.from_matrix(rot))
        vxyz = sg.gas.velocities
        assert_allclose_units(
            (vxyz_before + boost).dot(rot), vxyz, rtol=1.0e-4, atol=abstol_v
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
        boost = cosmo_array(
            [100, 100, 100],
            units=u.km / u.s,
            comoving=True,
            cosmo_factor=cosmo_factor(a**0, scale_factor=1.0),
        )
        sg.boost(boost)
        vxyz = sg.gas.velocities
        assert_allclose_units(
            vxyz_before.dot(rot) + boost, vxyz, rtol=1.0e-4, atol=abstol_v
        )


class TestCopyingTransformations:
    def test_auto_recentering_with_copied_coordinate_frame(self, sg):
        """
        Check that a SWIFTGalaxy initialised to copy the coordinate frame
        of an existing SWIFTGalaxy needs auto_recentre=False.
        """
        with pytest.raises(
            ValueError,
            match="Cannot use coordinate_frame_from with auto_recentre=True.",
        ):
            SWIFTGalaxy(
                toysnap_filename, ToyHF(), auto_recentre=True, coordinate_frame_from=sg
            )

    def test_copied_coordinate_transform(self, sg):
        """
        Check that a SWIFTGalaxy initialised to copy the coordinate frame
        of an existing SWIFTGalaxy adopts the correct coordinate frame.
        """
        sg.rotate(Rotation.from_matrix(rot))
        translation = cosmo_array(
            [1, 1, 1],
            units=u.Mpc,
            comoving=True,
            cosmo_factor=cosmo_factor(a**1, scale_factor=1.0),
        )
        sg.translate(translation)
        sg2 = SWIFTGalaxy(
            toysnap_filename, ToyHF(), auto_recentre=False, coordinate_frame_from=sg
        )
        assert_allclose_units(
            sg.gas.coordinates, sg2.gas.coordinates, rtol=1.0e-4, atol=abstol_c
        )

    def test_copied_velocity_transform(self, sg):
        """
        Check that a SWIFTGalaxy initialised to copy the coordinate frame
        of an existing SWIFTGalaxy adopts the correct velocity frame.
        """
        sg.rotate(Rotation.from_matrix(rot))
        translation = cosmo_array(
            [1, 1, 1],
            units=u.Mpc,
            comoving=True,
            cosmo_factor=cosmo_factor(a**1, scale_factor=1.0),
        )
        sg.translate(translation)
        sg2 = SWIFTGalaxy(
            toysnap_filename, ToyHF(), auto_recentre=False, coordinate_frame_from=sg
        )
        assert_allclose_units(
            sg.gas.velocities, sg2.gas.velocities, rtol=1.0e-4, atol=abstol_v
        )


class TestApplyTranslation:

    @pytest.mark.parametrize("comoving", [True, False])
    def test_comoving_physical_conversion(self, comoving):
        """
        The _apply_translation function should convert the offset to
        match the coordinates.
        """
        coords = cosmo_array(
            [[1, 2, 3], [4, 5, 6]],
            units=u.Mpc,
            comoving=comoving,
            cosmo_factor=cosmo_factor(a**1, scale_factor=1.0),
        )
        offset = cosmo_array(
            [1, 1, 1],
            units=u.Mpc,
            comoving=True,
            cosmo_factor=cosmo_factor(a**1, scale_factor=1.0),
        )
        result = _apply_translation(coords, offset)
        assert result.comoving == comoving

    @pytest.mark.parametrize("comoving", [True, False])
    def test_warn_comoving_missing(self, comoving):
        """
        If the offset does not have comoving information issue a warning.
        """
        coords = cosmo_array(
            [[1, 2, 3], [4, 5, 6]],
            units=u.Mpc,
            comoving=comoving,
            cosmo_factor=cosmo_factor(a**1, scale_factor=1.0),
        )
        offset = u.unyt_array(
            [1, 1, 1],
            units=u.Mpc,
        )
        msg = (
            "Translation assumed to be in comoving"
            if comoving
            else "Translation assumed to be in physical"
        )
        with pytest.warns(RuntimeWarning, match=msg):
            with pytest.warns(
                RuntimeWarning, match="Mixing arguments with and without"
            ):
                result = _apply_translation(coords, offset)
        assert result.comoving == comoving


class TestApply4Transform:

    @pytest.mark.parametrize("comoving", [True, False])
    def test_comoving_physical_conversion(self, comoving):
        """
        The _apply_4transform function should return comoving if input
        was comoving, physical otherwise.
        """
        coords = cosmo_array(
            [[1, 2, 3], [4, 5, 6]],
            units=u.Mpc,
            comoving=comoving,
            cosmo_factor=cosmo_factor(a**1, scale_factor=1.0),
        )
        # identity 4transform:
        transform = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
        result = _apply_4transform(coords, transform, transform_units=u.Mpc)
        assert result.comoving == comoving
