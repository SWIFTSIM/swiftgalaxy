import pytest
import numpy as np
import unyt as u
from unyt.testing import assert_allclose_units
from toysnap import present_particle_types
from scipy.spatial.transform import Rotation

abstol_c = 1 * u.pc  # less than this is ~0
abstol_v = 10 * u.m / u.s  # less than this is ~0
abstol_a = 1.0e-4 * u.rad


class TestCartesianCoordinates:
    @pytest.mark.parametrize("particle_name", present_particle_types.values())
    @pytest.mark.parametrize(
        "coordinate_name, mask",
        (
            ("x", np.s_[:, 0]),
            ("y", np.s_[:, 1]),
            ("z", np.s_[:, 2]),
            ("xyz", np.s_[...]),
        ),
    )
    @pytest.mark.parametrize(
        "coordinate_type, tol",
        (
            ("coordinates", abstol_c),
            ("velocities", abstol_v),
        ),
    )
    def test_cartesian_coordinates(
        self, sg, particle_name, coordinate_name, mask, coordinate_type, tol
    ):
        """
        Check that cartesian coordinates match the particle coordinates.
        """
        coordinate = getattr(getattr(sg, particle_name), coordinate_type)[mask]
        cartesian_coordinate = getattr(
            getattr(getattr(sg, particle_name), f"cartesian_{coordinate_type}"),
            coordinate_name,
        )
        assert_allclose_units(cartesian_coordinate, coordinate, rtol=1.0e-4, atol=tol)


class TestSphericalCoordinates:
    @pytest.mark.parametrize("particle_name", present_particle_types.values())
    @pytest.mark.parametrize("alias", ("r", "radius"))
    @pytest.mark.filterwarnings(
        "ignore:invalid value encountered in true_divide"
    )  # comes from r=0 particle, handled in definition of theta
    def test_spherical_r(self, sg, particle_name, alias):
        """
        Check spherical radius matches direct computation.
        """
        spherical_r = getattr(getattr(sg, particle_name).spherical_coordinates, alias)
        xyz = getattr(sg, particle_name).coordinates
        r_from_cartesian = np.sqrt(np.sum(np.power(xyz, 2), axis=1))
        assert_allclose_units(spherical_r, r_from_cartesian, rtol=1.0e-4, atol=abstol_c)

    @pytest.mark.parametrize("particle_name", present_particle_types.values())
    @pytest.mark.parametrize("alias", ("r", "radius"))
    @pytest.mark.filterwarnings(
        "ignore:invalid value encountered in true_divide"
    )  # comes from r=0 particle, handled in definition of theta
    def test_spherical_velocity_r(self, sg, particle_name, alias):
        """
        Check spherical radial velocity matches direct computation.
        """
        spherical_v_r = getattr(getattr(sg, particle_name).spherical_velocities, alias)
        xyz = getattr(sg, particle_name).coordinates
        vxyz = getattr(sg, particle_name).velocities
        r_from_cartesian = np.sqrt(np.sum(np.power(xyz, 2), axis=1))
        theta_from_cartesian = (
            np.where(r_from_cartesian == 0, 0, np.arcsin(xyz[:, 2] / r_from_cartesian))
            * u.rad
        )
        phi_from_cartesian = np.arctan2(xyz[:, 1], xyz[:, 0])
        phi_from_cartesian = (
            np.where(
                phi_from_cartesian < 0,
                phi_from_cartesian + 2 * np.pi,
                phi_from_cartesian,
            )
            * u.rad
        )
        v_r_from_cartesian = (
            np.cos(theta_from_cartesian) * np.cos(phi_from_cartesian) * vxyz[:, 0]
            + np.cos(theta_from_cartesian) * np.sin(phi_from_cartesian) * vxyz[:, 1]
            + np.sin(theta_from_cartesian) * vxyz[:, 2]
        )
        assert_allclose_units(
            spherical_v_r, v_r_from_cartesian, rtol=1.0e-4, atol=abstol_v
        )

    @pytest.mark.parametrize("particle_name", present_particle_types.values())
    @pytest.mark.parametrize("alias", ("theta", "lat", "latitude", "pol", "polar"))
    @pytest.mark.filterwarnings(
        "ignore:invalid value encountered in true_divide"
    )  # comes from r=0 particle, handled in definition of theta
    def test_spherical_theta(self, sg, particle_name, alias):
        """
        Check spherical polar angle matches direct computation.
        """
        spherical_theta = getattr(
            getattr(sg, particle_name).spherical_coordinates, alias
        )
        xyz = getattr(sg, particle_name).coordinates
        theta_from_cartesian = (
            np.where(
                (xyz == 0).all(axis=1),
                0,
                np.arcsin(xyz[:, 2] / np.sqrt(np.sum(np.power(xyz, 2), axis=1))),
            )
            * u.rad
        )
        assert_allclose_units(
            spherical_theta, theta_from_cartesian, rtol=1.0e-4, atol=abstol_a
        )

    @pytest.mark.parametrize("particle_name", present_particle_types.values())
    @pytest.mark.parametrize("alias", ("theta", "lat", "latitude", "pol", "polar"))
    @pytest.mark.filterwarnings(
        "ignore:invalid value encountered in true_divide"
    )  # comes from r=0 particle, handled in definition of theta
    def test_spherical_velocity_theta(self, sg, particle_name, alias):
        """
        Check spherical polar velocity matches direct computation.
        """
        spherical_v_theta = getattr(
            getattr(sg, particle_name).spherical_velocities, alias
        )
        xyz = getattr(sg, particle_name).coordinates
        vxyz = getattr(sg, particle_name).velocities
        r_from_cartesian = np.sqrt(np.sum(np.power(xyz, 2), axis=1))
        theta_from_cartesian = (
            np.where(r_from_cartesian == 0, 0, np.arcsin(xyz[:, 2] / r_from_cartesian))
            * u.rad
        )
        phi_from_cartesian = np.arctan2(xyz[:, 1], xyz[:, 0])
        phi_from_cartesian = (
            np.where(
                phi_from_cartesian < 0,
                phi_from_cartesian + 2 * np.pi,
                phi_from_cartesian,
            )
            * u.rad
        )
        v_theta_from_cartesian = (
            np.sin(theta_from_cartesian) * np.cos(phi_from_cartesian) * vxyz[:, 0]
            + np.sin(theta_from_cartesian) * np.sin(phi_from_cartesian) * vxyz[:, 1]
            - np.cos(theta_from_cartesian) * vxyz[:, 2]
        )
        assert_allclose_units(
            spherical_v_theta, v_theta_from_cartesian, rtol=1.0e-4, atol=abstol_v
        )

    @pytest.mark.parametrize("particle_name", present_particle_types.values())
    @pytest.mark.parametrize("alias", ("phi", "lon", "longitude", "az", "azimuth"))
    @pytest.mark.filterwarnings(
        "ignore:invalid value encountered in true_divide"
    )  # comes from r=0 particle, handled in definition of theta
    def test_spherical_phi(self, sg, particle_name, alias):
        """
        Check spherical azimuthal angle matches direct computation.
        """
        spherical_phi = getattr(getattr(sg, particle_name).spherical_coordinates, alias)
        xyz = getattr(sg, particle_name).coordinates
        phi_from_cartesian = np.arctan2(xyz[:, 1], xyz[:, 0])
        phi_from_cartesian = (
            np.where(
                phi_from_cartesian < 0,
                phi_from_cartesian + 2 * np.pi,
                phi_from_cartesian,
            )
            * u.rad
        )
        assert_allclose_units(
            spherical_phi, phi_from_cartesian, rtol=1.0e-4, atol=abstol_a
        )

    @pytest.mark.parametrize("particle_name", present_particle_types.values())
    @pytest.mark.parametrize("alias", ("phi", "lon", "longitude", "az", "azimuth"))
    @pytest.mark.filterwarnings(
        "ignore:invalid value encountered in true_divide"
    )  # comes from r=0 particle, handled in definition of theta
    def test_spherical_velocity_phi(self, sg, particle_name, alias):
        """
        Check spherical azimuthal velocity matches direct computation.
        """
        spherical_v_phi = getattr(
            getattr(sg, particle_name).spherical_velocities, alias
        )
        xyz = getattr(sg, particle_name).coordinates
        vxyz = getattr(sg, particle_name).velocities
        phi_from_cartesian = np.arctan2(xyz[:, 1], xyz[:, 0])
        phi_from_cartesian = (
            np.where(
                phi_from_cartesian < 0,
                phi_from_cartesian + 2 * np.pi,
                phi_from_cartesian,
            )
            * u.rad
        )
        v_phi_from_cartesian = (
            -np.sin(phi_from_cartesian) * vxyz[:, 0]
            + np.cos(phi_from_cartesian) * vxyz[:, 1]
        )
        assert_allclose_units(
            spherical_v_phi, v_phi_from_cartesian, rtol=1.0e-4, atol=abstol_v
        )


class TestCylindricalCoordinates:
    @pytest.mark.parametrize("particle_name", present_particle_types.values())
    @pytest.mark.parametrize("alias", ("rho", "R", "radius"))
    def test_cylindrical_rho(self, sg, particle_name, alias):
        """
        Check cylindrical radius matches direct computation.
        """
        spherical_rho = getattr(
            getattr(sg, particle_name).cylindrical_coordinates, alias
        )
        xyz = getattr(sg, particle_name).coordinates
        rho_from_cartesian = np.sqrt(np.sum(np.power(xyz[:, :2], 2), axis=1))
        assert_allclose_units(
            spherical_rho, rho_from_cartesian, rtol=1.0e-4, atol=abstol_c
        )

    @pytest.mark.parametrize("particle_name", present_particle_types.values())
    @pytest.mark.parametrize("alias", ("rho", "R", "radius"))
    def test_cylindrical_velocity_rho(self, sg, particle_name, alias):
        """
        Check cylindrical radial velocity matches direct computation.
        """
        cylindrical_v_rho = getattr(
            getattr(sg, particle_name).cylindrical_velocities, alias
        )
        xyz = getattr(sg, particle_name).coordinates
        vxyz = getattr(sg, particle_name).velocities
        phi_from_cartesian = np.arctan2(xyz[:, 1], xyz[:, 0])
        phi_from_cartesian = (
            np.where(
                phi_from_cartesian < 0,
                phi_from_cartesian + 2 * np.pi,
                phi_from_cartesian,
            )
            * u.rad
        )
        v_rho_from_cartesian = (
            np.cos(phi_from_cartesian) * vxyz[:, 0]
            + np.sin(phi_from_cartesian) * vxyz[:, 1]
        )
        assert_allclose_units(
            cylindrical_v_rho, v_rho_from_cartesian, rtol=1.0e-4, atol=abstol_v
        )

    @pytest.mark.parametrize("particle_name", present_particle_types.values())
    @pytest.mark.parametrize("alias", ("phi", "lon", "longitude", "az", "azimuth"))
    def test_cylindrical_phi(self, sg, particle_name, alias):
        """
        Check that cylindrical azimuthal angle matches direct computation.
        """
        cylindrical_phi = getattr(
            getattr(sg, particle_name).cylindrical_coordinates, alias
        )
        xyz = getattr(sg, particle_name).coordinates
        phi_from_cartesian = np.arctan2(xyz[:, 1], xyz[:, 0])
        phi_from_cartesian = (
            np.where(
                phi_from_cartesian < 0,
                phi_from_cartesian + 2 * np.pi,
                phi_from_cartesian,
            )
            * u.rad
        )
        assert_allclose_units(
            cylindrical_phi, phi_from_cartesian, rtol=1.0e-4, atol=abstol_a
        )

    @pytest.mark.parametrize("particle_name", present_particle_types.values())
    @pytest.mark.parametrize("alias", ("phi", "lon", "longitude", "az", "azimuth"))
    def test_cylindrical_velocity_phi(self, sg, particle_name, alias):
        """
        Check that cylindrical azimuthal velocity matches direct computation.
        """
        cylindrical_v_phi = getattr(
            getattr(sg, particle_name).cylindrical_velocities, alias
        )
        xyz = getattr(sg, particle_name).coordinates
        vxyz = getattr(sg, particle_name).velocities
        phi_from_cartesian = np.arctan2(xyz[:, 1], xyz[:, 0])
        phi_from_cartesian = (
            np.where(
                phi_from_cartesian < 0,
                phi_from_cartesian + 2 * np.pi,
                phi_from_cartesian,
            )
            * u.rad
        )
        v_phi_from_cartesian = (
            -np.sin(phi_from_cartesian) * vxyz[:, 0]
            + np.cos(phi_from_cartesian) * vxyz[:, 1]
        )
        assert_allclose_units(
            cylindrical_v_phi, v_phi_from_cartesian, rtol=1.0e-4, atol=abstol_v
        )

    @pytest.mark.parametrize("particle_name", present_particle_types.values())
    @pytest.mark.parametrize("alias", ("z", "height"))
    def test_cylindrical_z(self, sg, particle_name, alias):
        """
        Check that cylindrical height matches direct computation.
        """
        cylindrical_z = getattr(
            getattr(sg, particle_name).cylindrical_coordinates, alias
        )
        xyz = getattr(sg, particle_name).coordinates
        z_from_cartesian = xyz[:, 2]
        assert_allclose_units(
            cylindrical_z, z_from_cartesian, rtol=1.0e-4, atol=abstol_c
        )

    @pytest.mark.parametrize("particle_name", present_particle_types.values())
    @pytest.mark.parametrize("alias", ("z", "height"))
    def test_cylindrical_velocity_z(self, sg, particle_name, alias):
        cylindrical_v_z = getattr(
            getattr(sg, particle_name).cylindrical_velocities, alias
        )
        vxyz = getattr(sg, particle_name).velocities
        v_z_from_cartesian = vxyz[:, 2]
        assert_allclose_units(
            cylindrical_v_z, v_z_from_cartesian, rtol=1.0e-4, atol=abstol_v
        )


class TestInteractionWithCoordinateTransformations:
    @pytest.mark.parametrize("coordinate_type", ("coordinates", "velocities"))
    @pytest.mark.parametrize(
        "coordinate_system", ("cartesian", "spherical", "cylindrical")
    )
    @pytest.mark.parametrize("particle_name", present_particle_types.values())
    @pytest.mark.parametrize(
        "transform_function, transform_arg",
        (
            ("translate", np.zeros(3) * u.Mpc),
            ("boost", np.zeros(3) * u.km / u.s),
            ("recentre", np.zeros(3) * u.Mpc),
            ("recentre_velocity", np.zeros(3) * u.km / u.s),
            ("rotate", Rotation.from_matrix(np.eye(3))),
        ),
    )
    @pytest.mark.filterwarnings(
        "ignore:invalid value encountered in true_divide"
    )  # comes from r=0 particle, handled in definition of theta
    def test_void_derived_coordinates(
        self,
        sg,
        particle_name,
        coordinate_system,
        coordinate_type,
        transform_function,
        transform_arg,
    ):
        """
        Check that derived coordinates are deleted after transformations.
        """
        # load derived coordinates
        getattr(getattr(sg, particle_name), f"{coordinate_system}_{coordinate_type}")
        # check that they are loaded
        assert (
            getattr(
                getattr(sg, particle_name), f"_{coordinate_system}_{coordinate_type}"
            )
            is not None
        )
        # do a derived coordinate-voiding transformation
        getattr(sg, transform_function)(transform_arg)
        # bypass auto-computation of coordinates
        internal_coords = getattr(
            getattr(sg, particle_name), f"_{coordinate_system}_{coordinate_type}"
        )
        if coordinate_system == "cartesian":
            # coordinates are a view, should not be voided
            assert internal_coords is not None
        elif coordinate_system in ("spherical", "cylindrical"):
            assert internal_coords is None
        else:
            raise NotImplementedError

    @pytest.mark.parametrize("coordinate_type", ("coordinates", "velocities"))
    @pytest.mark.parametrize("particle_name", present_particle_types.values())
    @pytest.mark.parametrize(
        "transform_function, transform_arg",
        (
            ("translate", np.ones(3) * u.Mpc),
            ("boost", 100 * np.ones(3) * u.km / u.s),
            ("recentre", np.ones(3) * u.Mpc),
            ("recentre_velocity", 100 * np.ones(3) * u.km / u.s),
            ("rotate", Rotation.from_rotvec(np.pi / 2 * np.array([1, 1, 1]))),
        ),
    )
    def test_cartesian_coordinates_transform(
        self,
        sg,
        particle_name,
        coordinate_type,
        transform_function,
        transform_arg,
    ):
        """
        Check that cartesian coordinate views update with transformations.
        """
        # load cartesian coordinates
        before = getattr(getattr(sg, particle_name), f"cartesian_{coordinate_type}").xyz
        if coordinate_type == "coordinates":
            tol = abstol_c
            if transform_function == "translate":
                expected = before + transform_arg
            elif transform_function == "boost":
                expected = before
            elif transform_function == "recentre":
                expected = before - transform_arg
            elif transform_function == "recentre_velocity":
                expected = before
            elif transform_function == "rotate":
                expected = before.dot(transform_arg.as_matrix())
        elif coordinate_type == "velocities":
            tol = abstol_v
            if transform_function == "translate":
                expected = before
            elif transform_function == "boost":
                expected = before + transform_arg
            elif transform_function == "recentre":
                expected = before
            elif transform_function == "recentre_velocity":
                expected = before - transform_arg
            elif transform_function == "rotate":
                expected = before.dot(transform_arg.as_matrix())
        # do coordinate transformation
        getattr(sg, transform_function)(transform_arg)
        after = getattr(getattr(sg, particle_name), f"cartesian_{coordinate_type}").xyz
        assert_allclose_units(
            after,
            expected,
            rtol=1.0e-4,
            atol=tol
        )


class TestInteractionWithMasking:
    @pytest.mark.parametrize("coordinate_type", ("coordinates", "velocities"))
    @pytest.mark.parametrize(
        "coordinate_system", ("cartesian", "spherical", "cylindrical")
    )
    @pytest.mark.parametrize("particle_name", present_particle_types.values())
    @pytest.mark.parametrize("before_load", (True, False))
    @pytest.mark.filterwarnings(
        "ignore:invalid value encountered in true_divide"
    )  # comes from r=0 particle, handled in definition of theta
    def test_mask_swiftgalaxy_masks_derived_coordinates(
        self, sg, coordinate_type, coordinate_system, particle_name, before_load
    ):
        """
        Check that when we mask the SWIFTGalaxy, derived coordinates loaded in
        the future are also masked.
        """
        from swiftgalaxy.masks import MaskCollection

        coordinate_names = dict(
            cartesian=("x", "y", "z"),
            spherical=("r", "theta", "phi"),
            cylindrical=("rho", "phi", "z"),
        )[coordinate_system]
        tols = (
            dict(
                cartesian=(abstol_c, abstol_c, abstol_c),
                spherical=(abstol_c, abstol_a, abstol_a),
                cylindrical=(abstol_c, abstol_a, abstol_c),
            )[coordinate_system]
            if coordinate_type == "coordinates"
            else (abstol_v, abstol_v, abstol_v)
        )
        # load derived coordinates to record their values
        getattr(getattr(sg, particle_name), f"{coordinate_system}_{coordinate_type}")
        coordinates_before = {
            coordinate_name: getattr(
                getattr(
                    getattr(sg, particle_name), f"{coordinate_system}_{coordinate_type}"
                ),
                coordinate_name,
            )
            for coordinate_name in coordinate_names
        }
        if before_load:
            # unload derived coordinates
            sg._void_derived_coordinates()
        # mask every second particle
        mask = np.s_[::2]
        sg.mask_particles(MaskCollection(**{particle_name: mask}))
        # load derived coordinates
        getattr(getattr(sg, particle_name), f"{coordinate_system}_{coordinate_type}")
        for coordinate_name, tol in zip(coordinate_names, tols):
            assert_allclose_units(
                getattr(
                    getattr(
                        getattr(sg, particle_name),
                        f"{coordinate_system}_{coordinate_type}",
                    ),
                    coordinate_name,
                ),
                coordinates_before[coordinate_name][mask],
                rtol=1.0e-4,
                atol=tol,
            )

    @pytest.mark.parametrize("coordinate_name", ("r", "theta", "phi"))
    @pytest.mark.parametrize("coordinate_type", ("coordinates", "velocities"))
    @pytest.mark.filterwarnings(
        "ignore:invalid value encountered in true_divide"
    )  # comes from r=0 particle, handled in definition of theta
    def test_single_particle_produces_array_spherical(
        self, sg, coordinate_name, coordinate_type
    ):
        """
        Make sure an array is returned even for a single quantity (some
        calculations tend to produce unyt_quantities, which have shape ()).
        The black hole arrays in the test data have a single particle.
        """
        assert (
            getattr(
                getattr(sg.black_holes, f"spherical_{coordinate_type}"), coordinate_name
            ).shape
            is not tuple()
        )

    @pytest.mark.parametrize("coordinate_name", ("rho", "phi", "z"))
    @pytest.mark.parametrize("coordinate_type", ("coordinates", "velocities"))
    @pytest.mark.filterwarnings(
        "ignore:invalid value encountered in true_divide"
    )  # comes from r=0 particle, handled in definition of theta
    def test_single_particle_produces_array_cylindrical(
        self, sg, coordinate_name, coordinate_type
    ):
        """
        Make sure an array is returned even for a single quantity (some
        calculations tend to produce unyt_quantities, which have shape ()).
        The black hole arrays in the test data have a single particle.
        """
        assert (
            getattr(
                getattr(sg.black_holes, f"cylindrical_{coordinate_type}"),
                coordinate_name,
            ).shape
            is not tuple()
        )
