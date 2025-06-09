import pytest
from pathlib import Path
import numpy as np
import unyt as u
from swiftsimio.objects import cosmo_array, cosmo_factor, a
from swiftgalaxy import (
    SWIFTGalaxy,
    SWIFTGalaxies,
    SOAP,
    Velociraptor,
    Caesar,
    Standalone,
    MaskCollection,
)
from swiftgalaxy.demo_data import (
    _create_toysnap,
    _remove_toysnap,
    _create_toyvr,
    _remove_toyvr,
    _create_toycaesar,
    _remove_toycaesar,
    _create_toysoap,
    _remove_toysoap,
    ToyHF,
    _toysnap_filename,
    _toysoap_filename,
    _toysoap_virtual_snapshot_filename,
    _toyvr_filebase,
    _toycaesar_filename,
    _n_g_b,
    _n_dm_b,
    _centre_1,
    _centre_2,
    _vcentre_1,
    _vcentre_2,
)

hfs = ("vr", "caesar_halo", "caesar_galaxy", "sa", "soap")


@pytest.fixture(scope="function")
def toysnap():
    _create_toysnap()

    yield

    _remove_toysnap()


@pytest.fixture(scope="function")
def toysnap_withfof():
    _create_toysnap(withfof=True)

    yield

    _remove_toysnap()


@pytest.fixture(scope="function")
def toysoap_with_virtual_snapshot():
    _create_toysoap(create_virtual_snapshot=True)

    yield

    _remove_toysoap()


@pytest.fixture(scope="function")
def sg(request):
    _create_toysnap()

    yield SWIFTGalaxy(
        _toysnap_filename,
        ToyHF(),
        transforms_like_coordinates={"coordinates", "extra_coordinates"},
        transforms_like_velocities={"velocities", "extra_velocities"},
    )

    _remove_toysnap()


@pytest.fixture(scope="function")
def sgs(request):
    _create_toysnap()
    yield SWIFTGalaxies(
        _toysnap_filename,
        ToyHF(index=[0, 1]),
        transforms_like_coordinates={"coordinates", "extra_coordinates"},
        transforms_like_velocities={"velocities", "extra_velocities"},
        preload={  # just to keep warnings quiet
            "gas.particle_ids",
            "dark_matter.particle_ids",
            "stars.particle_ids",
            "black_holes.particle_ids",
        },
    )
    _remove_toysnap()


@pytest.fixture(scope="function")
def sg_custom_names():
    toysnap_custom_names_filename = "toysnap_custom_names.hdf5"
    alt_coord_name, alt_vel_name, alt_id_name = "my_coords", "my_vels", "my_ids"

    _create_toysnap(
        snapfile=toysnap_custom_names_filename,
        alt_coord_name="MyCoords",
        alt_vel_name="MyVels",
        alt_id_name="MyIds",
    )

    yield SWIFTGalaxy(
        toysnap_custom_names_filename,
        ToyHF(snapfile=toysnap_custom_names_filename),
        transforms_like_coordinates={alt_coord_name, "extra_coordinates"},
        transforms_like_velocities={alt_vel_name, "extra_velocities"},
        id_particle_dataset_name=alt_id_name,
        coordinates_dataset_name=alt_coord_name,
        velocities_dataset_name=alt_vel_name,
    )

    _remove_toysnap(snapfile=toysnap_custom_names_filename)


@pytest.fixture(scope="function")
def sg_autorecentre_off():
    _create_toysnap()

    yield SWIFTGalaxy(
        _toysnap_filename,
        ToyHF(snapfile=_toysnap_filename),
        transforms_like_coordinates={"coordinates", "extra_coordinates"},
        transforms_like_velocities={"velocities", "extra_velocities"},
        auto_recentre=False,
    )

    _remove_toysnap(snapfile=_toysnap_filename)


@pytest.fixture(scope="function")
def sg_soap():
    _create_toysnap(withfof=True)
    _create_toysoap(create_virtual_snapshot=True)

    yield SWIFTGalaxy(
        _toysoap_virtual_snapshot_filename,
        SOAP(
            soap_file=_toysoap_filename,
            soap_index=0,
        ),
        transforms_like_coordinates={"coordinates", "extra_coordinates"},
        transforms_like_velocities={"velocities", "extra_velocities"},
    )

    _remove_toysnap()
    _remove_toysoap()


@pytest.fixture(scope="function")
def sgs_soap():
    _create_toysnap(withfof=True)
    _create_toysoap(create_virtual_snapshot=True)

    yield SWIFTGalaxies(
        _toysoap_virtual_snapshot_filename,
        SOAP(
            soap_file=_toysoap_filename,
            soap_index=[0, 1],
        ),
        transforms_like_coordinates={"coordinates", "extra_coordinates"},
        transforms_like_velocities={"velocities", "extra_velocities"},
        preload={  # just to keep warnings quiet
            "gas.particle_ids",
            "dark_matter.particle_ids",
            "stars.particle_ids",
            "black_holes.particle_ids",
        },
    )

    _remove_toysnap()
    _remove_toysoap()


@pytest.fixture(scope="function")
def sg_vr():
    _create_toysnap()
    _create_toyvr()

    yield SWIFTGalaxy(
        _toysnap_filename,
        Velociraptor(velociraptor_filebase=_toyvr_filebase, halo_index=0),
        transforms_like_coordinates={"coordinates", "extra_coordinates"},
        transforms_like_velocities={"velocities", "extra_velocities"},
    )

    _remove_toysnap()
    _remove_toyvr()


@pytest.fixture(scope="function")
def sgs_vr():
    _create_toysnap()
    _create_toyvr()

    yield SWIFTGalaxies(
        _toysnap_filename,
        Velociraptor(velociraptor_filebase=_toyvr_filebase, halo_index=[0, 1]),
        transforms_like_coordinates={"coordinates", "extra_coordinates"},
        transforms_like_velocities={"velocities", "extra_velocities"},
        preload={  # just to keep warnings quiet
            "gas.particle_ids",
            "dark_matter.particle_ids",
            "stars.particle_ids",
            "black_holes.particle_ids",
        },
    )

    _remove_toysnap()
    _remove_toyvr()


@pytest.fixture(scope="function", params=["halo", "galaxy"])
def sg_caesar(request):
    _create_toysnap()
    _create_toycaesar()

    yield SWIFTGalaxy(
        _toysnap_filename,
        Caesar(
            caesar_file=_toycaesar_filename, group_type=request.param, group_index=0
        ),
        transforms_like_coordinates={"coordinates", "extra_coordinates"},
        transforms_like_velocities={"velocities", "extra_velocities"},
    )

    _remove_toysnap()
    _remove_toycaesar()


@pytest.fixture(scope="function", params=["halo", "galaxy"])
def sgs_caesar(request):
    _create_toysnap()
    _create_toycaesar()

    yield SWIFTGalaxies(
        _toysnap_filename,
        Caesar(
            caesar_file=_toycaesar_filename,
            group_type=request.param,
            group_index=[0, 1],
        ),
        transforms_like_coordinates={"coordinates", "extra_coordinates"},
        transforms_like_velocities={"velocities", "extra_velocities"},
        preload={  # just to keep warnings quiet
            "gas.particle_ids",
            "dark_matter.particle_ids",
            "stars.particle_ids",
            "black_holes.particle_ids",
        },
    )

    _remove_toysnap()
    _remove_toycaesar()


@pytest.fixture(scope="function")
def soap():
    _create_toysoap()

    yield SOAP(
        soap_file=_toysoap_filename,
        soap_index=0,
    )

    _remove_toysoap()


@pytest.fixture(scope="function")
def soap_multi():
    _create_toysoap()

    yield SOAP(
        soap_file=_toysoap_filename,
        soap_index=[0, 1],
    )

    _remove_toysoap()


@pytest.fixture(scope="function")
def vr():
    _create_toyvr()

    yield Velociraptor(velociraptor_filebase=_toyvr_filebase, halo_index=0)

    _remove_toyvr()


@pytest.fixture(scope="function")
def vr_multi():
    _create_toyvr()

    yield Velociraptor(velociraptor_filebase=_toyvr_filebase, halo_index=[0, 1])

    _remove_toyvr()


@pytest.fixture(scope="function", params=["halo", "galaxy"])
def caesar(request):
    _create_toycaesar()

    yield Caesar(
        caesar_file=_toycaesar_filename, group_type=request.param, group_index=0
    )

    _remove_toycaesar()


@pytest.fixture(scope="function", params=["halo", "galaxy"])
def caesar_multi(request):
    _create_toycaesar()

    yield Caesar(
        caesar_file=_toycaesar_filename, group_type=request.param, group_index=[0, 1]
    )

    _remove_toycaesar()


@pytest.fixture(scope="function")
def sa():
    yield Standalone(
        extra_mask=MaskCollection(
            gas=np.s_[_n_g_b // 2 :],
            dark_matter=np.s_[_n_dm_b // 2 :],
            stars=None,
            black_holes=None,
        ),
        centre=cosmo_array(
            [_centre_1, _centre_1, _centre_1],
            u.Mpc,
            comoving=True,
            cosmo_factor=cosmo_factor(a**1, 1.0),
        ),
        velocity_centre=cosmo_array(
            [_vcentre_1, _vcentre_1, _vcentre_1],
            u.km / u.s,
            comoving=True,
            cosmo_factor=cosmo_factor(a**0, 1.0),
        ),
        spatial_offsets=cosmo_array(
            [[-1, 1], [-1, 1], [-1, 1]],
            u.Mpc,
            comoving=True,
            cosmo_factor=cosmo_factor(a**1, 1.0),
        ),
    )


@pytest.fixture(scope="function")
def sa_multi():
    yield Standalone(
        extra_mask=None,
        centre=cosmo_array(
            [
                [_centre_1, _centre_1, _centre_1],
                [_centre_2, _centre_2, _centre_2],
            ],
            u.Mpc,
            comoving=True,
            cosmo_factor=cosmo_factor(a**1, 1.0),
        ),
        velocity_centre=cosmo_array(
            [
                [_vcentre_1, _vcentre_1, _vcentre_1],
                [_vcentre_2, _vcentre_2, _vcentre_2],
            ],
            u.km / u.s,
            comoving=True,
            cosmo_factor=cosmo_factor(a**0, 1.0),
        ),
        spatial_offsets=cosmo_array(
            [[-1, 1], [-1, 1], [-1, 1]],
            u.Mpc,
            comoving=True,
            cosmo_factor=cosmo_factor(a**1, 1.0),
        ),
    )


@pytest.fixture(scope="function")
def sg_sa():
    _create_toysnap()
    yield SWIFTGalaxy(
        _toysnap_filename,
        Standalone(
            extra_mask=MaskCollection(
                gas=np.s_[_n_g_b // 2 :],
                dark_matter=np.s_[_n_dm_b // 2 :],
                stars=None,
                black_holes=None,
            ),
            centre=cosmo_array(
                [_centre_1, _centre_1, _centre_1],
                u.Mpc,
                comoving=True,
                cosmo_factor=cosmo_factor(a**1, 1.0),
            ),
            velocity_centre=cosmo_array(
                [_vcentre_1, _vcentre_1, _vcentre_1],
                u.km / u.s,
                comoving=True,
                cosmo_factor=cosmo_factor(a**0, 1.0),
            ),
            spatial_offsets=cosmo_array(
                [[-1, 1], [-1, 1], [-1, 1]],
                u.Mpc,
                comoving=True,
                cosmo_factor=cosmo_factor(a**1, 1.0),
            ),
        ),
    )
    _remove_toysnap()


@pytest.fixture(scope="function")
def sgs_sa():
    _create_toysnap()
    yield SWIFTGalaxies(
        _toysnap_filename,
        Standalone(
            extra_mask=None,
            centre=cosmo_array(
                [
                    [_centre_1, _centre_1, _centre_1],
                    [_centre_2, _centre_2, _centre_2],
                ],
                u.Mpc,
                comoving=True,
                cosmo_factor=cosmo_factor(a**1, 1.0),
            ),
            velocity_centre=cosmo_array(
                [
                    [_vcentre_1, _vcentre_1, _vcentre_1],
                    [_vcentre_2, _vcentre_2, _vcentre_2],
                ],
                u.km / u.s,
                comoving=True,
                cosmo_factor=cosmo_factor(a**0, 1.0),
            ),
            spatial_offsets=cosmo_array(
                [[-1, 1], [-1, 1], [-1, 1]],
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
    _remove_toysnap()


@pytest.fixture(scope="function", params=hfs)
def sg_hf(request):
    _create_toysnap(withfof=request.param == "soap")
    if request.param in {"caesar_halo", "caesar_galaxy"}:
        _create_toycaesar()
        yield SWIFTGalaxy(
            _toysnap_filename,
            Caesar(
                caesar_file=_toycaesar_filename,
                group_type=request.param.split("_")[-1],
                group_index=0,
            ),
            transforms_like_coordinates={"coordinates", "extra_coordinates"},
            transforms_like_velocities={"velocities", "extra_velocities"},
        )
        _remove_toycaesar()
    elif request.param == "soap":
        _create_toysoap(create_virtual_snapshot=True)
        yield SWIFTGalaxy(
            _toysoap_virtual_snapshot_filename,
            SOAP(
                soap_file=_toysoap_filename,
                soap_index=0,
            ),
            transforms_like_coordinates={"coordinates", "extra_coordinates"},
            transforms_like_velocities={"velocities", "extra_velocities"},
        )
        _remove_toysoap()
    elif request.param == "vr":
        _create_toyvr()
        yield SWIFTGalaxy(
            _toysnap_filename,
            Velociraptor(velociraptor_filebase=_toyvr_filebase, halo_index=0),
            transforms_like_coordinates={"coordinates", "extra_coordinates"},
            transforms_like_velocities={"velocities", "extra_velocities"},
        )
        _remove_toyvr()
    elif request.param == "sa":
        yield Standalone(
            extra_mask=MaskCollection(
                gas=np.s_[_n_g_b // 2 :],
                dark_matter=np.s_[_n_dm_b // 2 :],
                stars=None,
                black_holes=None,
            ),
            centre=cosmo_array(
                [_centre_1, _centre_1, _centre_1],
                u.Mpc,
                comoving=True,
                cosmo_factor=cosmo_factor(a**1, 1.0),
            ),
            velocity_centre=cosmo_array(
                [_vcentre_1, _vcentre_1, _vcentre_1],
                u.km / u.s,
                comoving=True,
                cosmo_factor=cosmo_factor(a**0, 1.0),
            ),
            spatial_offsets=cosmo_array(
                [[-1, 1], [-1, 1], [-1, 1]],
                u.Mpc,
                comoving=True,
                cosmo_factor=cosmo_factor(a**1, 1.0),
            ),
        )
    _remove_toysnap()


@pytest.fixture(scope="function", params=hfs)
def hf(request):
    if request.param in {"caesar_halo", "caesar_galaxy"}:
        _create_toycaesar()

        yield Caesar(
            caesar_file=_toycaesar_filename,
            group_type=request.param.split("_")[-1],
            group_index=0,
        )

        _remove_toycaesar()
    elif request.param == "soap":
        _create_toysoap()

        yield SOAP(
            soap_file=_toysoap_filename,
            soap_index=0,
        )

        _remove_toysoap()
    elif request.param == "vr":
        _create_toyvr()

        yield Velociraptor(velociraptor_filebase=_toyvr_filebase, halo_index=0)

        _remove_toyvr()
    elif request.param == "sa":
        yield Standalone(
            extra_mask=MaskCollection(
                gas=np.s_[_n_g_b // 2 :],
                dark_matter=np.s_[_n_dm_b // 2 :],
                stars=None,
                black_holes=None,
            ),
            centre=cosmo_array(
                [_centre_1, _centre_1, _centre_1],
                u.Mpc,
                comoving=True,
                cosmo_factor=cosmo_factor(a**1, 1.0),
            ),
            velocity_centre=cosmo_array(
                [_vcentre_1, _vcentre_1, _vcentre_1],
                u.km / u.s,
                comoving=True,
                cosmo_factor=cosmo_factor(a**0, 1.0),
            ),
            spatial_offsets=cosmo_array(
                [[-1, 1], [-1, 1], [-1, 1]],
                u.Mpc,
                comoving=True,
                cosmo_factor=cosmo_factor(a**1, 1.0),
            ),
        )


@pytest.fixture(scope="function", params=hfs)
def hf_multi(request):
    if request.param in {"caesar_halo", "caesar_galaxy"}:
        _create_toycaesar()

        yield Caesar(
            caesar_file=_toycaesar_filename,
            group_type=request.param.split("_")[-1],
            group_index=[0, 1],
        )

        _remove_toycaesar()
    elif request.param == "soap":
        _create_toysoap(create_virtual_snapshot=Path(_toysnap_filename).is_file())

        yield SOAP(
            soap_file=_toysoap_filename,
            soap_index=[0, 1],
        )

        _remove_toysoap()
    elif request.param == "vr":
        _create_toyvr()

        yield Velociraptor(velociraptor_filebase=_toyvr_filebase, halo_index=[0, 1])

        _remove_toyvr()
    elif request.param == "sa":
        yield Standalone(
            extra_mask=None,
            centre=cosmo_array(
                [
                    [_centre_1, _centre_1, _centre_1],
                    [_centre_2, _centre_2, _centre_2],
                ],
                u.Mpc,
                comoving=True,
                cosmo_factor=cosmo_factor(a**1, 1.0),
            ),
            velocity_centre=cosmo_array(
                [
                    [_vcentre_1, _vcentre_1, _vcentre_1],
                    [_vcentre_2, _vcentre_2, _vcentre_2],
                ],
                u.km / u.s,
                comoving=True,
                cosmo_factor=cosmo_factor(a**0, 1.0),
            ),
            spatial_offsets=cosmo_array(
                [[-1, 1], [-1, 1], [-1, 1]],
                u.Mpc,
                comoving=True,
                cosmo_factor=cosmo_factor(a**1, 1.0),
            ),
        )


@pytest.fixture(scope="function", params=hfs)
def hf_multi_forwards_and_backwards(request):
    if request.param in {"caesar_halo", "caesar_galaxy"}:
        _create_toycaesar()

        yield Caesar(
            caesar_file=_toycaesar_filename,
            group_type=request.param.split("_")[-1],
            group_index=[0, 1],
        ), Caesar(
            caesar_file=_toycaesar_filename,
            group_type=request.param.split("_")[-1],
            group_index=[1, 0],
        )

        _remove_toycaesar()
    elif request.param == "soap":
        _create_toysoap(create_virtual_snapshot=Path(_toysnap_filename).is_file())

        yield SOAP(
            soap_file=_toysoap_filename,
            soap_index=[0, 1],
        ), SOAP(
            soap_file=_toysoap_filename,
            soap_index=[1, 0],
        )

        _remove_toysoap()
    elif request.param == "vr":
        _create_toyvr()

        yield Velociraptor(
            velociraptor_filebase=_toyvr_filebase, halo_index=[0, 1]
        ), Velociraptor(velociraptor_filebase=_toyvr_filebase, halo_index=[1, 0])

        _remove_toyvr()
    elif request.param == "sa":
        yield Standalone(
            extra_mask=None,
            centre=cosmo_array(
                [
                    [_centre_1, _centre_1, _centre_1],
                    [_centre_2, _centre_2, _centre_2],
                ],
                u.Mpc,
                comoving=True,
                cosmo_factor=cosmo_factor(a**1, 1.0),
            ),
            velocity_centre=cosmo_array(
                [
                    [_vcentre_1, _vcentre_1, _vcentre_1],
                    [_vcentre_2, _vcentre_2, _vcentre_2],
                ],
                u.km / u.s,
                comoving=True,
                cosmo_factor=cosmo_factor(a**0, 1.0),
            ),
            spatial_offsets=cosmo_array(
                [[-1, 1], [-1, 1], [-1, 1]],
                u.Mpc,
                comoving=True,
                cosmo_factor=cosmo_factor(a**1, 1.0),
            ),
        ), Standalone(
            extra_mask=None,
            centre=cosmo_array(
                [
                    [_centre_2, _centre_2, _centre_2],
                    [_centre_1, _centre_1, _centre_1],
                ],
                u.Mpc,
                comoving=True,
                cosmo_factor=cosmo_factor(a**1, 1.0),
            ),
            velocity_centre=cosmo_array(
                [
                    [_vcentre_2, _vcentre_2, _vcentre_2],
                    [_vcentre_1, _vcentre_1, _vcentre_1],
                ],
                u.km / u.s,
                comoving=True,
                cosmo_factor=cosmo_factor(a**0, 1.0),
            ),
            spatial_offsets=cosmo_array(
                [[-1, 1], [-1, 1], [-1, 1]],
                u.Mpc,
                comoving=True,
                cosmo_factor=cosmo_factor(a**1, 1.0),
            ),
        )


@pytest.fixture(scope="function", params=hfs)
def hf_multi_onetarget(request):
    if request.param in {"caesar_halo", "caesar_galaxy"}:
        _create_toycaesar()

        yield Caesar(
            caesar_file=_toycaesar_filename,
            group_type=request.param.split("_")[-1],
            group_index=[1],
        )

        _remove_toycaesar()
    elif request.param == "soap":
        _create_toysoap(create_virtual_snapshot=Path(_toysnap_filename).is_file())

        yield SOAP(
            soap_file=_toysoap_filename,
            soap_index=[1],
        )

        _remove_toysoap()
    elif request.param == "vr":
        _create_toyvr()

        yield Velociraptor(velociraptor_filebase=_toyvr_filebase, halo_index=[1])

        _remove_toyvr()
    elif request.param == "sa":
        yield Standalone(
            extra_mask=None,
            centre=cosmo_array(
                [
                    [_centre_2, _centre_2, _centre_2],
                ],
                u.Mpc,
                comoving=True,
                cosmo_factor=cosmo_factor(a**1, 1.0),
            ),
            velocity_centre=cosmo_array(
                [
                    [_vcentre_2, _vcentre_2, _vcentre_2],
                ],
                u.km / u.s,
                comoving=True,
                cosmo_factor=cosmo_factor(a**0, 1.0),
            ),
            spatial_offsets=cosmo_array(
                [[-1, 1], [-1, 1], [-1, 1]],
                u.Mpc,
                comoving=True,
                cosmo_factor=cosmo_factor(a**1, 1.0),
            ),
        )


@pytest.fixture(scope="function", params=hfs)
def hf_multi_zerotarget(request):
    if request.param in {"caesar_halo", "caesar_galaxy"}:
        _create_toycaesar()

        yield Caesar(
            caesar_file=_toycaesar_filename,
            group_type=request.param.split("_")[-1],
            group_index=[],
        )

        _remove_toycaesar()
    elif request.param == "soap":
        _create_toysoap(create_virtual_snapshot=Path(_toysnap_filename).is_file())

        yield SOAP(
            soap_file=_toysoap_filename,
            soap_index=[],
        )

        _remove_toysoap()
    elif request.param == "vr":
        _create_toyvr()

        yield Velociraptor(velociraptor_filebase=_toyvr_filebase, halo_index=[])

        _remove_toyvr()
    elif request.param == "sa":
        yield Standalone(
            extra_mask=None,
            centre=cosmo_array(
                [],
                u.Mpc,
                comoving=True,
                cosmo_factor=cosmo_factor(a**1, 1.0),
            ),
            velocity_centre=cosmo_array(
                [],
                u.km / u.s,
                comoving=True,
                cosmo_factor=cosmo_factor(a**0, 1.0),
            ),
            spatial_offsets=cosmo_array(
                [[-1, 1], [-1, 1], [-1, 1]],
                u.Mpc,
                comoving=True,
                cosmo_factor=cosmo_factor(a**1, 1.0),
            ),
        )
