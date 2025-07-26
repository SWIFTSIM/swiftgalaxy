import pytest
from pathlib import Path
import numpy as np
import unyt as u
from swiftsimio.objects import cosmo_array
from swiftgalaxy import (
    SWIFTGalaxy,
    SWIFTGalaxies,
    SOAP,
    Velociraptor,
    Caesar,
    Standalone,
    MaskCollection,
)
from swiftgalaxy.masks import LazyMask
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
    _toysoap_membership_filebase,
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
def toysnap(tmp_path_factory):
    toysnap_filename = (
        tmp_path_factory.mktemp(_toysnap_filename.parent) / _toysnap_filename.name
    )
    _create_toysnap(snapfile=toysnap_filename)

    yield {"toysnap_filename": toysnap_filename}

    _remove_toysnap(snapfile=toysnap_filename)


@pytest.fixture(scope="function")
def toysnap_withfof(tmp_path_factory):
    toysnap_filename = (
        tmp_path_factory.mktemp(_toysnap_filename.parent) / _toysnap_filename.name
    )
    _create_toysnap(snapfile=toysnap_filename, withfof=True)

    yield {"toysnap_filename": toysnap_filename}

    _remove_toysnap(snapfile=toysnap_filename)


@pytest.fixture(scope="function")
def toysoap_with_virtual_snapshot(tmp_path_factory):
    tp = tmp_path_factory.mktemp(_toysoap_filename.parent)
    toysoap_filename = tp / _toysoap_filename.name
    membership_filebase = tp / _toysoap_membership_filebase.name
    toysnap_filename = tp / _toysnap_filename.name
    toysoap_virtual_snapshot_filename = tp / _toysoap_virtual_snapshot_filename.name
    _create_toysnap(toysnap_filename, withfof=True)
    _create_toysoap(
        filename=toysoap_filename,
        membership_filebase=membership_filebase,
        create_virtual_snapshot=True,
        create_virtual_snapshot_from=toysnap_filename,
        virtual_snapshot_filename=toysoap_virtual_snapshot_filename,
    )

    yield {
        "toysoap_filename": toysoap_filename,
        "membership_filebase": membership_filebase,
        "toysnap_filename": toysnap_filename,
        "toysoap_virtual_snapshot_filename": toysoap_virtual_snapshot_filename,
    }

    _remove_toysoap(
        filename=toysoap_filename,
        membership_filebase=membership_filebase,
        virtual_snapshot_filename=toysoap_virtual_snapshot_filename,
    )
    _remove_toysnap(snapfile=toysnap_filename)


@pytest.fixture(scope="function")
def sg(request, tmp_path_factory):
    toysnap_filename = (
        tmp_path_factory.mktemp(_toysnap_filename.parent) / _toysnap_filename.name
    )
    _create_toysnap(snapfile=toysnap_filename)

    yield SWIFTGalaxy(
        toysnap_filename,
        ToyHF(snapfile=toysnap_filename),
        transforms_like_coordinates={"coordinates", "extra_coordinates"},
        transforms_like_velocities={"velocities", "extra_velocities"},
    )

    _remove_toysnap(snapfile=toysnap_filename)


@pytest.fixture(scope="function")
def sgs(request, tmp_path_factory):
    toysnap_filename = (
        tmp_path_factory.mktemp(_toysnap_filename.parent) / _toysnap_filename.name
    )
    _create_toysnap(snapfile=toysnap_filename)
    yield SWIFTGalaxies(
        toysnap_filename,
        ToyHF(snapfile=toysnap_filename, index=[0, 1]),
        transforms_like_coordinates={"coordinates", "extra_coordinates"},
        transforms_like_velocities={"velocities", "extra_velocities"},
        preload={  # just to keep warnings quiet
            "gas.particle_ids",
            "dark_matter.particle_ids",
            "stars.particle_ids",
            "black_holes.particle_ids",
        },
    )
    _remove_toysnap(snapfile=toysnap_filename)


@pytest.fixture(scope="function")
def sg_custom_names(tmp_path_factory):
    tp = tmp_path_factory.mktemp(_toysnap_filename.parent)
    toysnap_custom_names_filename = tp / "toysnap_custom_names.hdf5"
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
def sg_autorecentre_off(tmp_path_factory):
    toysnap_filename = (
        tmp_path_factory.mktemp(_toysnap_filename.parent) / _toysnap_filename.name
    )
    _create_toysnap(snapfile=toysnap_filename)

    yield SWIFTGalaxy(
        toysnap_filename,
        ToyHF(snapfile=toysnap_filename),
        transforms_like_coordinates={"coordinates", "extra_coordinates"},
        transforms_like_velocities={"velocities", "extra_velocities"},
        auto_recentre=False,
    )

    _remove_toysnap(snapfile=toysnap_filename)


@pytest.fixture(scope="function")
def sg_soap(tmp_path_factory):
    tp = tmp_path_factory.mktemp(_toysnap_filename.parent)
    toysnap_filename = tp / _toysnap_filename.name
    toysoap_filename = tp / _toysoap_filename.name
    membership_filebase = tp / _toysoap_membership_filebase.name
    toysoap_virtual_snapshot_filename = tp / _toysoap_virtual_snapshot_filename.name
    _create_toysnap(snapfile=toysnap_filename, withfof=True)
    _create_toysoap(
        filename=toysoap_filename,
        membership_filebase=membership_filebase,
        create_virtual_snapshot=True,
        create_virtual_snapshot_from=toysnap_filename,
        virtual_snapshot_filename=toysoap_virtual_snapshot_filename,
    )

    yield SWIFTGalaxy(
        toysoap_virtual_snapshot_filename,
        SOAP(
            soap_file=toysoap_filename,
            soap_index=0,
        ),
        transforms_like_coordinates={"coordinates", "extra_coordinates"},
        transforms_like_velocities={"velocities", "extra_velocities"},
    )

    _remove_toysnap(snapfile=toysnap_filename)
    _remove_toysoap(
        filename=toysoap_filename,
        membership_filebase=membership_filebase,
        virtual_snapshot_filename=toysoap_virtual_snapshot_filename,
    )


@pytest.fixture(scope="function")
def sgs_soap(tmp_path_factory):
    tp = tmp_path_factory.mktemp(_toysnap_filename.parent)
    toysnap_filename = tp / _toysnap_filename.name
    membership_filebase = tp / _toysoap_membership_filebase.name
    toysoap_filename = tp / _toysoap_filename.name
    toysoap_virtual_snapshot_filename = tp / _toysoap_virtual_snapshot_filename.name
    _create_toysnap(snapfile=toysnap_filename, withfof=True)
    _create_toysoap(
        filename=toysoap_filename,
        membership_filebase=membership_filebase,
        create_virtual_snapshot=True,
        create_virtual_snapshot_from=toysnap_filename,
        virtual_snapshot_filename=toysoap_virtual_snapshot_filename,
    )

    yield SWIFTGalaxies(
        toysoap_virtual_snapshot_filename,
        SOAP(
            soap_file=toysoap_filename,
            soap_index=[0, 1],
        ),
        transforms_like_coordinates={"coordinates", "extra_coordinates"},
        transforms_like_velocities={"velocities", "extra_velocities"},
        preload={  # just to keep warnings quiet
            "gas.particle_ids",
            "dark_matter.particle_ids",
        },
    )

    _remove_toysnap(snapfile=toysnap_filename)
    _remove_toysoap(
        filename=toysoap_filename,
        membership_filebase=membership_filebase,
        virtual_snapshot_filename=toysoap_virtual_snapshot_filename,
    )


@pytest.fixture(scope="function")
def sg_vr(tmp_path_factory):
    tp = tmp_path_factory.mktemp(_toysnap_filename.parent)
    toysnap_filename = tp / _toysnap_filename.name
    toyvr_filebase = tp / _toyvr_filebase.name
    _create_toysnap(snapfile=toysnap_filename)
    _create_toyvr(filebase=toyvr_filebase)

    yield SWIFTGalaxy(
        toysnap_filename,
        Velociraptor(velociraptor_filebase=toyvr_filebase, halo_index=0),
        transforms_like_coordinates={"coordinates", "extra_coordinates"},
        transforms_like_velocities={"velocities", "extra_velocities"},
    )

    _remove_toysnap(snapfile=toysnap_filename)
    _remove_toyvr(filebase=toyvr_filebase)


@pytest.fixture(scope="function")
def sgs_vr(tmp_path_factory):
    tp = tmp_path_factory.mktemp(_toysnap_filename.parent)
    toysnap_filename = tp / _toysnap_filename.name
    toyvr_filebase = tp / _toyvr_filebase.name
    _create_toysnap(snapfile=toysnap_filename)
    _create_toyvr(filebase=toyvr_filebase)

    yield SWIFTGalaxies(
        toysnap_filename,
        Velociraptor(velociraptor_filebase=toyvr_filebase, halo_index=[0, 1]),
        transforms_like_coordinates={"coordinates", "extra_coordinates"},
        transforms_like_velocities={"velocities", "extra_velocities"},
        preload={  # just to keep warnings quiet
            "gas.particle_ids",
            "dark_matter.particle_ids",
        },
    )

    _remove_toysnap(snapfile=toysnap_filename)
    _remove_toyvr(filebase=toyvr_filebase)


@pytest.fixture(scope="function", params=["halo", "galaxy"])
def sg_caesar(request, tmp_path_factory):
    tp = tmp_path_factory.mktemp(_toysnap_filename.parent)
    toysnap_filename = tp / _toysnap_filename.name
    toycaesar_filename = tp / _toycaesar_filename.name
    _create_toysnap(snapfile=toysnap_filename)
    _create_toycaesar(filename=toycaesar_filename)

    yield SWIFTGalaxy(
        toysnap_filename,
        Caesar(caesar_file=toycaesar_filename, group_type=request.param, group_index=0),
        transforms_like_coordinates={"coordinates", "extra_coordinates"},
        transforms_like_velocities={"velocities", "extra_velocities"},
    )

    _remove_toysnap(snapfile=toysnap_filename)
    _remove_toycaesar(filename=toycaesar_filename)


@pytest.fixture(scope="function", params=["halo", "galaxy"])
def sgs_caesar(request, tmp_path_factory):
    tp = tmp_path_factory.mktemp(_toysnap_filename.parent)
    toysnap_filename = tp / _toysnap_filename.name
    toycaesar_filename = tp / _toycaesar_filename.name
    _create_toysnap(snapfile=toysnap_filename)
    _create_toycaesar(filename=toycaesar_filename)

    yield SWIFTGalaxies(
        toysnap_filename,
        Caesar(
            caesar_file=toycaesar_filename,
            group_type=request.param,
            group_index=[0, 1],
        ),
        transforms_like_coordinates={"coordinates", "extra_coordinates"},
        transforms_like_velocities={"velocities", "extra_velocities"},
        preload={  # just to keep warnings quiet
            "gas.particle_ids",
            "dark_matter.particle_ids",
        },
    )

    _remove_toysnap(snapfile=toysnap_filename)
    _remove_toycaesar(filename=toycaesar_filename)


@pytest.fixture(scope="function")
def soap(tmp_path_factory):
    toysoap_filename = (
        tmp_path_factory.mktemp(_toysnap_filename.parent) / _toysoap_filename.name
    )
    _create_toysoap(filename=toysoap_filename, create_membership=False)

    yield SOAP(
        soap_file=toysoap_filename,
        soap_index=0,
    )

    _remove_toysoap(
        filename=toysoap_filename,
    )


@pytest.fixture(scope="function")
def soap_multi(tmp_path_factory):
    tp = tmp_path_factory.mktemp(_toysnap_filename.parent)
    toysoap_filename = tp / _toysoap_filename.name
    membership_filebase = tp / _toysoap_membership_filebase
    _create_toysoap(filename=toysoap_filename, membership_filebase=membership_filebase)

    yield SOAP(
        soap_file=toysoap_filename,
        soap_index=[0, 1],
    )

    _remove_toysoap(
        filename=toysoap_filename,
        membership_filebase=membership_filebase,
    )


@pytest.fixture(scope="function")
def vr(tmp_path_factory):
    toyvr_filebase = (
        tmp_path_factory.mktemp(_toyvr_filebase.parent) / _toyvr_filebase.name
    )
    _create_toyvr(filebase=toyvr_filebase)

    yield Velociraptor(velociraptor_filebase=toyvr_filebase, halo_index=0)

    _remove_toyvr(filebase=toyvr_filebase)


@pytest.fixture(scope="function")
def vr_multi(tmp_path_factory):
    toyvr_filebase = (
        tmp_path_factory.mktemp(_toyvr_filebase.parent) / _toyvr_filebase.name
    )
    _create_toyvr(filebase=toyvr_filebase)

    yield Velociraptor(velociraptor_filebase=toyvr_filebase, halo_index=[0, 1])

    _remove_toyvr(filebase=toyvr_filebase)


@pytest.fixture(scope="function", params=["halo", "galaxy"])
def caesar(request, tmp_path_factory):
    toycaesar_filename = (
        tmp_path_factory.mktemp(_toycaesar_filename.parent) / _toycaesar_filename.name
    )
    _create_toycaesar(filename=toycaesar_filename)

    yield Caesar(
        caesar_file=toycaesar_filename, group_type=request.param, group_index=0
    )

    _remove_toycaesar(filename=toycaesar_filename)


@pytest.fixture(scope="function", params=["halo", "galaxy"])
def caesar_multi(request, tmp_path_factory):
    toycaesar_filename = (
        tmp_path_factory.mktemp(_toycaesar_filename.parent) / _toycaesar_filename.name
    )
    _create_toycaesar(filename=toycaesar_filename)

    yield Caesar(
        caesar_file=toycaesar_filename, group_type=request.param, group_index=[0, 1]
    )

    _remove_toycaesar(filename=toycaesar_filename)


@pytest.fixture(scope="function")
def sa(tmp_path_factory):
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
            scale_factor=1.0,
            scale_exponent=1,
        ),
        velocity_centre=cosmo_array(
            [_vcentre_1, _vcentre_1, _vcentre_1],
            u.km / u.s,
            comoving=True,
            scale_factor=1.0,
            scale_exponent=0,
        ),
        spatial_offsets=cosmo_array(
            [[-1, 1], [-1, 1], [-1, 1]],
            u.Mpc,
            comoving=True,
            scale_factor=1.0,
            scale_exponent=1,
        ),
    )


@pytest.fixture(scope="function")
def sa_multi(tmp_path_factory):
    yield Standalone(
        extra_mask=None,
        centre=cosmo_array(
            [
                [_centre_1, _centre_1, _centre_1],
                [_centre_2, _centre_2, _centre_2],
            ],
            u.Mpc,
            comoving=True,
            scale_factor=1.0,
            scale_exponent=1,
        ),
        velocity_centre=cosmo_array(
            [
                [_vcentre_1, _vcentre_1, _vcentre_1],
                [_vcentre_2, _vcentre_2, _vcentre_2],
            ],
            u.km / u.s,
            comoving=True,
            scale_factor=1.0,
            scale_exponent=0,
        ),
        spatial_offsets=cosmo_array(
            [[-1, 1], [-1, 1], [-1, 1]],
            u.Mpc,
            comoving=True,
            scale_factor=1.0,
            scale_exponent=1,
        ),
    )


@pytest.fixture(scope="function")
def sg_sa(tmp_path_factory):
    tp = tmp_path_factory.mktemp(_toysnap_filename.parent)
    toysnap_filename = tp / _toysnap_filename.name
    _create_toysnap(snapfile=toysnap_filename)
    yield SWIFTGalaxy(
        toysnap_filename,
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
                scale_factor=1.0,
                scale_exponent=1,
            ),
            velocity_centre=cosmo_array(
                [_vcentre_1, _vcentre_1, _vcentre_1],
                u.km / u.s,
                comoving=True,
                scale_factor=1.0,
                scale_exponent=0,
            ),
            spatial_offsets=cosmo_array(
                [[-1, 1], [-1, 1], [-1, 1]],
                u.Mpc,
                comoving=True,
                scale_factor=1.0,
                scale_exponent=1,
            ),
        ),
    )
    _remove_toysnap(snapfile=toysnap_filename)


@pytest.fixture(scope="function")
def sgs_sa(tmp_path_factory):
    tp = tmp_path_factory.mktemp(_toysnap_filename.parent)
    toysnap_filename = tp / _toysnap_filename.name
    _create_toysnap(snapfile=toysnap_filename)
    yield SWIFTGalaxies(
        toysnap_filename,
        Standalone(
            extra_mask=None,
            centre=cosmo_array(
                [
                    [_centre_1, _centre_1, _centre_1],
                    [_centre_2, _centre_2, _centre_2],
                ],
                u.Mpc,
                comoving=True,
                scale_factor=1.0,
                scale_exponent=1,
            ),
            velocity_centre=cosmo_array(
                [
                    [_vcentre_1, _vcentre_1, _vcentre_1],
                    [_vcentre_2, _vcentre_2, _vcentre_2],
                ],
                u.km / u.s,
                comoving=True,
                scale_factor=1.0,
                scale_exponent=0,
            ),
            spatial_offsets=cosmo_array(
                [[-1, 1], [-1, 1], [-1, 1]],
                u.Mpc,
                comoving=True,
                scale_factor=1.0,
                scale_exponent=1,
            ),
        ),
        preload={  # just to keep warnings quiet
            "gas.particle_ids",
            "dark_matter.particle_ids",
        },
    )
    _remove_toysnap(snapfile=toysnap_filename)


@pytest.fixture(scope="function", params=hfs)
def sg_hf(request, tmp_path_factory):
    tp = tmp_path_factory.mktemp(_toysnap_filename.parent)
    toysnap_filename = tp / _toysnap_filename.name
    _create_toysnap(snapfile=toysnap_filename, withfof=request.param == "soap")
    if request.param in {"caesar_halo", "caesar_galaxy"}:
        toycaesar_filename = tp / _toycaesar_filename.name
        _create_toycaesar(filename=toycaesar_filename)
        yield SWIFTGalaxy(
            toysnap_filename,
            Caesar(
                caesar_file=toycaesar_filename,
                group_type=request.param.split("_")[-1],
                group_index=0,
            ),
            transforms_like_coordinates={"coordinates", "extra_coordinates"},
            transforms_like_velocities={"velocities", "extra_velocities"},
        )
        _remove_toycaesar(filename=toycaesar_filename)
    elif request.param == "soap":
        toysoap_filename = tp / _toysoap_filename.name
        membership_filebase = tp / _toysoap_membership_filebase.name
        toysoap_virtual_snapshot_filename = tp / _toysoap_virtual_snapshot_filename.name
        _create_toysoap(
            filename=toysoap_filename,
            membership_filebase=membership_filebase,
            create_virtual_snapshot=True,
            create_virtual_snapshot_from=toysnap_filename,
            virtual_snapshot_filename=toysoap_virtual_snapshot_filename,
        )
        yield SWIFTGalaxy(
            toysoap_virtual_snapshot_filename,
            SOAP(
                soap_file=toysoap_filename,
                soap_index=0,
            ),
            transforms_like_coordinates={"coordinates", "extra_coordinates"},
            transforms_like_velocities={"velocities", "extra_velocities"},
        )
        _remove_toysoap(
            filename=toysoap_filename,
            membership_filebase=membership_filebase,
            virtual_snapshot_filename=toysoap_virtual_snapshot_filename,
        )
    elif request.param == "vr":
        toyvr_filebase = tp / _toyvr_filebase.name
        _create_toyvr(filebase=toyvr_filebase)
        yield SWIFTGalaxy(
            toysnap_filename,
            Velociraptor(velociraptor_filebase=toyvr_filebase, halo_index=0),
            transforms_like_coordinates={"coordinates", "extra_coordinates"},
            transforms_like_velocities={"velocities", "extra_velocities"},
        )
        _remove_toyvr(filebase=toyvr_filebase)
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
                scale_factor=1.0,
                scale_exponent=1,
            ),
            velocity_centre=cosmo_array(
                [_vcentre_1, _vcentre_1, _vcentre_1],
                u.km / u.s,
                comoving=True,
                scale_factor=1.0,
                scale_exponent=0,
            ),
            spatial_offsets=cosmo_array(
                [[-1, 1], [-1, 1], [-1, 1]],
                u.Mpc,
                comoving=True,
                scale_factor=1.0,
                scale_exponent=1,
            ),
        )
        _remove_toysnap(snapfile=toysnap_filename)


@pytest.fixture(scope="function", params=hfs)
def hf(request, tmp_path_factory):
    tp = tmp_path_factory.mktemp(_toysnap_filename.parent)
    if request.param in {"caesar_halo", "caesar_galaxy"}:
        toycaesar_filename = tp / _toycaesar_filename.name
        _create_toycaesar(filename=toycaesar_filename)

        yield Caesar(
            caesar_file=toycaesar_filename,
            group_type=request.param.split("_")[-1],
            group_index=0,
        )

        _remove_toycaesar(filename=toycaesar_filename)
    elif request.param == "soap":
        toysoap_filename = tp / _toysoap_filename.name
        membership_filebase = tp / _toysoap_membership_filebase.name
        _create_toysoap(
            filename=toysoap_filename, membership_filebase=membership_filebase
        )

        yield SOAP(
            soap_file=toysoap_filename,
            soap_index=0,
        )

        _remove_toysoap(
            filename=toysoap_filename,
            membership_filebase=membership_filebase,
        )
    elif request.param == "vr":
        toyvr_filebase = tp / _toyvr_filebase.name
        _create_toyvr(filebase=toyvr_filebase)

        yield Velociraptor(velociraptor_filebase=toyvr_filebase, halo_index=0)

        _remove_toyvr(filebase=toyvr_filebase)
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
                scale_factor=1.0,
                scale_exponent=1,
            ),
            velocity_centre=cosmo_array(
                [_vcentre_1, _vcentre_1, _vcentre_1],
                u.km / u.s,
                comoving=True,
                scale_factor=1.0,
                scale_exponent=0,
            ),
            spatial_offsets=cosmo_array(
                [[-1, 1], [-1, 1], [-1, 1]],
                u.Mpc,
                comoving=True,
                scale_factor=1.0,
                scale_exponent=1,
            ),
        )


@pytest.fixture(scope="function", params=hfs)
def hf_multi(request, tmp_path_factory):
    tp = tmp_path_factory.mktemp(_toysnap_filename.parent)
    toysnap_filename = tp / _toysnap_filename.name
    _create_toysnap(snapfile=toysnap_filename, withfof=request.param == "soap")
    if request.param in {"caesar_halo", "caesar_galaxy"}:
        toycaesar_filename = tp / _toycaesar_filename.name
        _create_toycaesar(filename=toycaesar_filename)

        yield Caesar(
            caesar_file=toycaesar_filename,
            group_type=request.param.split("_")[-1],
            group_index=[0, 1],
        )

        _remove_toycaesar(filename=toycaesar_filename)
    elif request.param == "soap":
        toysoap_filename = tp / _toysoap_filename.name
        membership_filebase = tp / _toysoap_membership_filebase.name
        toysoap_virtual_snapshot_filename = tp / _toysoap_virtual_snapshot_filename.name
        _create_toysoap(
            filename=toysoap_filename,
            membership_filebase=membership_filebase,
            create_virtual_snapshot=Path(toysnap_filename).is_file(),
            create_virtual_snapshot_from=toysnap_filename,
            virtual_snapshot_filename=toysoap_virtual_snapshot_filename,
        )

        yield SOAP(
            soap_file=toysoap_filename,
            soap_index=[0, 1],
        )

        _remove_toysoap(
            filename=toysoap_filename,
            membership_filebase=membership_filebase,
            virtual_snapshot_filename=toysoap_virtual_snapshot_filename,
        )

    elif request.param == "vr":
        toyvr_filebase = tp / _toyvr_filebase.name
        _create_toyvr(filebase=toyvr_filebase)

        yield Velociraptor(velociraptor_filebase=toyvr_filebase, halo_index=[0, 1])

        _remove_toyvr(filebase=toyvr_filebase)
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
                scale_factor=1.0,
                scale_exponent=1,
            ),
            velocity_centre=cosmo_array(
                [
                    [_vcentre_1, _vcentre_1, _vcentre_1],
                    [_vcentre_2, _vcentre_2, _vcentre_2],
                ],
                u.km / u.s,
                comoving=True,
                scale_factor=1.0,
                scale_exponent=0,
            ),
            spatial_offsets=cosmo_array(
                [[-1, 1], [-1, 1], [-1, 1]],
                u.Mpc,
                comoving=True,
                scale_factor=1.0,
                scale_exponent=1,
            ),
        )
    _remove_toysnap(snapfile=toysnap_filename)


@pytest.fixture(scope="function", params=hfs)
def hf_multi_forwards_and_backwards(request, tmp_path_factory):
    tp = tmp_path_factory.mktemp(_toysnap_filename.parent)
    toysnap_filename = tp / _toysnap_filename.name
    _create_toysnap(snapfile=toysnap_filename, withfof=request.param == "soap")
    if request.param in {"caesar_halo", "caesar_galaxy"}:
        toycaesar_filename = tp / _toycaesar_filename.name
        _create_toycaesar(filename=toycaesar_filename)

        yield Caesar(
            caesar_file=toycaesar_filename,
            group_type=request.param.split("_")[-1],
            group_index=[0, 1],
        ), Caesar(
            caesar_file=toycaesar_filename,
            group_type=request.param.split("_")[-1],
            group_index=[1, 0],
        )

        _remove_toycaesar(filename=toycaesar_filename)
    elif request.param == "soap":
        toysoap_filename = tp / _toysoap_filename.name
        membership_filebase = tp / _toysoap_membership_filebase.name
        toysoap_virtual_snapshot_filename = tp / _toysoap_virtual_snapshot_filename.name
        _create_toysoap(
            filename=toysoap_filename,
            membership_filebase=membership_filebase,
            create_virtual_snapshot=Path(toysnap_filename).is_file(),
            create_virtual_snapshot_from=toysnap_filename,
            virtual_snapshot_filename=toysoap_virtual_snapshot_filename,
        )

        yield SOAP(
            soap_file=toysoap_filename,
            soap_index=[0, 1],
        ), SOAP(
            soap_file=toysoap_filename,
            soap_index=[1, 0],
        )

        _remove_toysoap(
            filename=toysoap_filename,
            membership_filebase=membership_filebase,
            virtual_snapshot_filename=toysoap_virtual_snapshot_filename,
        )
    elif request.param == "vr":
        toyvr_filebase = tp / _toyvr_filebase.name
        _create_toyvr(filebase=toyvr_filebase)

        yield Velociraptor(
            velociraptor_filebase=toyvr_filebase, halo_index=[0, 1]
        ), Velociraptor(velociraptor_filebase=toyvr_filebase, halo_index=[1, 0])

        _remove_toyvr(filebase=toyvr_filebase)
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
                scale_factor=1.0,
                scale_exponent=1,
            ),
            velocity_centre=cosmo_array(
                [
                    [_vcentre_1, _vcentre_1, _vcentre_1],
                    [_vcentre_2, _vcentre_2, _vcentre_2],
                ],
                u.km / u.s,
                comoving=True,
                scale_factor=1.0,
                scale_exponent=0,
            ),
            spatial_offsets=cosmo_array(
                [[-1, 1], [-1, 1], [-1, 1]],
                u.Mpc,
                comoving=True,
                scale_factor=1.0,
                scale_exponent=1,
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
                scale_factor=1.0,
                scale_exponent=1,
            ),
            velocity_centre=cosmo_array(
                [
                    [_vcentre_2, _vcentre_2, _vcentre_2],
                    [_vcentre_1, _vcentre_1, _vcentre_1],
                ],
                u.km / u.s,
                comoving=True,
                scale_factor=1.0,
                scale_exponent=0,
            ),
            spatial_offsets=cosmo_array(
                [[-1, 1], [-1, 1], [-1, 1]],
                u.Mpc,
                comoving=True,
                scale_factor=1.0,
                scale_exponent=1,
            ),
        )
    _remove_toysnap(snapfile=toysnap_filename)


@pytest.fixture(scope="function", params=hfs)
def hf_multi_onetarget(request, tmp_path_factory):
    tp = tmp_path_factory.mktemp(_toysnap_filename.parent)
    toysnap_filename = tp / _toysnap_filename.name
    _create_toysnap(snapfile=toysnap_filename, withfof=request.param == "soap")
    if request.param in {"caesar_halo", "caesar_galaxy"}:
        toycaesar_filename = tp / _toycaesar_filename.name
        _create_toycaesar(filename=toycaesar_filename)

        yield Caesar(
            caesar_file=toycaesar_filename,
            group_type=request.param.split("_")[-1],
            group_index=[1],
        )

        _remove_toycaesar(filename=toycaesar_filename)
    elif request.param == "soap":
        toysoap_filename = tp / _toysoap_filename.name
        membership_filebase = tp / _toysoap_membership_filebase.name
        toysoap_virtual_snapshot_filename = tp / _toysoap_virtual_snapshot_filename.name
        _create_toysoap(
            filename=toysoap_filename,
            membership_filebase=membership_filebase,
            create_virtual_snapshot=Path(toysnap_filename).is_file(),
            create_virtual_snapshot_from=toysnap_filename,
            virtual_snapshot_filename=toysoap_virtual_snapshot_filename,
        )

        yield SOAP(
            soap_file=toysoap_filename,
            soap_index=[1],
        )

        _remove_toysoap(
            filename=toysoap_filename,
            membership_filebase=membership_filebase,
            virtual_snapshot_filename=toysoap_virtual_snapshot_filename,
        )
    elif request.param == "vr":
        toyvr_filebase = tp / _toyvr_filebase.name
        _create_toyvr(filebase=toyvr_filebase)

        yield Velociraptor(velociraptor_filebase=toyvr_filebase, halo_index=[1])

        _remove_toyvr(filebase=toyvr_filebase)
    elif request.param == "sa":
        yield Standalone(
            extra_mask=None,
            centre=cosmo_array(
                [
                    [_centre_2, _centre_2, _centre_2],
                ],
                u.Mpc,
                comoving=True,
                scale_factor=1.0,
                scale_exponent=1,
            ),
            velocity_centre=cosmo_array(
                [
                    [_vcentre_2, _vcentre_2, _vcentre_2],
                ],
                u.km / u.s,
                comoving=True,
                scale_factor=1.0,
                scale_exponent=0,
            ),
            spatial_offsets=cosmo_array(
                [[-1, 1], [-1, 1], [-1, 1]],
                u.Mpc,
                comoving=True,
                scale_factor=1.0,
                scale_exponent=1,
            ),
        )
    _remove_toysnap(snapfile=toysnap_filename)


@pytest.fixture(scope="function", params=hfs)
def hf_multi_zerotarget(request, tmp_path_factory):
    tp = tmp_path_factory.mktemp(_toysnap_filename.parent)
    toysnap_filename = tp / _toysnap_filename.name
    _create_toysnap(snapfile=toysnap_filename, withfof=request.param == "soap")
    if request.param in {"caesar_halo", "caesar_galaxy"}:
        toycaesar_filename = tp / _toycaesar_filename.name
        _create_toycaesar(filename=toycaesar_filename)

        yield Caesar(
            caesar_file=toycaesar_filename,
            group_type=request.param.split("_")[-1],
            group_index=[],
        )

        _remove_toycaesar(filename=toycaesar_filename)
    elif request.param == "soap":
        toysoap_filename = tp / _toysoap_filename.name
        membership_filebase = tp / _toysoap_membership_filebase.name
        toysoap_virtual_snapshot_filename = tp / _toysoap_virtual_snapshot_filename.name
        _create_toysoap(
            filename=toysoap_filename,
            membership_filebase=membership_filebase,
            create_virtual_snapshot=Path(toysnap_filename).is_file(),
            create_virtual_snapshot_from=toysnap_filename,
            virtual_snapshot_filename=toysoap_virtual_snapshot_filename,
        )

        yield SOAP(
            soap_file=toysoap_filename,
            soap_index=[],
        )

        _remove_toysoap(
            filename=toysoap_filename,
            membership_filebase=membership_filebase,
            virtual_snapshot_filename=toysoap_virtual_snapshot_filename,
        )
    elif request.param == "vr":
        toyvr_filebase = tp / _toyvr_filebase.name
        _create_toyvr(filebase=toyvr_filebase)

        yield Velociraptor(velociraptor_filebase=toyvr_filebase, halo_index=[])

        _remove_toyvr(filebase=toyvr_filebase)
    elif request.param == "sa":
        yield Standalone(
            extra_mask=None,
            centre=cosmo_array(
                [],
                u.Mpc,
                comoving=True,
                scale_factor=1.0,
                scale_exponent=1,
            ),
            velocity_centre=cosmo_array(
                [],
                u.km / u.s,
                comoving=True,
                scale_factor=1.0,
                scale_exponent=0,
            ),
            spatial_offsets=cosmo_array(
                [[-1, 1], [-1, 1], [-1, 1]],
                u.Mpc,
                comoving=True,
                scale_factor=1.0,
                scale_exponent=1,
            ),
        )
    _remove_toysnap(snapfile=toysnap_filename)


@pytest.fixture(scope="function")
def lm():

    def mf(mask_loaded_data=None):
        """
        A simple mask function.

        Parameters
        ----------
        mask_loaded_data : bool
            For toggling masking data as it is loaded. Ignored in this test case.

        Returns
        -------
        out : ndarray
            A simple mask array.
        """
        return np.ones(10, dtype=bool)

    lm = LazyMask(mask_function=mf)
    yield lm
