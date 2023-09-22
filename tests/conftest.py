import pytest
from swiftgalaxy import SWIFTGalaxy, Velociraptor, Caesar
from toysnap import (
    create_toysnap,
    remove_toysnap,
    create_toyvr,
    remove_toyvr,
    create_toycaesar,
    remove_toycaesar,
    ToyHF,
    toysnap_filename,
    toyvr_filebase,
    toycaesar_filename,
)

hfs = ("vr", "caesar_halo", "caesar_galaxy")


@pytest.fixture(scope="function")
def sg():
    create_toysnap()

    yield SWIFTGalaxy(
        toysnap_filename,
        ToyHF(),
        transforms_like_coordinates={"coordinates", "extra_coordinates"},
        transforms_like_velocities={"velocities", "extra_velocities"},
    )

    remove_toysnap()


@pytest.fixture(scope="function")
def sg_custom_names():
    toysnap_custom_names_filename = "toysnap_custom_names.hdf5"
    alt_coord_name, alt_vel_name, alt_id_name = "my_coords", "my_vels", "my_ids"

    create_toysnap(
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

    remove_toysnap(snapfile=toysnap_custom_names_filename)


@pytest.fixture(scope="function")
def sg_autorecentre_off():
    create_toysnap()

    yield SWIFTGalaxy(
        toysnap_filename,
        ToyHF(snapfile=toysnap_filename),
        transforms_like_coordinates={"coordinates", "extra_coordinates"},
        transforms_like_velocities={"velocities", "extra_velocities"},
        auto_recentre=False,
    )

    remove_toysnap(snapfile=toysnap_filename)


@pytest.fixture(scope="function")
def sg_vr():
    create_toysnap()
    create_toyvr()

    yield SWIFTGalaxy(
        toysnap_filename,
        Velociraptor(velociraptor_filebase=toyvr_filebase, halo_index=0),
        transforms_like_coordinates={"coordinates", "extra_coordinates"},
        transforms_like_velocities={"velocities", "extra_velocities"},
    )

    remove_toysnap()
    remove_toyvr()


@pytest.fixture(scope="function")
def vr():
    create_toyvr()

    yield Velociraptor(velociraptor_filebase=toyvr_filebase, halo_index=0)

    remove_toyvr()


# SHOULD BE ABLE TO __MERGE THE galaxy AND halo CASES WITH PARAMETRIZE?
@pytest.fixture(scope="function")
def caesar_halo():
    create_toycaesar()

    yield Caesar(caesar_file=toycaesar_filename, group_type="halo", group_index=0)

    remove_toycaesar()


@pytest.fixture(scope="function")
def caesar_galaxy():
    create_toycaesar()

    yield Caesar(caesar_file=toycaesar_filename, group_type="galaxy", group_index=0)

    remove_toycaesar()


@pytest.fixture(scope="function")
def sg_caesar_halo():
    create_toysnap()
    create_toycaesar()

    yield SWIFTGalaxy(
        toysnap_filename,
        Caesar(caesar_file=toycaesar_filename, group_type="halo", group_index=0),
        transforms_like_coordinates={"coordinates", "extra_coordinates"},
        transforms_like_velocities={"velocities", "extra_velocities"},
    )

    remove_toysnap()
    remove_toycaesar()


@pytest.fixture(scope="function")
def sg_caesar_galaxy():
    create_toysnap()
    create_toycaesar()

    yield SWIFTGalaxy(
        toysnap_filename,
        Caesar(caesar_file=toycaesar_filename, group_type="galaxy", group_index=0),
        transforms_like_coordinates={"coordinates", "extra_coordinates"},
        transforms_like_velocities={"velocities", "extra_velocities"},
    )

    remove_toysnap()
    remove_toycaesar()
