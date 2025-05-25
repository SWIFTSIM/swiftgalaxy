import pytest
from swiftgalaxy import SWIFTGalaxy, Standalone
from toysnap import (
    create_toysnap,
    remove_toysnap,
    toysnap_filename,
    n_g_all,
    n_dm_all,
    n_s_all,
    n_bh_all,
)
from swiftsimio.objects import cosmo_array, cosmo_factor, a
import unyt as u


class TestSWIFTGalaxyCreation:
    def test_sg_creation(self, sg):
        """
        Make sure we can create a SWIFTGalaxy without error.
        """
        pass  # fixture created SWIFTGalaxy

    def test_soap_creation(self, soap):
        """
        Make sure we can create a SOAP without error.
        """
        pass  # fixture created SOAP interface

    def test_vr_creation(self, vr):
        """
        Make sure we can create a Velociraptor without error.
        """
        pass  # fixture created Velociraptor interface

    def test_caesar_creation(self, caesar):
        """
        Make sure we can create a Caesar without error.
        """
        pass  # fixture created Caesar interface

    def test_sa_creation(self, sa):
        """
        Make sure we can create a Standalone without error.
        """
        pass  # fixture created Standalone interface

    def test_sg_soap_creation(self, sg_soap):
        """
        Make sure we can create a SWIFTGalaxy with SOAP without error.
        """
        pass  # fixture created SWIFTGalaxy with SOAP interface

    def test_sg_vr_creation(self, sg_vr):
        """
        Make sure we can create a SWIFTGalaxy with velociraptor without error.
        """
        pass  # fixture created SWIFTGalaxy with Velociraptor interface

    def test_sg_caesar_creation(self, sg_caesar):
        """
        Make sure we can create a SWIFTGalaxy with Caesar without error.
        """
        pass  # fixture created SWIFTGalaxy with Caesar interface

    def test_sg_sa_creation(self, sg_sa):
        """
        Make sure we can create a SWIFTGalaxy with Standalone without error.
        """
        pass  # fixture created SWIFTGalaxy with Standalone interface

    def test_tab_completion(self, sg):
        """
        Check that particle dataset names and named column names are in
        the namespace for tab completion via the dir() method.
        """
        for prop in ("coordinates", "masses", "hydrogen_ionization_fractions"):
            # check some data fields & named column sets
            assert prop in dir(sg.gas)
        for prop in ("neutral", "ionized"):
            # check named column data fields
            assert prop in dir(sg.gas.hydrogen_ionization_fractions)
        # and check something that we inherited:
        assert "generate_empty_properties" in dir(sg.gas)
        # finally, check that we didn't lazy-load everything!
        assert sg.gas._particle_dataset._coordinates is None

    def test_no_masks(self):
        """
        Check that if we have no masks we read everything in the box (and warn about it).
        """
        try:
            create_toysnap()
            with pytest.warns(UserWarning, match="No spatial_offsets provided."):
                sa = Standalone(
                    extra_mask=None,
                    centre=cosmo_array(
                        [0, 0, 0],
                        u.Mpc,
                        comoving=True,
                        cosmo_factor=cosmo_factor(a**1, 1.0),
                    ),
                    velocity_centre=cosmo_array(
                        [0, 0, 0],
                        u.km / u.s,
                        comoving=True,
                        cosmo_factor=cosmo_factor(a**0, 1.0),
                    ),
                    spatial_offsets=None,
                )
            sg = SWIFTGalaxy(
                toysnap_filename,
                sa,
                transforms_like_coordinates={"coordinates", "extra_coordinates"},
                transforms_like_velocities={"velocities", "extra_velocities"},
            )
            # check that extra mask is blank for all particle types:
            assert sg._extra_mask.gas is None
            assert sg._extra_mask.dark_matter is None
            assert sg._extra_mask.stars is None
            assert sg._extra_mask.black_holes is None
            # check that cell mask is blank for all particle types:
            for cell_mask in sg._spatial_mask.cell_mask.values():
                assert cell_mask.all()
            # check that we read all the particles:
            assert sg.gas.masses.size == n_g_all
            assert sg.dark_matter.masses.size == n_dm_all
            assert sg.stars.masses.size == n_s_all
            assert sg.black_holes.masses.size == n_bh_all
        finally:
            remove_toysnap()


class TestSWIFTGalaxiesCreation:
    def test_sgs_creation(self, sgs):
        pass  # fixture created SWIFTGalaxies

    def test_soap_multi_creation(self, soap_multi):
        pass  # fixture created SOAP interface

    def test_vr_creation(self, vr_multi):
        pass  # fixture created Velociraptor interface

    def test_caesar_creation(self, caesar_multi):
        pass  # fixture created Caesar interface

    def test_sa_creation(self, sa_multi):
        pass  # fixture created Standalone interface

    def test_sgs_soap_creation(self, sgs_soap):
        pass  # fixture created SWIFTGalaxy with SOAP interface

    def test_sgs_vr_creation(self, sgs_vr):
        pass  # fixture created SWIFTGalaxy with Velociraptor interface

    def test_sgs_caesar_creation(self, sgs_caesar):
        pass  # fixture created SWIFTGalaxy with Caesar interface

    def test_sgs_sa_creation(self, sgs_sa):
        pass  # fixture created SWIFTGalaxy with Standalone interface


class TestDeletion:

    def test_dataset_deleter(self, sg):
        """
        Check that we can delete a dataset's array.
        """
        sg.gas.coordinates  # lazy-load some data
        assert sg.gas._internal_dataset._coordinates is not None
        del sg.gas.coordinates
        assert sg.gas._internal_dataset._coordinates is None
