"""
Tests checking that we can create objects.

If these fail something fundamental has gone wrong.
"""

from pathlib import Path
from swiftgalaxy.demo_data import _create_toyvr, _create_toycaesar, _create_toysoap
from swiftgalaxy.halo_catalogues import Caesar, Velociraptor, SOAP, Standalone
from swiftgalaxy.reader import SWIFTGalaxy
from swiftgalaxy.iterator import SWIFTGalaxies


class TestSWIFTGalaxyCreation:
    """Test that our fixtures can create the objects that we want."""

    def test_sg_creation(self, sg):
        """Make sure we can create a SWIFTGalaxy without error."""
        assert isinstance(sg, SWIFTGalaxy)

    def test_soap_creation(self, soap):
        """Make sure we can create a SOAP without error."""
        assert isinstance(soap, SOAP)

    def test_soap_recreation(self, toysoap_with_virtual_snapshot):
        """
        Make sure we can try to create a soap file set when one already exists.

        This uses the fixture. We should skip making files.
        """
        assert toysoap_with_virtual_snapshot["toysoap_filename"].is_file()
        assert Path(
            str(toysoap_with_virtual_snapshot["membership_filebase"]) + ".0.hdf5"
        ).is_file()
        assert toysoap_with_virtual_snapshot["toysnap_filename"].is_file()
        assert toysoap_with_virtual_snapshot[
            "toysoap_virtual_snapshot_filename"
        ].is_file()
        _create_toysoap(
            filename=toysoap_with_virtual_snapshot["toysoap_filename"],
            membership_filebase=toysoap_with_virtual_snapshot["membership_filebase"],
            create_membership=True,
            create_virtual_snapshot=True,
            create_virtual_snapshot_from=toysoap_with_virtual_snapshot[
                "toysnap_filename"
            ],
            virtual_snapshot_filename=toysoap_with_virtual_snapshot[
                "toysoap_virtual_snapshot_filename"
            ],
        )

    def test_vr_creation(self, vr):
        """Make sure we can create a Velociraptor without error."""
        assert isinstance(vr, Velociraptor)

    def test_vr_recreation(self, vr):
        """
        Make sure we can try to create a velociraptor file set when one already exists.

        This uses the fixture. We should skip making files.
        """
        for f in vr.velociraptor_files.values():
            assert Path(f).is_file()
        filebase = vr.velociraptor_files["properties"].split(".")[0]
        _create_toyvr(filebase=filebase)

    def test_caesar_creation(self, caesar):
        """Make sure we can create a Caesar without error."""
        assert isinstance(caesar, Caesar)

    def test_caesar_recreation(self, caesar):
        """
        Make sure we can try to create a caesar file when one already exists.

        This uses the fixture. We should skip making the file.
        """
        assert caesar.caesar_file.is_file()
        _create_toycaesar(filename=caesar.caesar_file)

    def test_sa_creation(self, sa):
        """Make sure we can create a Standalone without error."""
        assert isinstance(sa, Standalone)

    def test_sg_soap_creation(self, sg_soap):
        """Make sure we can create a SWIFTGalaxy with SOAP without error."""
        assert isinstance(sg_soap, SWIFTGalaxy)
        assert isinstance(sg_soap.halo_finder, SOAP)

    def test_sg_vr_creation(self, sg_vr):
        """Make sure we can create a SWIFTGalaxy with velociraptor without error."""
        assert isinstance(sg_vr, SWIFTGalaxy)
        assert isinstance(sg_vr.halo_finder, Velociraptor)

    def test_sg_caesar_creation(self, sg_caesar):
        """Make sure we can create a SWIFTGalaxy with Caesar without error."""
        assert isinstance(sg_caesar, SWIFTGalaxy)
        assert isinstance(sg_caesar.halo_finder, Caesar)

    def test_sg_sa_creation(self, sg_sa):
        """Make sure we can create a SWIFTGalaxy with Standalone without error."""
        assert isinstance(sg_sa, SWIFTGalaxy)
        assert isinstance(sg_sa.halo_finder, Standalone)

    def test_tab_completion(self, sg):
        """
        Check that particle dataset names and named column names are tab-completeable.

        They should be in the namespace for tab completion via the dir() method.
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

    def test_internal_refs(self, sg):
        """Check that datasets and namedcolumns store a reference to their swiftgalaxy."""
        assert sg.gas._swiftgalaxy is sg
        assert sg.gas.hydrogen_ionization_fractions._swiftgalaxy is sg


class TestSWIFTGalaxiesCreation:
    """Test that our fixtures can create the objects that we want."""

    def test_sgs_creation(self, sgs):
        """Test that the fixture creates a SWIFTGalaxies."""
        assert isinstance(sgs, SWIFTGalaxies)

    def test_soap_multi_creation(self, soap_multi):
        """Test that the fixture creates a SOAP."""
        assert isinstance(soap_multi, SOAP)

    def test_vr_creation(self, vr_multi):
        """Test that the fixture creates a Velociraptor."""
        assert isinstance(vr_multi, Velociraptor)

    def test_caesar_creation(self, caesar_multi):
        """Test that the fixture creates a Caesar."""
        assert isinstance(caesar_multi, Caesar)

    def test_sa_creation(self, sa_multi):
        """Test that the fixture creates a Standalone."""
        assert isinstance(sa_multi, Standalone)

    def test_sgs_soap_creation(self, sgs_soap):
        """Test that the fixture creates a SWIFTGalaxies with SOAP."""
        assert isinstance(sgs_soap, SWIFTGalaxies)
        assert isinstance(sgs_soap.halo_catalogue, SOAP)

    def test_sgs_vr_creation(self, sgs_vr):
        """Test that the fixture creates a SWIFTGalaxies with Velociraptor."""
        assert isinstance(sgs_vr, SWIFTGalaxies)
        assert isinstance(sgs_vr.halo_catalogue, Velociraptor)

    def test_sgs_caesar_creation(self, sgs_caesar):
        """Test that the fixture creates a SWIFTGalaxies with Caesar."""
        assert isinstance(sgs_caesar, SWIFTGalaxies)
        assert isinstance(sgs_caesar.halo_catalogue, Caesar)

    def test_sgs_sa_creation(self, sgs_sa):
        """Test that the fixture creates a SWIFTGalaxies with Standalone."""
        assert isinstance(sgs_sa, SWIFTGalaxies)
        assert isinstance(sgs_sa.halo_catalogue, Standalone)


class TestDeletion:
    """Test that datasets are deleteable."""

    def test_dataset_deleter(self, sg):
        """Check that we can delete a dataset's array."""
        sg.gas.coordinates  # lazy-load some data
        assert sg.gas._internal_dataset._coordinates is not None
        del sg.gas.coordinates
        assert sg.gas._internal_dataset._coordinates is None
