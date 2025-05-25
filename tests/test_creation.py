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
