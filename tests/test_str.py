"""
Tests of string representations of objects.
"""

import pytest
from swiftgalaxy.demo_data import web_examples, generated_examples


class TestStr:

    def test_coordinate_helper_str(self, sg):
        """
        Check that we get a sensible string representation of a coordinate helper.
        """
        string = str(sg.gas.spherical_coordinates)
        assert "Available coordinates:" in string
        assert "radius" in string
        assert "azimuth" in string
        assert repr(sg.gas.spherical_coordinates) == string

    def test_namedcolumn_fullname(self, sg):
        """
        Check that the _fullname of a namedcolumns matches the dataset + namedcolumns
        name.
        """
        assert (
            sg.gas.hydrogen_ionization_fractions._fullname
            == "gas.hydrogen_ionization_fractions"
        )

    def test_sg_string(self, sg):
        """
        Check that the swiftgalaxy has a string representation (not the one from
        SWIFTDataset).
        """
        string = str(sg)
        assert "SWIFTGalaxy at" in string
        assert repr(sg) == string

    @pytest.mark.parametrize("demodata", [web_examples, generated_examples])
    def test_exampledata_string(self, demodata):
        """
        Check that the webexample has an informative string representation.
        """
        string = str(demodata)
        for k in demodata.available_examples:
            assert k in string
