"""
Tests of string representations of objects.
"""


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
