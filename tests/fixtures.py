import pytest
from swiftgalaxy import SWIFTGalaxy
from toysnap import create_toysnap, remove_toysnap, ToyHF, toysnap_filename


@pytest.fixture(scope='module')
def sg():

    create_toysnap()

    yield SWIFTGalaxy(toysnap_filename, ToyHF())

    remove_toysnap()
