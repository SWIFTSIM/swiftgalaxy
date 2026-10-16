"""
Test support for remote file access with hdfstream.

Currently only :class:`~swiftgalaxy.halo_catalogues.SOAP` and the snapshot file passed
to a :class:`~swiftgalaxy.reader.SWIFTGalaxy` or
:class:`~swiftgalaxy.iterator.SWIFTGalaxies` support remote access. The
:class:`~swiftgalaxy.halo_catalogues.Caesar` and
:class:`~swiftgalaxy.halo_catalogues.Velociraptor` use loaders that do not support remote
file access with hdfstream, and :class:`~swiftgalaxy.halo_catalogues.Standalone` does not
read any files.
"""

import pytest
from pathlib import Path
from swiftgalaxy.halo_catalogues import SOAP
from swiftgalaxy.reader import SWIFTGalaxy
from swiftgalaxy.iterator import SWIFTGalaxies
from swiftsimio.optional_packages import HDFSTREAM_AVAILABLE, hdfstream


def open_with_hdfstream(
    filename: str, request: pytest.FixtureRequest
) -> "hdfstream.RemoteFile":
    """Open a remote file using hdfstream."""
    if not HDFSTREAM_AVAILABLE:
        pytest.skip("hdfstream (optional dependency) not available.")
    server = request.config.getoption("--hdfstream-server")
    prefix = request.config.getoption("--hdfstream-prefix")
    if server is None:
        pytest.skip("hdfstream server URL not specified.")
    return hdfstream.open(server, str(Path(prefix) / Path(filename)))


def test_soap_with_hdfstream(request: pytest.FixtureRequest):
    """Load a remote file with :class:`~swiftgalaxy.halo_catalogues.SOAP`."""
    soap = SOAP(open_with_hdfstream("SOAPEagleSingle.hdf5", request), soap_index=0)
    assert soap.bound_subhalo.stellar_mass.size == 1
    return


def test_sg_with_hdfstream(request: pytest.FixtureRequest):
    """Load a remote file with :class:`~swiftgalaxy.reader.SWIFTGalaxy`."""
    sg = SWIFTGalaxy(
        open_with_hdfstream("EagleSingleVirtual.hdf5", request),
        SOAP(open_with_hdfstream("SOAPEagleSingle.hdf5", request), soap_index=0),
    )
    assert sg.gas.coordinates.size > 0
    assert sg.halo_catalogue.bound_subhalo.stellar_mass.size == 1


def test_sgs_with_hdfstream(request: pytest.FixtureRequest):
    """Load a remote file with :class:`~swiftgalaxy.iterator.SWIFTGalaxies`."""
    sgs = SWIFTGalaxies(
        open_with_hdfstream("EagleSingleVirtual.hdf5", request),
        SOAP(open_with_hdfstream("SOAPEagleSingle.hdf5", request), soap_index=[0, 1]),
    )
    for sg in sgs:
        assert sg.gas.coordinates.size > 0
        assert sg.halo_catalogue.bound_subhalo.stellar_mass.size == 1
