"""
Extend SWIFTSimIO to analyse individual galaxies.

SWIFTGalaxy is a module that extends SWIFTSimIO tailored to analyses of particles
belonging to individual simulated galaxies. It inherits from and extends the functionality
of the SWIFTDataset. It understands the content of halo catalogues (supported: SOAP,
Velociraptor, Caesar) and therefore which particles belong to a galaxy or other group of
particles, and its integrated properties. The particles occupy a coordinate frame that is
enforced to be consistent, such that particles loaded on-the-fly will match e.g. rotations
and translations of particles already in memory. Intuitive masking of particle datasets is
also enabled. Finally, some utilities to make working in cylindrical and spherical
coordinate systems more convenient are also provided.
"""

from .reader import SWIFTGalaxy
from .iterator import SWIFTGalaxies
from .halo_catalogues import SOAP, Velociraptor, Caesar, Standalone
from .masks import MaskCollection
from .__version__ import __version__

__all__ = [
    "SWIFTGalaxy",
    "SWIFTGalaxies",
    "SOAP",
    "Velociraptor",
    "Caesar",
    "Standalone",
    "MaskCollection",
    "__version__",
]

name = "swiftgalaxy"
