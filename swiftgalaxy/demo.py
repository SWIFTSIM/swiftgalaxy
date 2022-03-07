import socket
import unyt as u
from os import path
from _swiftgalaxy import SWIFTGalaxy

snapnum = 23
if 'cosma' in socket.gethostname():
    base_dir = '/cosma7/data/dp004/dc-chai1/HAWK/'\
        '106e3_104b2_norm_0p3_new_cooling_L006N188/'
elif ('autarch' in socket.gethostname()) \
     or ('farseer' in socket.gethostname()):
    base_dir = '/home/koman/'\
        '106e3_104b2_norm_0p3_new_cooling_L006N188/'
velociraptor_filebase = path.join(base_dir, 'halo_{:04d}'.format(snapnum))
snapshot_filename = path.join(base_dir, 'colibre_{:04d}.hdf5'.format(snapnum))

target_halo_index = 3

# First we set up a galaxy object.
# The arguments are similar to those for SWIFTDataset and the
# velociraptor catalogue tool.
# extra_mask=None results in spatial masking only (i.e. sub-cubes)
# extra_mask='bound_only' selects only bound particles (accroding to
# velociraptor)
# It will also accept a SWIFTMask defined by user.
# More masking modes could be defined...
# As you might expect, any particle array loaded will be masked
# accordingly, automatically during read.
# auto_recentre will place the origin of the coordinates at the
# potential minimum (given centre_type=minpot) of the object of
# interest, and boost the velocity frame to the velocity of the
# centre of potential as given by velociraptor. All box wrapping
# is taken care of behind the scenes.
SG = SWIFTGalaxy(
    snapshot_filename,
    velociraptor_filebase,
    target_halo_index,
    extra_mask='bound_only',
    auto_recentre=True,  # the default
    centre_type='minpot'  # the default
)

# Coordinate transformations are supported:
#   - Position translation
#   - Velocity translation
#   - Position & velocity rotation
# via the functions SWIFTGalaxy.rotate, SWIFTGalaxy.translate
# and SWIFTGalaxy.recentre (implemented via translate).
# The SWIFTGalaxy object keeps a memory of the transformation
# stack and will transform any newly loaded particle arrays
# accordingly when they are read in, so everything always exists
# in a common coordinate frame. Currently it is assumed that each
# particle type has a single coordinate array and a single velocity
# array, and that they have a common name (which is configurable, but
# defaults to 'coordinates' and 'velocities') across types.
# An example rotation, using axis-angle specification (rotation matrices
# are also supported):
SG.rotate(angle_axis=(60 * u.deg, 'z'))

# Spherical and cylindrical coordinates are also supported. These are
# lazily evaluated. After a translation or rotation they just get
# deleted, since transforming them ends up being as expensive as just
# re-evaluating them from the cartesian coordinates next time they are
# referenced. I haven't completely settled on the best way to expose
# these via the interface. There's also a cartesian coordinate interface
# for completeness; this shares memory with the ordinary coordinate array
# so is inexpensive. The names of the coordinates are trivial to alias
# given the implementation, for example cylindrical radius is R or rho,
# currently. Not sure what to do with spherical coordinates orther than
# lon/lat, since theta & phi are used interchangeably in the literature.
# These are all return cosmo_arrays, though I have a little bit of work
# left to ensure they have all their attributes preserved through the
# magic on the backend.
SG.gas.spherical_coordinates.r
SG.gas.spherical_coordinates.lon
SG.gas.spherical_coordinates.lat
SG.gas.cylindrical_coordinates.R  # or .rho
SG.gas.cylindrical_coordinates.phi  # or .lon
SG.gas.cylindrical_coordinates.z
SG.gas.cartesian_coordinates.x
SG.gas.cartesian_coordinates.y
SG.gas.cartesian_coordinates.z
SG.gas.cartesian_coordinates.xyz

# Beyond this, I think it would make sense to build in a few canned
# analysis routines. For example, a function to calculate the circular
# velocity (total, and per particle type). I guess there are many
# possibilities, would try to keep it to the most generic and
# unambiguous things.
