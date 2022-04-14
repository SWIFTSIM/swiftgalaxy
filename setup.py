from setuptools import setup
from swiftgalaxy import __version__

setup(
    name='swiftgalaxy',
    version=__version__,
    description='Code abstraction of objects (galaxies) in simulations.',
    url='https://github.com/kyleaoman/swiftgalaxy',
    author='Kyle Oman',
    author_email='kyle.a.oman@durham.ac.uk',
    license='GNU GPL v3',
    packages=['swiftgalaxy'],
    install_requires=['numpy', 'scipy', 'h5py', 'unyt', 'swiftsimio'],
    extras_require=dict(
        velociraptor='velociraptor'
    ),
    include_package_data=True,
    zip_safe=False
)
