from setuptools import setup

setup(
    name='swiftgalaxy',
    version='0.0',
    description='Code abstraction of objects (galaxies) in simulations.',
    url='https://github.com/kyleaoman/swiftgalaxy',
    author='Kyle Oman',
    author_email='kyle.a.oman@durham.ac.uk',
    license='GNU GPL v3',
    packages=['swiftgalaxy'],
    install_requires=['numpy', 'h5py', 'unyt', 'swiftsimio'],
    extras_require=dict(
        velociraptor='velociraptor'
    ),
    include_package_data=True,
    zip_safe=False
)
