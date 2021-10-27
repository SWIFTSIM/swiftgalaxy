from setuptools import setup

setup(
    name='swiftgalaxy',
    version='0.0',
    description='Code abstraction of objects (galaxies) in simulations.',
    url='',
    author='Kyle Oman',
    author_email='kyle.a.oman@durham.ac.uk',
    license='GNU GPL v3',
    packages=['swiftgalaxy'],
    install_requires=['numpy', 'astropy', 'h5py', 'unyt', 'swiftsimio',
                      'velociraptor'],
    include_package_data=True,
    zip_safe=False
)
