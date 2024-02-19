from setuptools import setup, find_namespace_packages

setup(
    name='population_structure',
    version='0.0.1',
    author='Eyal Haluts',
    author_email='eyal.haluts@mail.huji.ac.il',
    description='First version of the population_structure package published on PyPi.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    python_requires='>=3.7',
    install_requires=['scipy', "importlib_resources", "numpy"],
    packages=find_namespace_packages(where='src'),
    package_dir={'': 'src'},
    package_data={"population_structure": ['*.dll'],
                  "population_structure.data": ['*.dll']},
    include_package_data=True
)
