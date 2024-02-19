from setuptools import setup, find_namespace_packages

setup(
    name='population_structure',
    version='0.0.4',
    author='Eyal Haluts',
    author_email='eyal.haluts@mail.huji.ac.il',
    description='This version contains some changes so the C library used in the Migration class  '
                'is loaded correctly for Windows/Linux and that the code works in case the '
                'loading of the library fails (uses the less efficient pure python function).',
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
