from setuptools import setup, find_namespace_packages

setup(
    name='population_structure',
    version='0.0.9',
    author='Eyal Haluts',
    author_email='eyal.haluts@mail.huji.ac.il',
    description='Fixed a syntax error in matrix_generator.py imports which made it crash. Also added a function'
                'in helper_funcs.py to calculate the average distance between two Fst matrices, and changed'
                'the name of the function matrix_distance to migration_matrix_distance (as it actually'
                'calculated distance between migration two matrices).',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    python_requires='>=3.7',
    install_requires=['scipy', "importlib_resources", "numpy"],
    packages=find_namespace_packages(where='src'),
    package_dir={'': 'src'},
    package_data={"population_structure": ['*.dll', '*.so'],
                  "population_structure.data": ['*.dll', '*.so']},
    include_package_data=True
)
