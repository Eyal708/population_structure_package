from setuptools import setup, find_namespace_packages

setup(
    name='population_structure',
    version='1.0.0',
    author='Eyal Haluts',
    author_email='eyal.haluts@mail.huji.ac.il',
    description='Added the function conservative_migration_from_binary_matrix to the helper_funcs module.'
                'This function generates a conservative migration matrix from a binary migration matrix'
                '(See documentation for more details).',
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
