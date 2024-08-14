from setuptools import setup, find_namespace_packages

setup(
    name='population_structure',
    version='1.0.1',
    author='Eyal Haluts',
    author_email='eyal.haluts@mail.huji.ac.il',
    description='Major changes in the the utils module. The f_to_m function previously used a numeric solver which '
                'often' \
                'failed due to constraining the solution with conservative migration constraints. Now, using the '
                'constraints' \
                'the numerical solver adds the constraints as part of the equations to minimize, and it possible to run'
                'the function without the conservative migration constraints.',
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
