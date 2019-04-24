from setuptools import setup, find_packages

# Setup parameters for Google Cloud ML Engine
setup(name='trainer',
      version='0.1',
      packages=find_packages(),
      description='scancube ml training',
      author='Marwin Lumbanzila',
      license='Scancube',
      install_requires=['keras', 'h5py', 'pillow', 'scikit-learn'],
      zip_safe=False
      )