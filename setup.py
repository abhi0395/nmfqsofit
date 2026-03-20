
import subprocess
from setuptools import setup, find_packages

def get_version():
    try:
        version = subprocess.check_output([
            'git', 'describe', '--tags', '--abbrev=0'
        ]).decode('utf-8').strip()
        return version
    except Exception:
        return '0.0.0'

setup(
    name='nmfqsofit',
    version=get_version(),
    packages=find_packages(),
    install_requires=[
        'numpy',
        'astropy',
        'matplotlib',
        'psutil',
        'NonnegMFPy',
        'tqdm',
        'scipy',
        'pyyaml',
    ],
    extras_require={
        'dev': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'nmfqsofit=nmfqsofit.parallel_main:main',
        ],
    },
    author='Abhijeet Anand',
    author_email='abhijeetanand2011@gmail.com',
    description='Nonnegative matrix factorization based continuum fitting for Quasars using vectorized NMF module or NNLS methods',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/abhi0395/nmfqsofit',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
