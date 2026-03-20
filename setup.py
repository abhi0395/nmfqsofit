
from setuptools import setup, find_packages

setup(
    name='nmfqsofit',
    use_scm_version=True,
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
    python_requires='>=3.9',
)
