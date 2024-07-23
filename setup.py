from setuptools import setup, find_packages

setup(
    name='nmfqsofit',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scikit-learn',
        'astropy',
        'matplotlib',
        'psutil'
    ],
    entry_points={
        'console_scripts': [
            'run_nmfqsofit=nmfqsofit.nmfqsofit:main',
        ],
    },
    author='Abhijeet Anand',
    author_email='AbhijeetAnand@lbl.gov',
    description='NMF fitting for QSOs using astropy and scikit-learn',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/nmfqsofit',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
