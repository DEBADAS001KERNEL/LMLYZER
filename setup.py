from setuptools import setup, find_packages

setup(
    name='lmlyzer',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'transformers',
        'pyfiglet',
        'psutil',
        'torch',
    ],
    entry_points={
        'console_scripts': [
            'lmlyzer = lmlyzer.shell:main', 
        ],
    },
)
