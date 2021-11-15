from setuptools import setup, find_packages

requirements = [
    "torch",
    "numpy"
]

setup(
    name='deepquantum',
    version='2.0.0',
    packages=find_packages(where="."),
    url='',
    license='',
    author='TuringQ',
    author_email='qianlong@turingq.com',
    install_requires=requirements,
    description='deepquantum for qml'
)
