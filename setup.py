from setuptools import setup, find_packages

setup(
    name='my_nlp_project',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'gensim',
        'pandas',
        'scikit-learn',
    ],
)

