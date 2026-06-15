from distutils.core import setup
from setuptools import find_packages

DESCRIPTION = 'A collection of Machine Learning techniques for data management, engineering and augmentation.'
LONG_DESCRIPTION = '<p>DeepCoreML is a collection of Machine Learning techniques for data management, engineering, ' \
    'and augmentation. More specifically, DeepCoreML includes modules for:</p>'\
    '<ul>' \
    '<li>Data management</li>' \
    '<li>Text data preprocessing</li>' \
    '<li>Text representation, vectorization, embeddings</li>' \
    '<li>Dimensionality reduction</li>' \
    '<li>Generative modeling</li>' \
    '<li>Imbalanced datasets</li>' \
    '</ul>' \
    '<p><b>Licence:</b> Apache License, 2.0 (Apache-2.0)</p>' \
    '<p><b>Dependencies:</b>NumPy, Pandas, Matplotlib, Seaborn, joblib, Synthetic Data Vault (SDV), pyTorch,' \
    'scikit-learn, xgboost, imblearn, Reversible Data Transforms (RDT), tqdm.</p>'\
    '<p><b>GitHub repository:</b> '\
    '<a href="https://github.com/lakritidis/DeepCoreML">https://github.com/lakritidis/DeepCoreML</a></p>' \
    '<p><b>Publications:</b><ul>' \
    '<li>L. Akritidis, P. Bozanis, "A Clustering-Based Resampling Technique with Cluster Structure Analysis for' \
    'Software Defect Detection in Imbalanced Datasets", Information Sciences, vol. 674, pp. 120724, 2024.</li>' \
    '<li>L. Akritidis, A. Fevgas, M. Alamaniotis, P. Bozanis, "Conditional Data Synthesis with Deep Generative Models '\
    'for Imbalanced Dataset Oversampling", In Proceedings of the 35th IEEE International Conference on Tools with '\
    'Artificial Intelligence (ICTAI), pp. 444-451, 2023, 2023.</li>' \
    '</ul></p>'

setup(
    name='DeepCoreML',
    version='0.4.6',
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    author="Leonidas Akritidis",
    author_email="lakritidis@ihu.gr",
    maintainer="Leonidas Akritidis",
    maintainer_email="lakritidis@ihu.gr",
    packages=find_packages(),
    package_data={'': ['generators/*']},
    url='https://github.com/lakritidis/DeepCoreML',
    install_requires=["numpy",
                      "pandas",
                      "matplotlib",
                      "seaborn",
                      "joblib",
                      "sdv",
                      "kmodes",
                      "torch>=2.0.0",
                      "scikit-learn>=1.4.0",
                      "xgboost",
                      "imblearn>=0.0",
                      "rdt>=1.3.0,<2.0",
                      "tqdm",
                      "umap-learn",
                      "hdbscan",
                      "gower"],
    license="Apache",
    keywords=[
        "tabular data", "tabular data synthesis", "data engineering", "imbalanced data", "GAN", "VAE", "oversampling",
        "machine learning", "deep learning"]
)
