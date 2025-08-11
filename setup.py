from distutils.core import setup
from setuptools import find_packages

DESCRIPTION = 'Artificial Tabular Data Synthesizers'
LONG_DESCRIPTION = ('<p>ARTSyn is a library containing models and algorithm implementations for synthesizing artificial '
                    'tabular data. Such synthetic data are frequently useful in numerous classification and regression '
                    'tasks under the presence of imbalanced datasets. Examples include fault/defect detection, '
                    'intrusion detection, medical diagnoses, financial predictions, etc.</p>'
                    '<p>Most models in ARTSyn support conditional data generation, namely, generation of data instances '
                    'that belong to a particular class. The models accept tabular data in CSV format and additional '
                    'information about the column structure (e.g. columns with numeric/discrete values, class columns, '
                    'etc.). Then, they are trained to generate additional samples either from a specific class, or '
                    'without any condition. For the moment, ARTSyn emphasizes on Generative Adversarial Networks (GANs), '
                    'but more models and algorithms will be supported in the future.</p>'
                    '<p><b>Licence:</b> Apache License, 2.0 (Apache-2.0)</p>'
                    '<p><b>Dependencies:</b>NumPy, Pandas, Matplotlib, Seaborn, joblib, Synthetic Data Vault (SDV), '
                    'pyTorch, scikit-learn, xgboost, imblearn, Reversible Data Transforms (RDT), tqdm.</p>'
                    '<p><b>GitHub repository:</b> '
                    '<a href="https://github.com/lakritidis/ARTSyn">https://github.com/lakritidis/artsyn</a></p>'
                    '<p><b>Publications:</b><ul>'
                    '<li>L. Akritidis, P. Bozanis, "A Conditional GAN for Tabular Data Generation with Probabilistic'
                    'Sampling of Latent Subspaces", arXiv preprint arXiv:2508.00472, 2025</li>'
                    '<li>L. Akritidis, P. Bozanis, "A Clustering-Based Resampling Technique with Cluster Structure '
                    'Analysis for Software Defect Detection in Imbalanced Datasets", Information Sciences, vol. 674,'
                    'pp. 120724, 2024</li>'
                    '<li>L. Akritidis, A. Fevgas, M. Alamaniotis, P. Bozanis, "Conditional Data Synthesis with Deep '
                    'Generative Models for Imbalanced Dataset Oversampling", In Proceedings of the 35th IEEE '
                    'International Conference on Tools with Artificial Intelligence, pp. 444-451, 2023</li>'
                    '<li>L. Akritidis, P. Bozanis, "A Multi-Dimensional Survey on Learning from Imbalanced Data", '
                    'Lecture Notes in Networks and Systems, 1093 LNNS, pp. 13–45, 2024</li>'
                    '</ul></p>')
setup(
    name='ARTSyn',
    version='0.5.2',
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    author="Leonidas Akritidis",
    author_email="lakritidis@ihu.gr",
    maintainer="Leonidas Akritidis",
    maintainer_email="lakritidis@ihu.gr",
    packages=find_packages(),
    package_data={'': ['generators/*']},
    url='https://github.com/lakritidis/artsyn',
    install_requires=["numpy",
                      "pandas",
                      "matplotlib",
                      "seaborn",
                      "joblib",
                      "sdv",
                      "torch>=2.0.0",
                      "scikit-learn>=1.4.0",
                      "xgboost",
                      "imblearn>=0.0",
                      "rdt>=1.3.0,<2.0",
                      "tqdm",
                      "timm"],
    license="Apache",
    keywords=[
        "tabular data", "tabular data synthesis", "data engineering", "imbalanced data", "GAN", "VAE", "oversampling",
        "machine learning", "deep learning"]
)
