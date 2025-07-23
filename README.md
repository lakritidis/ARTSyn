# TADS
## Tabular Data Synthesizers (Formerly DeepCoreML)

TADS is a library containing models and algorithm implementations for synthesizing artificial tabular data. Such synthetic data are frequently useful in numerous classification and regression tasks under the presence of imbalanced datasets. Examples include fault/defect detection, intrusion detection, medical diagnoses, financial predictions, etc.

Most models in TADS support conditional data generation, namely, generation of data instances that belong to a particular class. The models accept tabular data in CSV format and additional information about the column structure (e.g. columns with numeric/discrete values, class columns, etc.). Then, they are trained to generate additional samples either from a specific class, or without any condition. For the moment, TADS emphasizes on Generative Adversarial Networks (GANs), but more models and algorithms will be supported in the future.

The library is licensed under the Apache License, 2.0 (Apache-2.0)

Install with `pip install tads`

Relevant Publications:

* L. Akritidis, P. Bozanis, "A Clustering-Based Resampling Technique with Cluster Structure Analysis for Software Defect Detection in Imbalanced Datasets", Information Sciences, vol. 674, pp. 120724, 2024.
* L. Akritidis, A. Fevgas, M. Alamaniotis, P. Bozanis, "Conditional Data Synthesis with Deep Generative Models for Imbalanced Dataset Oversampling", In Proceedings of the 35th IEEE International Conference on Tools with Artificial Intelligence (ICTAI), pp. 444-451, 2023.
* L. Akritidis, P. Bozanis, "A Multi-Dimensional Survey on Learning from Imbalanced Data", Chapter in Machine Learning Paradigms - Advances in Theory and Applications of Learning from Imbalanced Data, to appear, 2023.
