# ARTSyn
## A collection of models and algorithms for Tabular Data Synthesis (Formerly DeepCoreML)

ARTSyn (Artificial Table Synthesizers) is a library containing models and algorithm implementations for synthesizing
artificial tabular data. Such synthetic data are frequently useful in numerous classification and regression tasks
under the presence of imbalanced datasets. Examples include fault/defect detection, intrusion detection, medical
diagnoses, financial predictions, etc.

Most models in ARTSyn support conditional data generation, namely, generation of data instances that belong to a
particular class. The models accept tabular data in CSV format and additional information about the column structure
(e.g. columns with numeric/discrete values, class columns, etc.). Then, they are trained to generate additional
samples either from a specific class, or without any condition. For the moment, ARTSyn emphasizes on Generative
Adversarial Networks (GANs), but more models and algorithms will be supported in the future.

The library is licensed under the Apache License, 2.0 (Apache-2.0)

Install with `pip install artsyn`

Relevant Publications:

* L. Akritidis, P. Bozanis, "[A Conditional GAN for Tabular Data Generation with Probabilistic Sampling of Latent Subspaces](https://arxiv.org/abs/2508.00472)", arXiv preprint arXiv:2508.00472, 2025
* L. Akritidis, P. Bozanis, "[A Clustering-Based Resampling Technique with Cluster Structure Analysis for Software Defect Detection in Imbalanced Datasets](https://www.sciencedirect.com/science/article/abs/pii/S0020025524006376)", Information Sciences, vol. 674, pp. 120724, 2024
* L. Akritidis, A. Fevgas, M. Alamaniotis, P. Bozanis, "[Conditional Data Synthesis with Deep Generative Models for Imbalanced Dataset Oversampling](https://ieeexplore.ieee.org/document/10356482)", In Proceedings of the 35th IEEE International Conference on Tools with Artificial Intelligence (ICTAI), pp. 444-451, 2023
* L. Akritidis, P. Bozanis, "[A Multi-Dimensional Survey on Learning from Imbalanced Data](https://link.springer.com/chapter/10.1007/978-3-031-67426-6_2)", Lecture Notes in Networks and Systems, 1093 LNNS, pp. 13–45, 2024