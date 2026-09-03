**Multimodal Breast Cancer Detection**
An AI/ML research project investigating unimodal and multimodal approaches to breast cancer classification using clinical data, mammogram images, and textual patient information.

The project implements and compares models trained on individual modalities (unimodal) against models combining multiple sources of information (multimodal). This allows the contribution and predictive value of each modality to be evaluated alongside the performance of multimodal data fusion.

The study also investigates transfer learning, GAN-generated synthetic mammograms, and modality ablation to evaluate model robustness when clinical, imaging, or textual information is missing or replaced.

**What This Project Explores**
Unimodal models — models trained independently on clinical, image, or text data
Multimodal models — models combining multiple modalities
Model comparison — benchmarking unimodal and multimodal approaches
Transfer learning — using pre-trained models such as EfficientNet
Medical image classification — analysing mammogram images using deep learning
Clinical data classification — using traditional ML and neural networks
Medical NLP — processing clinical text using transformer-based models
GAN-generated mammograms — investigating synthetic images as replacements for real images
Modality ablation — testing performance when one or more input modalities are removed
Robustness analysis — evaluating how models perform under incomplete or missing data

**Models & Technologies**
Python · Scikit-learn · TensorFlow/Keras · PyTorch · HuggingFace Transformers · EfficientNet · GANs

The project includes models such as Logistic Regression, SVM, MLP, EfficientNet, CNNs, and transformer-based models, with the different approaches evaluated using metrics including accuracy, precision, recall, F1-score, and AUC.

**Experimental Approach**

The experiments were structured around three main areas:

Unimodal experiments
Each data type was evaluated independently to establish baseline performance.
Multimodal experiments
Clinical, image, and text features were combined to investigate whether integrating different sources of information improves classification.
Ablation & synthetic-data experiments
Individual modalities were removed or replaced with synthetic GAN-generated images to assess robustness and investigate the potential of synthetic medical data.

**Key Findings**
The results showed that unimodal clinical-data models could outperform multimodal configurations, demonstrating the importance and predictive strength of structured clinical metadata.

The strongest overall model was an MLP trained on the WDBC dataset, achieving 99.04% accuracy and an F1-score of 0.9913.

Among the multimodal experiments, the configuration using GAN-generated synthetic mammograms achieved 96.3% accuracy. A random modality-ablation experiment achieved 92.6% accuracy and an F1-score of 0.93, demonstrating the ability of the system to maintain strong performance when modalities are unavailable.

**Disclaimer**

This project was developed for academic and research purposes. It is not a clinical diagnostic system and should not be used to make medical decisions.
