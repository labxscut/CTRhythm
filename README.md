# CTRhythm
CNN Transformer Hybrid ECG Rhythm Classifier

## Objective
In recent years, the widespread use of monitoring devices such as cardiac monitors and smartwatches has led to a significant increase in single-lead ECG data. There is also a pressing demand for accurate AF detection methods to expand the utilization of these data. We aim to develop an automated data-driven heart rhythm classifier that can accurately detect Atrial Fibrillation (AF) in single-lead ECG data. Existing convolutional neural network (CNN) and convolutional recurrent neural network (CRNN) based methods had limited performance. This is because of their reliance on local information alone, though heart rhythms show rich long-range dependency. 

## Novelty
We introduce an innovative automated ECG classifier, CTRhythm, designed for AF detection. Distinguishing itself from existing methods that overlook long-range dependency and rely on local feature selection, CTRhythm integrates local features with long-range dependency using CNN and transformer models, leading to a significant improvement in classification performance. CTRhythm achieved superior performance over existing methods using single-lead ECG data from two independent cohorts.

## Methods
CTRhythm consists of four modules in order: 
1. 1D Convolutional Neural Network (CNN) Module: this module downsamples the input ECG sequence and captures local patterns with CNN encoders;
2. Position Encoding Module: this module applies positional encoding to the CNN-encoded sequence to incorporate the position information; 
3. Transformer Encoder Module: this module carries out attention-based learning of long-range dependency using position-CNN-encoded sequence;
4. Classification Module: this module employs a global average pooling layer, a linear layer and a SoftMax layer to provide classification results and prediction scores.
![image](https://github.com/labxscut/CTRhythm/assets/131430090/972862c7-d5e0-4bc9-8ee1-b758ef195dac)
