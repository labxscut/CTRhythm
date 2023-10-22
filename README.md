# CTRhythm: Accurate Atrial Fibrillation Detection from Single-Lead ECG Rhythm by Convolutional Neural Network and Transformer Integration


## Summary
We developed a CNN and transformer-based model, CTRhythm, for the accurate detection of AF using single-lead ECG data.
![image](https://github.com/labxscut/CTRhythm/assets/131430090/a0cc47f2-8f72-4cc0-850b-bcac9d4d412d)

CTRhythm consists of four modules in order: 
1. 1D Convolutional Neural Network (CNN) Module: this module downsamples the input ECG sequence and captures local patterns with CNN encoders;
2. Position Encoding Module: this module applies positional encoding to the CNN-encoded sequence to incorporate the position information; 
3. Transformer Encoder Module: this module carries out attention-based learning of long-range dependency using position-CNN-encoded sequence;
4. Classification Module: this module employs a global average pooling layer, a linear layer and a SoftMax layer to provide classification results and prediction scores.

## Data
The CINC2017 original data is available in PhysioNet: https://physionet.org/content/challenge-2017/1.0.0/




