# CTRhythm: Accurate Atrial Fibrillation Detection from Single-Lead ECG Rhythm by Convolutional Neural Network and Transformer Integration


## Summary
We developed a CNN and transformer-based model, CTRhythm, for the accurate detection of AF using single-lead ECG data.

![image](./fig1.svg)


CTRhythm consists of four modules in order: 
1. 1D Convolutional Neural Network Module (1-D CNN): this module downsamples the input ECG sequence and aggregates local patterns using CNN (Fig. 2b); 
2. Position Encoding Module: this module applies positional encoding to the CNN-encoded sequence to incorporate contextual position information (Fig. 2a);
3. Transformer Encoder Module: this module performs attention-based learning of long-range dependencies using the position-CNN-encoded sequence (Fig. 2d);
4. Classification Module: this module consists of a global average pooling layer, a linear layer, and a SoftMax layer to generate prediction scores and labels (Fig. 2a).

## Data
The CINC2017 original data is available in PhysioNet: https://physionet.org/content/challenge-2017/1.0.0/




