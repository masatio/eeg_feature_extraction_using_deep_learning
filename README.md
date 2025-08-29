# EEG-based Visual Stimulus Classification with Deep Learning

This project explores whether computer vision models can be enhanced by incorporating **human perception signals**, specifically EEG recordings collected while subjects view visual stimuli. Using the **EEG-ImageNet** dataset, several unimodal and multimodal approaches were evaluated for classifying images and their corresponding brain activity.

---

## Key Contributions
- Reproduced and extended experiments from Mishra et al. (2022) on **EEG-based image feature extraction**.  
- Implemented and compared five approaches:  
  1. **ResNet50** on visual stimuli only  
  2. **LSTM** on raw EEG signals  
  3. **EfficientNetB2** on EEG heatmaps  
  4. **Multimodal (ResNet50 + LSTM)** combining images and EEG signals  
  5. **Multimodal (ResNet50 + EfficientNetB2)** combining images and EEG heatmaps  
- Showed that **EEG-to-heatmap encoding significantly improves classification performance** compared to raw EEG features.  
- Demonstrated the **power of multimodal fusion**, reaching up to **95% accuracy**.

---

## Dataset
- **EEG-ImageNet (2020 update)**  
- 6 subjects, 40 image classes (reduced to 39 due to missing samples)  
- 11,682 EEG recordings (128 channels, 1000 Hz, ~500 ms per trial)  
- Signals filtered ([14–70] Hz) and normalized  

---

## Results

| Model                        | Accuracy |
|------------------------------|----------|
| ResNet50 (images)            | **0.85** |
| LSTM (EEG)                   | 0.29     |
| EfficientNetB2 (EEG heatmaps)| 0.51     |
| ResNet50 + LSTM (multimodal) | 0.79     |
| ResNet50 + EfficientNetB2    | **0.95** |

---

## Conclusion
- **Heatmap representation** of EEG signals provides richer features than raw signals  
- **Multimodal models** (vision + EEG) significantly outperform unimodal approaches  
- This approach shows promise for advancing **human–computer interaction**, **assistive technologies**, and **biomedical applications** where decoding human perception is critical  

---

## References
- Mishra, A., Raj, N., & Bajwa, G. (2022). *EEG-based image feature extraction for visual classification using deep learning*. IEEE IDSTA.  
- Palazzo, S. et al. (2020). *Decoding brain representations by multimodal learning of neural activity and visual features*. IEEE TPAMI.  

---

👉 [GitHub Repository with Implementations](https://github.com/masatio/eeg_feature_extraction_using_deep_learning)
