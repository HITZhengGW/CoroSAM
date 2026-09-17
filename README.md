# CoroSAM: Enhancing SAM With Frequency and Orientation Awareness for Coronary Artery Segmentation in X-Ray Angiography
This is the official repo for the paper [CoroSAM: Enhancing SAM With Frequency and Orientation Awareness for Coronary Artery Segmentation in X-Ray Angiography](https://ieeexplore.ieee.org/document/11355979), accepted at BIBM 2025.

## Usage

1. `orientation_map.py` — extracts the orientation map of the input image.
2. `gsfa.py` — use this to replace LoRA fine-tuning the encoder of SAM.
3. `oga.py` — use this code to fine-tune the decoder of SAM.

> **Note:** For the construction of the SAM model, please refer to [SAMed](https://arxiv.org/pdf/2304.13785).

## Citing CoroSAM

If you find CoroSAM useful, please cite our paper.

```
@INPROCEEDINGS{11355979,
  author={Zheng, Guowei and Bo, Pengbo and Liu, Liangliang and Cong, Zhaoyang and Tang, Kegeng and Zhang, Caiming},
  booktitle={2025 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)}, 
  title={CoroSAM: Enhancing SAM With Frequency and Orientation Awareness for Coronary Artery Segmentation in X-Ray Angiography}, 
  year={2025},
  volume={},
  number={},
  pages={3349-3356},
  keywords={Image segmentation;Angiography;Frequency-domain analysis;Biological system modeling;Logic gates;Robustness;Data models;X-ray imaging;Arteries;Signal to noise ratio;Segment Anything Model;Coronary Artery Segmentation;X-ray Angiography},
  doi={10.1109/BIBM66473.2025.11355979}}
