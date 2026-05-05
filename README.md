# medical-image-data-augmentation-benchmark
Benchmarking data augmentation, GANs, and diffusion models for improving rare disease detection in chest X-ray images.

## Dataset

Repo uses a simplified single-label subset of NIH ChestX-ray14 prepared for
augmentation experiments.

Current local layout:

```text
data/
├── images/
│   ├── 00000002_000.png
│   └── ...
└── labels.csv
```

`labels.csv` contains one row per image:

```csv
image,label,patient_id
00000002_000.png,No Finding,2
```

Preprocessing applied:

- removed all images with more than one disease label
- kept `No Finding` as its own class
- sampled 20% from each remaining class with seed `42`
- copied the selected images into one flat `data/images/` directory
- stored labels in `data/labels.csv`

Dataset size:

- images: `18,264`
- labels: `18,264`
- disk size: about `6.9 GB`

Class distribution:

| Class | Images | Share |
| --- | ---: | ---: |
| No Finding | 12,072 | 66.10% |
| Infiltration | 1,909 | 10.45% |
| Atelectasis | 843 | 4.62% |
| Effusion | 791 | 4.33% |
| Nodule | 541 | 2.96% |
| Pneumothorax | 439 | 2.40% |
| Mass | 428 | 2.34% |
| Consolidation | 262 | 1.43% |
| Pleural_Thickening | 225 | 1.23% |
| Cardiomegaly | 219 | 1.20% |
| Emphysema | 178 | 0.97% |
| Fibrosis | 145 | 0.79% |
| Edema | 126 | 0.69% |
| Pneumonia | 64 | 0.35% |
| Hernia | 22 | 0.12% |
