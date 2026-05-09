# medical-image-data-augmentation-benchmark
Benchmarking data augmentation, GANs, and diffusion models for improving rare disease detection in chest X-ray images.

## Dataset

Repo currently contains two simplified single-label NIH ChestX-ray14 layouts:

- `data/` - earlier 20% per-class subset
- `data_single_label_full/` - full single-label subset used by the baseline notebook

Full baseline layout:

```text
data_single_label_full/
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
- copied the selected images into one flat `data_single_label_full/images/` directory
- stored labels in `data_single_label_full/labels.csv`

Dataset size:

- images: `91,324`
- labels: `91,324`
- removed multi-label rows: `20,796`
- disk size: about `35 GB`

Class distribution:

| Class | Images | Share |
| --- | ---: | ---: |
| No Finding | 60,361 | 66.10% |
| Infiltration | 9,547 | 10.45% |
| Atelectasis | 4,215 | 4.62% |
| Effusion | 3,955 | 4.33% |
| Nodule | 2,705 | 2.96% |
| Pneumothorax | 2,194 | 2.40% |
| Mass | 2,139 | 2.34% |
| Consolidation | 1,310 | 1.43% |
| Pleural_Thickening | 1,126 | 1.23% |
| Cardiomegaly | 1,093 | 1.20% |
| Emphysema | 892 | 0.98% |
| Fibrosis | 727 | 0.80% |
| Edema | 628 | 0.69% |
| Pneumonia | 322 | 0.35% |
| Hernia | 110 | 0.12% |
