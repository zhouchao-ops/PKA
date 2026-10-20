---
license: Apache License 2.0
pretty_name: Allcondition200K
configs:
- config_name: default
  data_files:
  - split: train 
    path: train/*.parquet
  - split: test
    path: test/*.parquet
size_categories:
- 1K<n<10K

dataset_info:
  features:
  - name: image
    dtype: image
  - name: collection
    dtype: string
  - name: quality_assessment
    struct:
    - name: compositeStructure
      dtype: int64
    - name: objectConsistency
      dtype: int64
    - name: imageQuality
      dtype: int64
  - name: description
    struct:
    - name: item
      dtype: string
    - name: description_0
      dtype: string
    - name: description_1
      dtype: string
    - name: category
      dtype: string
    - name: description_valid
      dtype: bool
  - name: sketch
    dtype: image
  - name: canny
    dtype: image
  - name: depth
    dtype: image
  - name: normal
    dtype: image
  - name: segmentation
    dtype: image
  - name: albedo
    dtype: image
  - name: irradiance
    dtype: image
---

