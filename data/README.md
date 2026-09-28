# Dataset Setup Guide

This directory holds the raw dataset files used in the benchmark experiments.

### Expected Files and Directory Structure

1. **Scene-15**:
   - File: `data/Scene15_multiview.mat`
   - Content: Multi-view features for 15 natural scene categories (contains `X1`, `X2`, `X3`, `Y`).

2. **UCI Handwritten (Multiple Features)**:
   - File: `data/HandWritten.mat`
   - Content: Digit dataset containing 6 feature views (`mfeat-fou`, `mfeat-fac`, `mfeat-kar`, `mfeat-pix`, `mfeat-zer`, `mfeat-mor`).

3. **CUB-200-2011**:
   - Directory: `data/CUB_200_2011/images/` or `CUB_200_2011/images/`
   - Content: Image folders of bird species. The loader automatically extracts ResNet-18 features and RGB 3D color histograms, caching them as `cub_features_v2.npz`.

4. **MNIST + USPS**:
   - Automatically downloaded via `sklearn.datasets.fetch_openml`.
   - Fallback: If USPS is unavailable, generates HOG features from MNIST digits.

5. **Reuters-21578**:
   - Automatically downloaded via `nltk.download("reuters")`.
   - Views: Word-level TF-IDF and Character n-gram TF-IDF.

*Note: Environment variables `PATH_SCENE15`, `PATH_HANDWRITTEN`, and `PATH_CUB` can also be used to point directly to custom file locations.*
