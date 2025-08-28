# Custom Embedding Pipeline for SNI_2 and week4-TBI_3 Training

This repository contains a modified behavioral embedding pipeline specifically designed to train on `SNI_2.mat` and `week4-TBI_3.mat` files, then re-embed all other files in your dataset using this trained embedding space.

## Overview

### Key Features
- **Targeted Training**: Uses only SNI_2.mat and week4-TBI_3.mat for creating the embedding space
- **Comprehensive Re-embedding**: Re-embeds all remaining 28 files using the trained space
- **Batch Correction**: Includes advanced batch correction and parameter optimization
- **Visualization Suite**: Creates comprehensive plots and analyses
- **Python Interface**: Provides Python utilities for data management and analysis

### Pipeline Architecture
1. Load and process training data (SNI_2.mat + week4-TBI_3.mat)
2. Create flipped versions for asymmetry analysis
3. Perform PCA and wavelet decomposition on training data
4. Generate t-SNE embedding and watershed regions from training data
5. Re-embed all other files onto the trained embedding space
6. Create comprehensive visualizations and analyses

## File Structure

```
your_project/
├── custom_embedding_pipeline_SNI_TBI.m    # Main MATLAB pipeline
├── python_embedding_interface.py          # Python interface and utilities
├── README_SNI_TBI_Embedding.md           # This documentation
└── validation_script.m                   # Validation and testing script
```

## Your Data Structure

The pipeline expects your DANNCE data to be organized as follows:

```
/work/rl349/dannce/mouse14/allData/
├── DRG_1.mat          ├── SC_1.mat           ├── week4-DRG_1.mat
├── DRG_2.mat          ├── SC_2.mat           ├── week4-DRG_2.mat
├── DRG_3.mat          ├── SC_3.mat           ├── week4-DRG_3.mat
├── DRG_4.mat          ├── SC_4.mat           ├── week4-SC_1.mat
├── DRG_5.mat          ├── SC_5.mat           ├── week4-SC_2.mat
├── IT_1.mat           ├── SC_6.mat           ├── week4-SC_3.mat
├── IT_2.mat           ├── SNI_1.mat          ├── week4-SNI_1.mat
├── SNI_2.mat [TRAIN]  ├── SNI_3.mat          ├── week4-SNI_2.mat
└── week4-TBI_3.mat [TRAIN]                  ├── week4-SNI_3.mat
                                             ├── week4-TBI_1.mat
                                             ├── week4-TBI_2.mat
                                             └── week4-TBI_4.mat
```

**Training Files**: SNI_2.mat and week4-TBI_3.mat (marked [TRAIN])
**Re-embedding Files**: All remaining 28 files

## Quick Start

### Prerequisites

**MATLAB Requirements:**
- MATLAB R2018b or newer
- Statistics and Machine Learning Toolbox (for t-SNE)
- Signal Processing Toolbox (for filtering functions)
- MotionMapper toolbox functions (`findWavelets`, `findTemplatesFromData`, etc.)

**Python Requirements (Optional):**
```bash
pip install numpy pandas matplotlib seaborn scipy pathlib
```

### Step 1: Validate Your Data

#### Option A: Python Validation (Recommended)
```bash
cd /path/to/your/project
python python_embedding_interface.py
```

This will:
- Check file availability
- Validate data formats
- Create overview visualizations
- Export metadata summaries
- Generate recommendations

#### Option B: MATLAB Validation
```matlab
% Check if training files exist
data_dir = '/work/rl349/dannce/mouse14/allData';
training_files = {'SNI_2.mat', 'week4-TBI_3.mat'};

for i = 1:length(training_files)
    file_path = fullfile(data_dir, training_files{i});
    if exist(file_path, 'file')
        fprintf('✓ Found: %s\n', training_files{i});
        data = load(file_path);
        if isfield(data, 'pred')
            fprintf('  Shape: %s\n', mat2str(size(data.pred)));
        end
    else
        fprintf('✗ Missing: %s\n', training_files{i});
    end
end
```

### Step 2: Run the Main Pipeline

```matlab
% Make sure you're in the correct directory
cd /path/to/your/project

% Run the main embedding pipeline
custom_embedding_pipeline_SNI_TBI
```

The pipeline will:
1. Load SNI_2.mat and week4-TBI_3.mat
2. Create flipped versions for asymmetry analysis
3. Perform PCA on training data (saves `vecsMus_SNI_TBI_training.mat`)
4. Generate training embeddings (saves `trainingSignalData_SNI_TBI.mat`)
5. Create t-SNE embedding (saves `train_SNI_TBI.mat`)
6. Generate watershed regions (saves `watershed_SNI_TBI.mat`)
7. Re-embed all 28 other files onto the trained space
8. Create comprehensive visualizations
9. Save final results (saves `complete_embedding_results_SNI_TBI.mat`)

### Step 3: Analyze Results

The pipeline automatically creates several types of visualizations:

1. **Training Data Behavioral Density Map**: Shows the behavioral space learned from SNI_2 and week4-TBI_3
2. **All Groups Overview**: Shows how all re-embedded groups distribute in the trained space
3. **Individual Mouse Density Maps**: Separate plots for each experimental group
4. **Temporal Analysis**: Compares Week 1 vs Week 4 distributions

## Output Files

### MATLAB Files
- `vecsMus_SNI_TBI_training.mat`: PCA components and means from training data
- `trainingSignalData_SNI_TBI.mat`: Training data embeddings
- `train_SNI_TBI.mat`: t-SNE embedding of training data
- `watershed_SNI_TBI.mat`: Watershed regions derived from training data
- `complete_embedding_results_SNI_TBI.mat`: Complete results including all re-embeddings

### Python Files
- `dataset_summary.csv`: Detailed metadata for all files
- `dataset_summary_summary.csv`: Summary statistics
- `dataset_overview.png`: Data overview visualizations
- `preprocessing_results.pkl`: Complete preprocessing check results

## Advanced Usage

### Customizing Parameters

Edit the parameters in `custom_embedding_pipeline_SNI_TBI.m`:

```matlab
% PCA parameters
nPCA = 15;              % Number of PCA components (default: 15)
pcaModes = 20;          % Number of wavelet modes (default: 20)

% Frequency parameters
parameters.minF = 0.5;  % Minimum frequency (default: 0.5 Hz)
parameters.maxF = 20;   % Maximum frequency (default: 20 Hz)

% Subsampling
numPerDataSet = 320;    % Points per dataset for t-SNE (default: 320)
```

### Loading Results in Python

```python
from python_embedding_interface import CustomEmbeddingInterface

# Initialize interface
interface = CustomEmbeddingInterface('/work/rl349/dannce/mouse14/allData')

# Load MATLAB results
results = interface.load_embedding_results('complete_embedding_results_SNI_TBI.mat')

# Access embeddings
if results:
    embeddings = results['zEmbeddings_all']
    labels = results['reembedding_labels_all']
    print(f"Loaded {len(embeddings)} embeddings")
```

### Custom Analysis

```python
# Create custom analysis
import numpy as np
import matplotlib.pyplot as plt

# Load your results
interface = CustomEmbeddingInterface()
interface.load_raw_data(['SNI_2.mat', 'week4-TBI_3.mat'])  # Load specific files

# Access training data
sni_data = interface.raw_data['SNI_2.mat']
tbi_data = interface.raw_data['week4-TBI_3.mat']

print(f"SNI_2 data shape: {sni_data.shape}")
print(f"week4-TBI_3 data shape: {tbi_data.shape}")
```

## Troubleshooting

### Common Issues

**Issue: "Training file missing"**
```
Solution: Check that your data directory path is correct:
/work/rl349/dannce/mouse14/allData/
```

**Issue: "No 'pred' field found"**
```
Solution: Verify your .mat files contain the 'pred' field:
matlab> load('SNI_2.mat'); whos
```

**Issue: "Function 'findWavelets' not found"**
```
Solution: Add MotionMapper toolbox to MATLAB path:
matlab> addpath('/path/to/MotionMapper')
```

**Issue: "Insufficient memory"**
```
Solution: Reduce batch size or numPerDataSet:
batchSize = 15000;  % Reduce from 30000
numPerDataSet = 200;  % Reduce from 320
```

### Memory Requirements

- **Minimum RAM**: 16 GB
- **Recommended RAM**: 32 GB
- **Storage**: ~5 GB for all intermediate and final results

### Performance Optimization

1. **Use SSD storage** for faster I/O
2. **Increase MATLAB memory** if available:
   ```matlab
   feature('DefaultCharacterSet', 'UTF-8');
   maxNumCompThreads(8);  % Use 8 CPU threads
   ```
3. **Pre-allocate arrays** when possible
4. **Use parallel processing** if available:
   ```matlab
   parpool('local', 4);  % Use 4 parallel workers
   ```

## Understanding the Results

### Behavioral Space Interpretation

The trained embedding space (derived from SNI_2 and week4-TBI_3) represents:

1. **Density Peaks**: High-probability behavioral states
2. **Watershed Regions**: Distinct behavioral modules
3. **Trajectories**: Transitions between behavioral states
4. **Group Differences**: How different experimental groups occupy this space

### Key Metrics

- **Embedding Coordinates**: 2D positions in behavioral space
- **Watershed Regions**: Discrete behavioral modules (numbered)
- **Density Values**: Probability of behavioral states
- **Group Separations**: Statistical differences between conditions

### Validation Checks

1. **Training Data Quality**: Check that SNI_2 and week4-TBI_3 cover diverse behaviors
2. **Re-embedding Accuracy**: Verify that re-embedded data shows expected group differences
3. **Watershed Regions**: Ensure regions are biologically meaningful
4. **Consistency**: Compare with previous analyses if available

## Citation and References

This pipeline is based on the behavioral embedding methodology from:

- **MotionMapper**: Berman, G.J., et al. "Mapping the stereotyped behaviour of freely moving fruit flies." Journal of The Royal Society Interface (2014).
- **t-SNE**: van der Maaten, L. & Hinton, G. "Visualizing Data using t-SNE." Journal of Machine Learning Research (2008).
- **DANNCE**: Dunn, T.W., et al. "Geometric deep learning enables 3D kinematic profiling across species and environments." Nature Methods (2021).

## Support and Contribution

### Getting Help

1. Check the troubleshooting section above
2. Verify your data format matches the expected structure
3. Run the Python validation script first
4. Check MATLAB path and toolbox availability

### Customization

The pipeline is designed to be modular. Key customization points:

1. **Training Files**: Modify `training_files` variable
2. **Parameters**: Adjust PCA, frequency, and sampling parameters
3. **Visualizations**: Add custom plotting functions
4. **Analysis**: Extend with additional statistical tests

### File Organization

For best results, organize your workflow:

```
your_analysis/
├── data/                           # Your DANNCE data
├── scripts/                        # Pipeline scripts
├── results/                        # Generated results
│   ├── matlab/                    # .mat files
│   ├── figures/                   # Generated plots
│   └── summaries/                 # CSV exports
└── documentation/                  # Your notes and analyses
```

---

**Author**: Custom pipeline for SNI_2 and week4-TBI_3 training  
**Date**: Generated for behavioral embedding analysis  
**Version**: 1.0  

For questions or issues specific to this pipeline, please refer to the troubleshooting section or check the validation outputs.






