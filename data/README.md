# Data Directory

This directory contains input and processed data for the spontaneous pain analysis pipeline.

## Directory Structure

```
data/
├── raw/                    # Raw input data files
│   ├── mouseFileOrder.mat  # File list and metadata (REQUIRED)
│   ├── *.mat              # Individual mouse pose data files
│   └── README.md          # This file
└── processed/             # Intermediate processed data
    ├── *.mat              # Processed embedding results  
    └── cache/             # Temporary cache files
```

## Required Input Data

### Essential Files

1. **mouseFileOrder.mat** (REQUIRED)
   - Contains file list and experimental metadata
   - Required fields:
     - `mouseOrderShort`: Cell array of data filenames
     - `metadata`: Structure with group/week indices

### Training Data Files
- **SNI_2.mat**: Training data for SNI condition
- **week4-TBI_3.mat**: Training data for TBI condition

### Re-embedding Data Files
The pipeline expects the following data files for re-embedding:

#### Week 1 Data
- `DRG_1.mat` through `DRG_5.mat` (5 files)
- `SC_1.mat` through `SC_6.mat` (6 files)

#### Week 2 Data  
- `IT_1.mat`, `IT_2.mat` (2 files)
- `SNI_1.mat`, `SNI_3.mat` (2 files, SNI_2 used for training)
- `week4-DRG_1.mat` through `week4-DRG_3.mat` (3 files)
- `week4-SC_1.mat` through `week4-SC_3.mat` (3 files)
- `week4-SNI_1.mat` through `week4-SNI_3.mat` (3 files)
- `week4-TBI_1.mat`, `week4-TBI_2.mat`, `week4-TBI_4.mat` (3 files, TBI_3 used for training)

**Total: 30 data files (28 for re-embedding + 2 for training)**

## Data Format Requirements

### Pose Data Structure
Each .mat file should contain:
```matlab
data.pred  % Main pose data array
% Dimensions: [n_frames, 3, n_joints]
% - n_frames: Number of video frames
% - 3: [x, y, z] coordinates  
% - n_joints: 23 (for rat23 format) or 14 (for mouse14 format)
```

### Joint Layout (rat23 format)
```
Joints 1-3:   Head (Snout, EarL, EarR)
Joints 4-7:   Spine (Spine1, Spine2, Spine3, TailBase)
Joints 8-11:  Left arm (ArmL1, ArmL2, ArmL3, ArmL4)  
Joints 12-15: Right arm (ArmR1, ArmR2, ArmR3, ArmR4)
Joints 16-19: Left leg (LegL1, LegL2, LegL3, LegL4)
Joints 20-23: Right leg (LegR1, LegR2, LegR3, LegR4)
```

### Metadata Structure
```matlab
metadata.week1_indices.DRG = [1, 2, 3, 4, 5];           % Week 1 DRG file indices
metadata.week1_indices.SC = [9, 10, 11, 12, 13, 14];    % Week 1 SC file indices
metadata.week2_indices.DRG = [18, 19, 20];              % Week 2 DRG file indices
metadata.week2_indices.IT = [6, 7];                     % Week 2 IT file indices
metadata.week2_indices.SC = [21, 22, 23];               % Week 2 SC file indices
metadata.week2_indices.SNI = [15, 17];                  % Week 2 SNI file indices
```

## Data Validation

Before running the pipeline, validate your data:

```matlab
% Check data format and structure
run('scripts/setup_pipeline.m');

% Validate specific data files
validate_pose_data('data/raw/DRG_1.mat');
```

### Common Data Issues
- **Missing files**: Ensure all required files are present
- **Wrong dimensions**: Check that pose data has correct [frames, 3, joints] shape
- **Missing joints**: Verify all joint positions are included
- **Coordinate system**: Ensure consistent coordinate system across files
- **Frame rate**: Verify consistent temporal sampling

## Example Data Loading

```matlab
% Load file order and metadata
load('data/raw/mouseFileOrder.mat', 'mouseOrderShort', 'metadata');

% Load individual pose data file
data = load('data/raw/DRG_1.mat');
pose_data = data.pred;  % [frames x 3 x 23]

% Extract specific joint trajectories
snout_trajectory = squeeze(pose_data(:, :, 1));  % [frames x 3]
spine_trajectory = squeeze(pose_data(:, :, 4));  % [frames x 3]
```

## Data Organization Tips

1. **Consistent Naming**: Use consistent file naming conventions
2. **Backup Data**: Keep backups of original raw data
3. **Version Control**: Track data versions if preprocessing changes
4. **Documentation**: Document any preprocessing steps applied
5. **Quality Control**: Visually inspect data for obvious errors

## Troubleshooting

### File Not Found Errors
```bash
# Check if files exist
ls -la data/raw/*.mat

# Verify file permissions
chmod 644 data/raw/*.mat
```

### Data Format Issues
```matlab
% Check data structure
data = load('data/raw/yourfile.mat');
disp(fieldnames(data));
disp(size(data.pred));
```

### Memory Issues
- For large datasets, consider processing in chunks
- Use `whos` command to monitor memory usage
- Close MATLAB figures when not needed

## Getting Sample Data

If you need sample data for testing:

1. **Synthetic Data**: Use `examples/generate_sample_data.m`
2. **Test Data**: Small datasets available in `examples/test_data/`
3. **Documentation Data**: Minimal examples in user guide

For questions about data format or requirements, see the main documentation or contact the pipeline maintainers.