# Spontaneous Pain Analysis Pipeline

A comprehensive MATLAB pipeline for analyzing spontaneous pain behaviors using DANNCE pose estimation data. This pipeline uses behavioral embedding techniques to identify and quantify pain-related behavioral patterns in mice.

## Overview

This pipeline processes 3D pose estimation data from mice to:
1. Create behavioral embeddings using dimensionality reduction and t-SNE
2. Segment behavioral space into discrete regions using watershed segmentation
3. Re-embed individual videos onto the trained behavioral map
4. Perform statistical analysis of behavioral differences between groups
5. Export results for downstream analysis and visualization

## Quick Start

1. **Setup Environment**:
   ```matlab
   run('setup_pipeline.m')
   ```

2. **Configure Groups** (if needed):
   ```matlab
   edit('config/group_config.m')
   ```

3. **Run Full Pipeline**:
   ```matlab
   run('run_full_pipeline.m')
   ```

## Pipeline Structure

### Core Scripts
- `mouseEmbedding.m` - Main embedding pipeline with left-right flipping
- `analyze_saved_maps_and_counts.m` - Downstream analysis and visualization  
- `run_full_pipeline.m` - Automated full pipeline execution
- `setup_pipeline.m` - Environment setup and dependency checking

### Configuration
- `config/group_config.m` - Group definitions and experimental parameters
- `config/pipeline_config.m` - Pipeline parameters and settings

### Utilities
- `utils/` - Helper functions and validation scripts
- `dependencies/` - Required external functions and MEX files

### Data Requirements
- Input pose data in MATLAB format (.mat files)
- Expected structure: `data.pred` with dimensions [frames, 3, joints]
- Metadata file: `mouseFileOrder.mat` with file list and group assignments

## Output Structure

```
outputs/
├── results/           # Main pipeline results (.mat files)
├── figures/           # Visualizations and plots
│   ├── per_video/     # Individual video overlays
│   └── group_comparisons/
├── csv/               # Data exports for external analysis
│   ├── frame_indices_per_video/
│   ├── sequence_metadata/
│   └── statistical_analysis/
└── validation/        # Pipeline validation outputs
```

## Key Features

- **Robust Group Configuration**: Easy addition/modification of experimental groups
- **Left-Right Flipping**: Data augmentation for improved behavioral mapping
- **Watershed Segmentation**: Automatic behavioral region detection
- **Statistical Analysis**: Built-in group comparisons and significance testing
- **Comprehensive Exports**: CSV files for R, Python, or other downstream analysis
- **Visualization Suite**: Automatic generation of publication-ready figures

## Dependencies

### Required MATLAB Toolboxes
- Statistics and Machine Learning Toolbox (for t-SNE)
- Image Processing Toolbox (for watershed segmentation)
- Signal Processing Toolbox (recommended)

### External Dependencies
- MotionMapper toolbox (included in `dependencies/`)
- Keypoint3DAnimator (for 3D visualization)

### Hardware Requirements
- Minimum: 16 GB RAM
- Recommended: 32+ GB RAM
- Storage: ~5 GB for full pipeline results

## Usage Examples

### Basic Pipeline Execution
```matlab
% Run with default settings
run_full_pipeline();

% Run with custom configuration
config = load_pipeline_config();
config.groups = {'Control', 'Treatment'};
run_full_pipeline(config);
```

### Adding New Groups
```matlab
% Edit group configuration
edit('config/group_config.m');

% Add new group data
add_group_data('NewGroup', {'file1.mat', 'file2.mat'});

% Re-run pipeline
run_full_pipeline();
```

### Analysis Only (Skip Embedding)
```matlab
% Run only downstream analysis
analyze_saved_maps_and_counts();
```

## Validation

The pipeline includes comprehensive validation:
- Dependency checking (`utils/check_dependencies.m`)
- Data format validation (`utils/validate_input_data.m`)
- Results validation (`utils/validate_pipeline_results.m`)

Run validation:
```matlab
validate_full_pipeline();
```

## Troubleshooting

### Common Issues
1. **Missing t-SNE**: Install Statistics and Machine Learning Toolbox
2. **MEX file errors**: Run `utils/compile_mex_files.m`
3. **Memory issues**: Reduce batch size in `config/pipeline_config.m`
4. **Path issues**: Re-run `setup_pipeline.m`

### Getting Help
- Check `DEPENDENCY_ISSUES_AND_FIXES.md` for detailed troubleshooting
- Validate your setup with `validate_full_pipeline.m`
- Review example data formats in `examples/`

## Citation

If you use this pipeline in your research, please cite:
- The original MotionMapper paper
- Your specific publication using this pain analysis pipeline

## Version History

- v1.0: Initial release with mouse14/rat23 support
- v1.1: Added group configuration system
- v1.2: Enhanced validation and error handling
- v1.3: Comprehensive reorganization and documentation

## License

This project builds upon the MotionMapper toolbox and is provided for research use.