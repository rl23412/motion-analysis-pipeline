# Motion Analysis Pipeline with ComBat Batch Correction

A comprehensive MATLAB pipeline for analyzing behavioral motion data using DANNCE pose estimation with ComBat batch correction and SocialMapper-based behavioral embedding techniques.

## 🔬 **Overview**

This pipeline processes 3D pose estimation data to create behavioral embeddings using:

1. **ComBat Batch Correction** - Removes technical batch effects while preserving biological variation
2. **Behavioral Embedding** - Uses PCA, wavelet analysis, and t-SNE for dimensionality reduction  
3. **Watershed Segmentation** - Automatically segments behavioral space into discrete regions
4. **Re-embedding** - Projects new data onto trained behavioral maps
5. **Statistical Analysis** - Comprehensive group comparisons and visualization

### **Key Features**

- 🧪 **ComBat Integration** - Advanced batch correction for multi-batch datasets
- 🗺️ **SocialMapper Framework** - Built on proven behavioral analysis methods
- 📊 **Comprehensive Analysis** - Statistical testing and visualization suite
- ⚙️ **Easy Configuration** - Simple parameter modification and group management
- 🔄 **Modular Design** - Clean separation of training and analysis phases
- 📁 **Professional Structure** - Standard software engineering organization

## 🚀 **Quick Start**

### **Basic Usage**

```matlab
% 1. Add pipeline to MATLAB path (first time setup)
run('scripts/run_pipeline.m');

% 2. Run with default settings
results = run_pipeline();

% 3. Run with custom data directory
results = run_pipeline('data_dir', '/path/to/your/data');
```

### **Custom Configuration**

```matlab
% Load and modify configuration
config = pipeline_config();
config.data_dir = '/your/data/path';
config.parameters.pca.numComponents = 20;

% Run with custom config
results = run_pipeline('config', config);
```

## 📋 **Installation**

### **Prerequisites**

#### Required MATLAB Toolboxes
- Statistics and Machine Learning Toolbox (for t-SNE)
- Image Processing Toolbox (for watershed segmentation)
- Signal Processing Toolbox (recommended)

#### Hardware Requirements
- **Minimum**: 16 GB RAM
- **Recommended**: 32+ GB RAM for large datasets
- **Storage**: ~5 GB for complete pipeline results

### **Setup**

1. **Clone or download this repository**
2. **Open MATLAB** and navigate to the pipeline directory
3. **Run the pipeline** - all paths will be set automatically:
   ```matlab
   results = run_pipeline();
   ```

The pipeline automatically:
- ✅ Adds all necessary paths
- ✅ Validates installation and dependencies  
- ✅ Checks for required MATLAB toolboxes
- ✅ Creates output directories

## 📊 **Pipeline Workflow**

### **Phase 1: Training**
1. **Load Data** - Import pose estimation files (.mat format)
2. **Feature Extraction** - Calculate pairwise distances and joint velocities
3. **PCA Analysis** - Dimensionality reduction with batch processing
4. **ComBat Correction** - Remove batch effects while preserving biology
5. **t-SNE Embedding** - Create behavioral map from corrected features
6. **Watershed Regions** - Segment behavioral space automatically

### **Phase 2: Re-embedding** 
1. **Load New Data** - Import files for analysis
2. **Feature Matching** - Extract same features as training
3. **Project to Map** - Re-embed onto trained behavioral space
4. **Region Assignment** - Map behaviors to watershed regions

### **Phase 3: Analysis**
1. **Statistical Testing** - Group comparisons and significance tests
2. **Visualization** - Generate comprehensive plots and overlays
3. **Export Results** - Save analysis outputs and figures

## ⚙️ **Configuration**

### **Main Configuration File**

Edit `config/pipeline_config.m` to customize:

```matlab
% Data settings
config.data_dir = '/path/to/your/data';

% Algorithm parameters  
config.parameters.pca.numComponents = 15;
config.parameters.tsne.numPerDataSet = 320;
config.watershed.sigma_density = 0.8;

% ComBat settings
config.combat.enabled = true;
config.combat.parametric = true;
```

### **File Management**

The pipeline automatically handles files matching these patterns:
- `DRG_*.mat` - DRG group files
- `SC_*.mat` - SC group files  
- `SNI_*.mat` - SNI group files
- `IT_*.mat` - IT group files
- `week4-*.mat` - Week 4 timepoint files
- `TBI_*.mat` - TBI group files

To add new file types, modify `config.files.all_files` in the configuration.

## 📁 **Project Structure**

```
motion-analysis-pipeline/
├── README.md                    # This file
├── scripts/
│   └── run_pipeline.m          # Main execution script
├── src/
│   ├── core/                   # Core pipeline functions
│   ├── utils/                  # Utility functions  
│   └── analysis/               # Analysis functions
├── config/
│   └── pipeline_config.m       # Main configuration
├── dependencies/
│   ├── SocialMapper/           # SocialMapper utilities
│   └── Combat/                 # ComBat batch correction
├── tests/                      # Test functions
├── examples/                   # Usage examples
└── docs/                      # Additional documentation
```

## 🔍 **Key Functions**

### **Main Pipeline**
- `run_pipeline()` - Primary entry point
- `custom_embedding_pipeline()` - Core pipeline implementation
- `pipeline_config()` - Configuration management

### **Data Processing**
- `load_training_data()` - Load and validate training data
- `perform_pca_training()` - PCA with batch processing
- `create_training_embedding_combat()` - ComBat-corrected embedding
- `reembed_all_files()` - Re-embed new data onto trained space

### **Analysis Functions**
- `create_watershed_regions()` - Behavioral space segmentation
- `create_comprehensive_analysis()` - Statistical analysis and visualization
- `extract_metadata_from_filename()` - Parse experimental metadata

## 🧬 **Scientific Background**

### **SocialMapper Foundation**

This pipeline is built upon the SocialMapper framework:
- **Repository**: [SocialMapper](https://github.com/uklibaite/SocialMapper)
- **Method**: Combat-based motion mapping for behavioral analysis
- **Innovation**: Specialized for neuroscience applications with ComBat correction

### **ComBat Batch Correction**

ComBat addresses the critical issue of batch effects in behavioral data:
- **Problem**: Technical variation between experimental batches
- **Solution**: Empirical Bayes correction preserving biological differences
- **Benefit**: Improved statistical power and reproducibility

### **Behavioral Embedding Approach**

1. **Feature Extraction**: Pairwise joint distances, velocities, Z-coordinates
2. **PCA Preprocessing**: Dimensionality reduction for computational efficiency  
3. **Wavelet Analysis**: Time-frequency decomposition of behavioral patterns
4. **t-SNE Embedding**: Non-linear mapping preserving local neighborhood structure
5. **Watershed Segmentation**: Data-driven behavioral region discovery

## 📊 **Usage Examples**

### **Example 1: Standard Analysis**

```matlab
% Run complete pipeline with default settings
results = run_pipeline();

% Check number of behavioral regions found
fprintf('Found %d behavioral regions\n', max(results.LL(:)));

% Display group statistics
display_group_summary(results);
```

### **Example 2: Custom Analysis Parameters**

```matlab
% Load configuration and modify parameters
config = pipeline_config();
config.parameters.pca.numComponents = 20;
config.watershed.sigma_density = 1.0;
config.combat.enabled = false;  % Disable ComBat

% Run with modified settings  
results = run_pipeline('config', config);
```

### **Example 3: Analysis-Only Mode**

```matlab
% Skip training and use existing results
results = run_pipeline('skip_training', true);

% Generate additional custom analysis
analyze_specific_regions(results, [1, 5, 10, 15]);
```

## 🔧 **Troubleshooting**

### **Common Issues**

1. **Missing t-SNE function**
   ```
   Error: Undefined function 'tsne'
   Solution: Install Statistics and Machine Learning Toolbox
   ```

2. **ComBat not found**
   ```
   Warning: Using simplified batch correction
   Solution: Install full ComBat package for optimal results
   ```

3. **Memory issues with large datasets**
   ```matlab
   config.parameters.pca.batchSize = 10000;  % Reduce from default 30000
   config.parameters.tsne.numPerDataSet = 200;  % Reduce from default 320
   ```

4. **File not found errors**
   ```matlab
   config.data_dir = '/correct/path/to/your/data';  % Verify path
   ```

### **Performance Optimization**

- **Large datasets**: Reduce batch sizes and subsampling parameters
- **Memory constraints**: Process files in smaller batches
- **Speed**: Enable parallel processing if Parallel Computing Toolbox available

## 🤝 **Attribution**

### **SocialMapper Acknowledgment**

This pipeline is built upon the excellent SocialMapper framework:

> **SocialMapper**: Combat-based motion mapper for behavioral analysis  
> **Repository**: https://github.com/uklibaite/SocialMapper  
> **Method**: Behavioral embedding and analysis for motion data

### **ComBat Method**

ComBat batch correction method:
> Johnson, W.E., Li, C., Rabinovic, A. (2007). Adjusting batch effects in microarray expression data using empirical Bayes methods. *Biostatistics*, 8(1), 118-127.

### **MotionMapper Foundation**

Original behavioral embedding framework:
> Berman, G.J., Choi, D.M., Bialek, W., Shaevitz, J.W. (2014). Mapping the stereotyped behaviour of freely moving fruit flies. *Journal of The Royal Society Interface*, 11(99).

## 📄 **License**

This project maintains compatibility with the original SocialMapper license while adding specialized neuroscience applications.

## 🐛 **Reporting Issues**

Please report issues with:
1. **System information** (MATLAB version, OS, toolboxes)
2. **Data characteristics** (file sizes, number of files, experimental groups)
3. **Error messages** (complete stack traces)
4. **Configuration used** (relevant config parameters)

---

## 🧠 **Scientific Applications**

This pipeline is designed for:
- **Pain Research** - Spontaneous behavior analysis
- **Neuroscience** - Behavioral phenotyping  
- **Pharmacology** - Drug effect characterization
- **Disease Models** - Behavioral biomarker discovery

**Ready for reproducible behavioral analysis with professional software engineering standards!**