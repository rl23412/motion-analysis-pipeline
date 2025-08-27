# Spontaneous Pain Analysis Pipeline

A comprehensive MATLAB pipeline for analyzing spontaneous pain behaviors using DANNCE pose estimation data. This pipeline uses behavioral embedding techniques to identify and quantify pain-related behavioral patterns in mice.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Overview

This pipeline processes 3D pose estimation data from mice to:

1. **Create behavioral embeddings** using dimensionality reduction and t-SNE
2. **Segment behavioral space** into discrete regions using watershed segmentation
3. **Re-embed individual videos** onto the trained behavioral map
4. **Perform statistical analysis** of behavioral differences between groups
5. **Export results** for downstream analysis and visualization

### Key Features

- 🔄 **Left-Right Flipping**: Data augmentation for improved behavioral mapping
- 🗺️ **Watershed Segmentation**: Automatic behavioral region detection
- 📊 **Statistical Analysis**: Built-in group comparisons and significance testing
- 📁 **Comprehensive Exports**: CSV files for R, Python, or other downstream analysis
- 📈 **Visualization Suite**: Automatic generation of publication-ready figures
- ⚙️ **Easy Configuration**: Simple group management and parameter tuning

## Installation

### Prerequisites

#### Required MATLAB Toolboxes
- Statistics and Machine Learning Toolbox (for t-SNE)
- Image Processing Toolbox (for watershed segmentation)
- Signal Processing Toolbox (recommended)

#### Hardware Requirements
- **Minimum**: 16 GB RAM
- **Recommended**: 32+ GB RAM
- **Storage**: ~5 GB for full pipeline results

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd spontaneous-pain-pipeline
   ```

2. **Initialize the environment**:
   ```matlab
   run('scripts/setup_pipeline.m')
   ```

3. **Install dependencies** (if needed):
   ```bash
   ./scripts/install_dependencies.sh
   ```

4. **Validate installation**:
   ```matlab
   run('tests/validate_installation.m')
   ```

## Quick Start

### Basic Pipeline Execution

```matlab
% 1. Setup environment (first time only)
run('scripts/setup_pipeline.m')

% 2. Run full pipeline with default settings
run('scripts/run_pipeline.m')

% 3. Check results
ls outputs/
```

### Custom Configuration

```matlab
% 1. Edit configuration file
edit('config/pipeline_config.m')

% 2. Run with custom settings
config = load_config('config/my_experiment.m');
run_pipeline('config', config);
```

## Usage

### Running the Complete Pipeline

```matlab
% Full pipeline (training + analysis)
results = run_pipeline();

% Skip training (use existing model)
results = run_pipeline('skip_training', true);

% Analysis only
results = run_pipeline('analysis_only', true);
```

### Adding New Groups

```matlab
% Method 1: Edit configuration file
edit('config/group_config.m');

% Method 2: Programmatically add groups
config = load_config();
config = add_group(config, 'week1', 'NewGroup', {'file1.mat', 'file2.mat'}, [1 0 0], 'Description');
save_config(config, 'config/updated_config.m');
```

### Custom Analysis

```matlab
% Load pipeline results
load('outputs/results/pipeline_results.mat');

% Run custom downstream analysis
analyze_custom_regions(results, 'region_list', [1, 5, 10]);
```

## Project Structure

```
spontaneous-pain-pipeline/
├── README.md                    # This file
├── LICENSE                      # License information
├── .gitignore                   # Git ignore rules
├── requirements.txt             # Dependencies list
│
├── src/                         # Source code
│   ├── core/                    # Core pipeline functions
│   │   ├── mouse_embedding.m    # Main embedding pipeline
│   │   ├── analyze_maps.m       # Downstream analysis
│   │   └── ...
│   ├── utils/                   # Utility functions
│   │   ├── data_validation.m    # Input validation
│   │   ├── plotting.m           # Visualization utilities
│   │   └── ...
│   └── analysis/                # Analysis-specific functions
│       ├── statistical_tests.m  # Statistical analysis
│       ├── region_analysis.m    # Region-specific analysis
│       └── ...
│
├── config/                      # Configuration files
│   ├── pipeline_config.m        # Main pipeline configuration
│   ├── group_config.m          # Group definitions
│   └── examples/               # Example configurations
│
├── data/                       # Data directory
│   ├── raw/                    # Raw input data
│   ├── processed/              # Intermediate results
│   └── README.md               # Data format documentation
│
├── dependencies/               # External dependencies
│   ├── MotionMapper/           # MotionMapper toolbox
│   ├── SocialMapper/           # SocialMapper utilities
│   └── compiled/               # Compiled MEX files
│
├── scripts/                    # Automation scripts
│   ├── setup_pipeline.m        # Environment setup
│   ├── run_pipeline.m          # Main execution script
│   ├── install_dependencies.sh # Dependency installation
│   └── ...
│
├── tests/                      # Test suite
│   ├── validate_installation.m # Installation validation
│   ├── test_pipeline.m         # Pipeline testing
│   └── ...
│
├── examples/                   # Usage examples
│   ├── basic_usage.m           # Basic pipeline usage
│   ├── custom_analysis.m       # Custom analysis examples
│   └── ...
│
├── docs/                       # Documentation
│   ├── user_guide.md           # Detailed user guide
│   ├── api_reference.md        # Function reference
│   └── troubleshooting.md      # Common issues and solutions
│
└── outputs/                    # Generated outputs
    ├── results/                # Main pipeline results
    ├── figures/                # Generated figures
    ├── csv/                    # Data exports
    └── validation/             # Validation reports
```

## Configuration

### Pipeline Parameters

Edit `config/pipeline_config.m` to customize:

```matlab
config.parameters.pca.numComponents = 15;     % PCA components
config.parameters.tsne.perplexity = 30;       % t-SNE perplexity
config.parameters.watershed.connectivity = 18; % Watershed connectivity
```

### Group Configuration

Edit `config/group_config.m` to define experimental groups:

```matlab
% Add new group
config.groups.week1.Treatment = struct();
config.groups.week1.Treatment.files = {'treat1.mat', 'treat2.mat'};
config.groups.week1.Treatment.color = [1 0 0];  % Red
config.groups.week1.Treatment.description = 'Treatment group';
```

## Examples

### Example 1: Basic Pipeline

```matlab
% Run with default configuration
results = run_pipeline();

% Check results
fprintf('Analysis completed: %d regions found\n', results.num_regions);
show_summary_plots(results);
```

### Example 2: Custom Analysis

```matlab
% Load results
load('outputs/results/pipeline_results.mat');

% Analyze specific regions of interest
regions_of_interest = [1, 5, 10, 15];
custom_stats = analyze_regions(results, 'regions', regions_of_interest);

% Generate custom plots
plot_region_comparisons(custom_stats, 'save_figures', true);
```

### Example 3: Batch Processing

```matlab
% Process multiple datasets
datasets = {'experiment1', 'experiment2', 'experiment3'};

for i = 1:length(datasets)
    config = load_config(sprintf('config/%s_config.m', datasets{i}));
    results = run_pipeline('config', config, 'output_dir', sprintf('outputs/%s', datasets{i}));
end
```

## API Reference

### Main Functions

- `run_pipeline(varargin)` - Main pipeline execution
- `setup_pipeline()` - Environment setup and validation
- `load_config(config_file)` - Load configuration
- `validate_data(data_files)` - Input data validation

### Core Pipeline Functions

- `mouse_embedding(data, config)` - Main embedding pipeline
- `analyze_maps(results, config)` - Downstream analysis
- `create_behavioral_map(embeddings)` - Generate behavioral maps
- `statistical_analysis(group_data)` - Statistical comparisons

### Utility Functions

- `add_group(config, week, name, files, color, description)` - Add experimental group
- `validate_installation()` - Check dependencies
- `generate_report(results)` - Create analysis report

See `docs/api_reference.md` for detailed function documentation.

## Validation

The pipeline includes comprehensive validation:

```matlab
% Validate installation
validate_installation();

% Validate input data
validate_data('data/raw/');

% Validate pipeline results
validate_pipeline_results('outputs/results/');
```

## Troubleshooting

### Common Issues

1. **Missing t-SNE function**
   ```
   Error: Undefined function 'tsne'
   Solution: Install Statistics and Machine Learning Toolbox
   ```

2. **MEX file errors**
   ```bash
   # Recompile MEX files
   ./scripts/compile_mex_files.sh
   ```

3. **Memory issues**
   ```matlab
   % Reduce batch size in configuration
   config.parameters.pca.batchSize = 10000;  % Default: 30000
   ```

4. **Path issues**
   ```matlab
   % Re-run setup
   setup_pipeline();
   ```

See `docs/troubleshooting.md` for detailed solutions.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-analysis`)
3. Commit your changes (`git commit -am 'Add new analysis method'`)
4. Push to the branch (`git push origin feature/new-analysis`)
5. Create a Pull Request

### Development Guidelines

- Follow MATLAB coding standards
- Add tests for new functionality
- Update documentation
- Ensure backwards compatibility

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@article{spontaneous_pain_pipeline_2024,
  title={Spontaneous Pain Analysis Pipeline: Behavioral Embedding for Pain Assessment},
  author={[Your Name]},
  journal={[Journal]},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project builds upon excellent prior work:
- **[SocialMapper](https://github.com/uklibaite/SocialMapper)**: Combat-based motion mapping framework
- **MotionMapper**: Original behavioral embedding methodology  
- **DANNCE**: 3D pose estimation framework

See [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md) for detailed attribution and scientific references.

## Version History

- **v1.0.0** - Initial release with core pipeline
- **v1.1.0** - Added group configuration system  
- **v1.2.0** - Enhanced validation and error handling
- **v1.3.0** - Standard software engineering structure
- **v1.4.0** - Comprehensive documentation and examples

---

For questions, issues, or feature requests, please open an issue on GitHub or contact [maintainer email].