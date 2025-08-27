function config = pipeline_config()
%% PIPELINE_CONFIG Main configuration for spontaneous pain analysis pipeline
% This file contains all configuration parameters for the behavioral
% embedding and analysis pipeline.

config = struct();

%% Version and metadata
config.version = '1.4.0';
config.description = 'Spontaneous pain behavioral analysis pipeline';
config.created = datestr(now);

%% Data configuration
config.data = struct();
config.data.format = 'rat23';  % 'mouse14' or 'rat23'
config.data.expected_dimensions = [NaN, 3, 23];  % [frames, coords, joints]
config.data.coordinate_order = {'x', 'y', 'z'};
config.data.joint_names = {
    'Snout', 'EarL', 'EarR', 'Spine1', 'Spine2', 'Spine3', 'TailBase', ...
    'ArmL1', 'ArmL2', 'ArmL3', 'ArmL4', 'ArmR1', 'ArmR2', 'ArmR3', 'ArmR4', ...
    'LegL1', 'LegL2', 'LegL3', 'LegL4', 'LegR1', 'LegR2', 'LegR3', 'LegR4'
};

%% Pipeline parameters
config.parameters = struct();

% PCA parameters
config.parameters.pca = struct();
config.parameters.pca.num_components = 15;  % Number of PCA components
config.parameters.pca.batch_size = 30000;   % Batch size for PCA computation

% Wavelet parameters
config.parameters.wavelets = struct();
config.parameters.wavelets.min_freq = 0.5;     % Minimum frequency
config.parameters.wavelets.max_freq = 20;      % Maximum frequency  
config.parameters.wavelets.pca_modes = 20;     % Number of PCA modes
config.parameters.wavelets.sampling_freq = 50; % Sampling frequency (Hz)

% t-SNE parameters
config.parameters.tsne = struct();
config.parameters.tsne.num_per_dataset = 320;  % Templates per dataset
config.parameters.tsne.perplexity = 30;        % t-SNE perplexity
config.parameters.tsne.max_iter = 1000;        % Maximum iterations

% Watershed parameters
config.parameters.watershed = struct();
config.parameters.watershed.connectivity = 18;         % Connectivity (4, 8, 18)
config.parameters.watershed.density_range = [-65 65];  % Range for density computation
config.parameters.watershed.grid_size = 501;           % Grid size for density map
config.parameters.watershed.min_region_size = 10;      % Minimum region size (pixels)

%% Analysis parameters
config.analysis = struct();

% Region segmentation options
config.analysis.resegment_enable = true;          % Re-run watershed segmentation
config.analysis.resegment_sigma = 0;              % Gaussian blur before watershed
config.analysis.resegment_gamma = 1.7;            % Gamma correction (>1 sharpens)
config.analysis.resegment_min_density = 5e-6;     % Minimum density threshold
config.analysis.resegment_connectivity = 4;       % Watershed connectivity
config.analysis.resegment_fill_holes = true;      % Fill holes in regions
config.analysis.resegment_min_region_size = 10;   % Remove small regions
config.analysis.force_all_boundaries = true;      % Force boundaries around all regions

% Density filtering
config.analysis.filter_regions_by_density = false;  % Enable density filtering
config.analysis.region_density_percentile = 5;      % Percentile threshold
config.analysis.region_min_density = 1e-8;          # Absolute minimum density

# Display parameters
config.analysis.display_range_expand = 1;           # Expand coordinate range (>1 = zoom out)
config.analysis.crop_padding = 20;                  # Padding for cropped figures

%% Output configuration
config.output = struct();
config.output.base_dir = 'outputs';                 % Base output directory
config.output.save_intermediate = true;             % Save intermediate results
config.output.generate_figures = true;              % Generate visualization figures
config.output.export_csv = true;                    % Export CSV files
config.output.compression_level = 6;                % MAT file compression (0-9)
config.output.figure_format = 'png';                % Figure format ('png', 'jpg', 'eps')
config.output.figure_dpi = 300;                     % Figure resolution

%% Skeleton configuration for visualization
config.skeleton = struct();

# Joint connections for rat23 format
config.skeleton.joints_idx = [
    1 2; 1 3; 2 3; ...                    % head connections
    1 4; 4 5; 5 6; 6 7; ...               % spine
    4 8; 8 9; 9 10; 10 11; ...            % left arm
    4 12; 12 13; 13 14; 14 15; ...        % right arm
    6 16; 16 17; 17 18; 18 19; ...        % left leg
    6 20; 20 21; 21 22; 22 23             % right leg
];

% Color scheme for skeleton parts
config.skeleton.colors = struct();
config.skeleton.colors.head = [1 0.6 0.2];         % Orange
config.skeleton.colors.spine = [0.2 0.635 0.172];  % Green
config.skeleton.colors.left_front = [0 0 1];       % Blue
config.skeleton.colors.right_front = [1 0 0];      % Red
config.skeleton.colors.left_hind = [0 1 1];        % Cyan
config.skeleton.colors.right_hind = [1 0 1];       % Magenta

%% Validation parameters
config.validation = struct();
config.validation.check_data_integrity = true;      % Validate input data
config.validation.validate_results = true;          % Validate pipeline results
config.validation.generate_reports = true;          % Generate validation reports
config.validation.min_frames_per_file = 100;        % Minimum frames required
config.validation.max_missing_joints = 2;           % Maximum missing joints allowed

%% Performance parameters
config.performance = struct();
config.performance.use_parallel = false;            # Use parallel processing (if available)
config.performance.max_memory_gb = 16;              % Maximum memory usage (GB)
config.performance.chunk_size = 10000;              % Chunk size for large datasets
config.performance.enable_profiling = false;        % Enable performance profiling

%% Debugging and logging
config.debug = struct();
config.debug.verbose = true;                        % Display progress messages
config.debug.save_debug_info = false;               % Save debugging information
config.debug.plot_intermediate = false;             % Plot intermediate results
config.debug.log_level = 'info';                    % Log level ('debug', 'info', 'warning', 'error')

%% Group colors for visualization
config.group_colors = struct();
config.group_colors.DRG = [1 0 0];        % Red
config.group_colors.SC = [0 0 1];         % Blue  
config.group_colors.IT = [0 1 0];         % Green
config.group_colors.SNI = [1 0.5 0];      % Orange
config.group_colors.TBI = [1 0 1];        % Magenta
config.group_colors.Control = [0.5 0.5 0.5]; % Gray

%% Statistical analysis parameters
config.statistics = struct();
config.statistics.alpha_level = 0.05;               % Significance level
config.statistics.multiple_comparison_method = 'fdr'; % 'bonferroni', 'fdr', 'none'
config.statistics.min_group_size = 3;               % Minimum samples per group
config.statistics.bootstrap_iterations = 1000;      % Bootstrap iterations
config.statistics.confidence_level = 0.95;          % Confidence intervals

%% File patterns and naming
config.files = struct();
config.files.mouse_file_order = 'mouseFileOrder.mat';
config.files.pca_results = 'vecsValsMouse_weekdata.mat';
config.files.embedding_data = 'loneSignalDataAmps_weekdata.mat'; 
config.files.tsne_results = 'train_weekdata.mat';
config.files.final_results = 'mouseEmbeddingResults_weekdata.mat';
config.files.watershed_map = 'watershed_SNI_TBI.mat';
config.files.analysis_results = 'complete_embedding_results_SNI_TBI.mat';

%% Training data configuration
config.training = struct();
config.training.files = {'SNI_2.mat', 'week4-TBI_3.mat'}; % Training data files
config.training.description = 'Files used for behavioral embedding training';
config.training.use_flipping = true;                      % Use left-right flipping
config.training.validation_split = 0.0;                   % Validation data fraction

%% Sequence analysis parameters
config.sequences = struct();
config.sequences.min_length = 10;                   % Minimum sequence length (frames)
config.sequences.max_gap = 1;                       % Maximum gap in continuous sequences
config.sequences.quality_weights = [0.4, 0.3, 0.2, 0.1]; % [length, density, motion, compactness]
config.sequences.top_sequences_per_region = 35;     % Number of top sequences to save

%% Advanced options
config.advanced = struct();
config.advanced.custom_joint_mapping = [];          % Custom joint remapping (if needed)
config.advanced.feature_weights = [1.0, 0.25, 0.5]; % [wavelets, z_coords, velocities]
config.advanced.embedding_subsample_factor = 20;     % Subsample factor for t-SNE
config.advanced.reembedding_batch_size = 10000;     % Batch size for re-embedding

end

%% Helper functions for configuration management

function config = update_config(config, updates)
%UPDATE_CONFIG Update configuration with new values
%   config = update_config(config, updates)
%   
% Recursively updates configuration structure with values from updates struct
    
    if ~isstruct(updates)
        error('Updates must be a structure');
    end
    
    fields = fieldnames(updates);
    for i = 1:length(fields)
        field = fields{i};
        if isfield(config, field) && isstruct(config.(field)) && isstruct(updates.(field))
            config.(field) = update_config(config.(field), updates.(field));
        else
            config.(field) = updates.(field);
        end
    end
end

function validate_config(config)
%VALIDATE_CONFIG Validate configuration parameters
    
    % Check required fields
    required_fields = {'parameters', 'output', 'data'};
    for i = 1:length(required_fields)
        if ~isfield(config, required_fields{i})
            error('Missing required configuration field: %s', required_fields{i});
        end
    end
    
    # Validate parameter ranges
    if config.parameters.pca.num_components < 1 || config.parameters.pca.num_components > 50
        warning('PCA components should typically be between 1-50');
    end
    
    if config.parameters.wavelets.min_freq >= config.parameters.wavelets.max_freq
        error('Wavelet min_freq must be less than max_freq');
    end
    
    if config.parameters.watershed.grid_size < 100
        warning('Small grid size may result in poor resolution');
    end
    
    fprintf('Configuration validation passed\n');
end

function display_config_summary(config)
%DISPLAY_CONFIG_SUMMARY Display a summary of the current configuration
    
    fprintf('\n=== Pipeline Configuration Summary ===\n');
    fprintf('Version: %s\n', config.version);
    fprintf('Data format: %s (%d joints)\n', config.data.format, size(config.data.expected_dimensions, 3));
    fprintf('PCA components: %d\n', config.parameters.pca.num_components);
    fprintf('Wavelet frequency range: %.1f - %.1f Hz\n', ...
        config.parameters.wavelets.min_freq, config.parameters.wavelets.max_freq);
    fprintf('t-SNE templates per dataset: %d\n', config.parameters.tsne.num_per_dataset);
    fprintf('Watershed connectivity: %d\n', config.parameters.watershed.connectivity);
    fprintf('Output directory: %s\n', config.output.base_dir);
    fprintf('Generate figures: %s\n', mat2str(config.output.generate_figures));
    fprintf('Export CSV: %s\n', mat2str(config.output.export_csv));
    fprintf('=====================================\n\n');
end