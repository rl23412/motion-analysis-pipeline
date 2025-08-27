function config = pipeline_config()
% PIPELINE_CONFIG - Main configuration for the motion analysis pipeline
%
% This function returns the complete configuration structure for the
% custom embedding pipeline with ComBat batch correction
%
% Returns:
%   config - Structure containing all pipeline parameters

config = struct();

%% Data Configuration
config.data_dir = '/work/rl349/dannce/mouse14/allData';  % Default data directory

%% File Configuration
config.files = struct();

% Define all available files
config.files.all_files = {
    'DRG_1.mat', 'DRG_2.mat', 'DRG_3.mat', 'DRG_4.mat', 'DRG_5.mat', ...
    'IT_1.mat', 'IT_2.mat','IT_3.mat', ...
    'SC_1.mat', 'SC_2.mat', 'SC_3.mat', 'SC_4.mat', 'SC_5.mat', 'SC_6.mat', ...
    'SNI_1.mat', 'SNI_2.mat', 'SNI_3.mat', ...
    'week4-DRG_1.mat', 'week4-DRG_2.mat', 'week4-DRG_3.mat', ...
    'week4-SC_1.mat', 'week4-SC_2.mat', 'week4-SC_3.mat', ...
    'week4-SNI_1.mat', 'week4-SNI_2.mat', 'week4-SNI_3.mat', ...
    'week4-TBI_1.mat', 'week4-TBI_2.mat', 'week4-TBI_3.mat', 'week4-TBI_4.mat'
};

% Use all files for training (robust embedding)
config.files.training_files = config.files.all_files;

% Re-embed all files to produce outputs for every dataset
config.files.reembedding_files = config.files.all_files;

%% Skeleton Configuration (mouse14 format - 14 joints)
config.skeleton = struct();

% Joint connections
config.skeleton.joints_idx = [
    1 2; 1 3; 2 3; ...    % head connections: Snout-EarL, Snout-EarR, EarL-EarR
    1 4; 4 5; 5 6; ...    % spine: Snout-SpineF, SpineF-SpineM, SpineM-Tail(base)
    4 7; 7 8; ...         % left front limb: SpineF-ForShdL, ForShdL-ForepawL
    4 9; 9 10; ...        % right front limb: SpineF-ForeShdR, ForeShdR-ForepawR
    5 11; 11 12; ...      % left hind limb: SpineM-HindShdL, HindShdL-HindpawL
    5 13; 13 14           % right hind limb: SpineM-HindShdR, HindShdR-HindpawR
];

% Define colors for visualization
chead = [1 .6 .2];        % orange
cspine = [.2 .635 .172];  % green
cLF = [0 0 1];            % blue (left front)
cRF = [1 0 0];            % red (right front)
cLH = [0 1 1];            % cyan (left hind)
cRH = [1 0 1];            % magenta (right hind)

config.skeleton.color = [
    chead; chead; chead; ...      % head connections (3)
    cspine; cspine; cspine; ...   % spine connections (3)
    cLF; cLF; ...                 % left front limb (2)
    cRF; cRF; ...                 % right front limb (2)  
    cLH; cLH; ...                 % left hind limb (2)
    cRH; cRH                      % right hind limb (2)
];

%% Algorithm Parameters
config.parameters = struct();

% Sampling and frequency parameters
config.parameters.samplingFreq = 50;    % Hz
config.parameters.minF = 0.5;           % Hz
config.parameters.maxF = 20;            % Hz
config.parameters.numModes = 20;        % Number of wavelet modes

% PCA parameters
config.parameters.pca = struct();
config.parameters.pca.numComponents = 15;    % Number of PCA components
config.parameters.pca.pcaModes = 20;         % PCA modes for processing
config.parameters.pca.batchSize = 30000;     % Batch size for PCA computation

% t-SNE parameters
config.parameters.tsne = struct();
config.parameters.tsne.numPerDataSet = 320;  % Subsampling for t-SNE

% Re-embedding parameters
config.parameters.reembedding = struct();
config.parameters.reembedding.batchSize = 10000;  % Batch size for re-embedding

%% ComBat Batch Correction Parameters
config.combat = struct();
config.combat.enabled = true;                    % Enable ComBat correction
config.combat.parametric = true;                 % Use parametric ComBat
config.combat.use_covariates = true;             % Include covariates
config.combat.handle_confounding = true;         % Handle confounded covariates

%% Watershed Parameters
config.watershed = struct();
config.watershed.sigma_density = 0.8;           % Density estimation sigma
config.watershed.grid_size = 501;               % Grid size for density map
config.watershed.connectivity = 18;             % Watershed connectivity
config.watershed.padding = 0.05;                % Padding for symmetric bounds
config.watershed.vSmooth = 0.5;                 % Smoothing for region finding

%% Visualization Colors for Groups
config.colors = struct();
config.colors.DRG = [1 0 0];           % Red
config.colors.SC = [0 0 1];            % Blue
config.colors.IT = [0 1 0];            % Green
config.colors.SNI = [1 0.5 0];         % Orange
config.colors.TBI = [0.5 0 1];         % Purple
config.colors.week4_DRG = [0.7 0 0];   % Dark Red
config.colors.week4_SC = [0 0 0.7];    % Dark Blue
config.colors.week4_SNI = [0.7 0.3 0]; % Dark Orange
config.colors.week4_TBI = [0.3 0 0.7]; % Dark Purple

%% Processing Options
config.options = struct();
config.options.verbose = true;                  % Verbose output
config.options.save_intermediates = true;       % Save intermediate results
config.options.create_visualizations = true;    % Generate plots
config.options.parallel_processing = false;     % Use parallel processing (if available)

%% Output Configuration
config.output = struct();
config.output.base_dir = 'outputs';             % Base output directory
config.output.save_format = 'v7.3';             % MAT file format for large files
config.output.figure_format = 'png';            % Figure format
config.output.figure_dpi = 300;                 % Figure DPI

%% Dependencies Configuration
config.dependencies = struct();
config.dependencies.socialmapper_path = 'dependencies/SocialMapper';
config.dependencies.combat_path = 'dependencies/Combat';
config.dependencies.required_toolboxes = {
    'Statistics and Machine Learning Toolbox'
    'Image Processing Toolbox'
    'Signal Processing Toolbox'
};

end