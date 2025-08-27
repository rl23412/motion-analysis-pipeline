function results = custom_embedding_pipeline(varargin)
% CUSTOM_EMBEDDING_PIPELINE - Main pipeline for SNI_TBI behavioral embedding
%
% This pipeline creates behavioral embeddings using ComBat batch correction
% and re-embeds all files in the dataset using trained embedding space
%
% Key Features:
% 1. Uses all files for training PCA and t-SNE embedding with ComBat correction
% 2. Re-embeds all files using the trained embedding space
% 3. Creates watershed regions and behavioral analysis
% 4. Includes comprehensive visualization and analysis
%
% Usage:
%   results = custom_embedding_pipeline()  % Use default config
%   results = custom_embedding_pipeline('data_dir', '/path/to/data')
%   results = custom_embedding_pipeline('config', config_struct)
%
% Inputs:
%   data_dir    - Directory containing .mat files (default: config)
%   config      - Configuration structure (optional)
%
% Outputs:
%   results     - Structure containing all pipeline results
%
% Dependencies:
%   - SocialMapper utilities (findWavelets, findTemplatesFromData, etc.)
%   - ComBat batch correction
%   - MATLAB Statistics and Machine Learning Toolbox (for tsne)
%   - Image Processing Toolbox (for watershed)
%
% Author: Motion Analysis Pipeline
% Based on SocialMapper framework

%% Parse inputs
p = inputParser;
addParameter(p, 'data_dir', [], @ischar);
addParameter(p, 'config', [], @isstruct);
addParameter(p, 'output_dir', 'outputs', @ischar);
addParameter(p, 'skip_training', false, @islogical);
addParameter(p, 'verbose', true, @islogical);
parse(p, varargin{:});

data_dir = p.Results.data_dir;
config = p.Results.config;
output_dir = p.Results.output_dir;
skip_training = p.Results.skip_training;
verbose = p.Results.verbose;

% Load configuration if not provided
if isempty(config)
    config = load_pipeline_config();
end

% Set data directory from config if not provided
if isempty(data_dir)
    data_dir = config.data_dir;
end

if verbose
    fprintf('=== CUSTOM EMBEDDING PIPELINE FOR SNI_TBI TRAINING ===\n');
    fprintf('Data directory: %s\n', data_dir);
    fprintf('Output directory: %s\n', output_dir);
    fprintf('Skip training: %s\n\n', mat2str(skip_training));
end

%% Create output directory
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%% SECTION 1: SETUP AND CONFIGURATION
if verbose
    fprintf('[SECTION 1] Setup and Configuration\n');
end

% Define all available files from config
all_files = config.files.all_files;
training_files = config.files.training_files;
reembedding_files = config.files.reembedding_files;

if verbose
    fprintf('Total files: %d\n', length(all_files));
    fprintf('Training files: %d\n', length(training_files));
    fprintf('Re-embedding files: %d\n', length(reembedding_files));
end

% Create metadata for file organization
metadata = create_file_metadata(all_files);

%% SECTION 2: TRAINING PHASE (conditional)
if ~skip_training
    if verbose
        fprintf('\n[SECTION 2] Training Phase\n');
    end
    
    % Load and prepare training data
    [training_data, training_labels] = load_training_data(data_dir, training_files, verbose);
    
    % Set up skeleton and parameters
    [skeleton, parameters] = setup_skeleton_and_parameters(config);
    
    % Extract features and perform PCA
    [mus, vecs, vals] = perform_pca_training(training_data, training_labels, parameters, verbose);
    
    % Create training embedding with ComBat correction
    [signalData, signalAmps, Y_training] = create_training_embedding_combat(...
        training_data, training_labels, mus, vecs, parameters, verbose);
    
    % Create watershed regions
    [D, LL, LL2, llbwb, xx] = create_watershed_regions(Y_training, config.watershed, verbose);
    
    % Save training results
    save_training_results(output_dir, mus, vecs, vals, signalData, signalAmps, ...
        Y_training, D, LL, LL2, llbwb, xx, parameters);
    
else
    % Load existing training results
    if verbose
        fprintf('\n[SECTION 2] Loading Existing Training Results\n');
    end
    [mus, vecs, Y_training, D, LL, LL2, llbwb, xx, parameters, skeleton] = ...
        load_training_results(output_dir, config);
end

%% SECTION 3: RE-EMBEDDING PHASE
if verbose
    fprintf('\n[SECTION 3] Re-embedding Phase\n');
end

% Load all files for re-embedding
[reembedding_data, reembedding_labels, reembedding_metadata] = ...
    load_reembedding_data(data_dir, reembedding_files, verbose);

% Re-embed all files onto trained space
[zEmbeddings_all, wrFINE_all] = reembed_all_files(...
    reembedding_data, reembedding_labels, mus, vecs, Y_training, ...
    xx, LL, parameters, verbose);

%% SECTION 4: ANALYSIS AND VISUALIZATION
if verbose
    fprintf('\n[SECTION 4] Analysis and Visualization\n');
end

% Create comprehensive results structure
results = create_results_structure(training_files, reembedding_files, ...
    reembedding_labels, reembedding_metadata, zEmbeddings_all, wrFINE_all, ...
    Y_training, D, LL, LL2, llbwb, parameters, skeleton);

% Save complete results
save(fullfile(output_dir, 'complete_embedding_results_SNI_TBI.mat'), 'results', '-v7.3');

% Create visualizations and analysis
create_comprehensive_analysis(results, config.colors, output_dir, verbose);

%% Display summary
if verbose
    display_pipeline_summary(results, training_files, reembedding_files);
end

if verbose
    fprintf('\nPipeline completed successfully!\n');
    fprintf('Results saved to: %s\n', fullfile(output_dir, 'complete_embedding_results_SNI_TBI.mat'));
end

end

%% HELPER FUNCTION STUBS (to be implemented in separate files)

function config = load_pipeline_config()
% Load default pipeline configuration
config = [];
% This will load from config/pipeline_config.m
end

function metadata = create_file_metadata(file_list)
% Create metadata structure for organizing files
metadata = struct();
metadata.all_files = file_list;
metadata.groups = {};
metadata.weeks = {};

for i = 1:length(file_list)
    [group, week] = extract_metadata_from_filename(file_list{i});
    metadata.groups{i} = group;
    metadata.weeks{i} = week;
end
end

function [group, week] = extract_metadata_from_filename(filename)
% Extract group and week information from filename

% Remove .mat extension
name = filename(1:end-4);

% Check if it's a week4 file
if startsWith(name, 'week4-')
    week = 'week4';
    name = name(7:end); % Remove 'week4-' prefix
else
    week = 'week1';
end

% Extract group
if startsWith(name, 'DRG')
    group = 'DRG';
elseif startsWith(name, 'IT')
    group = 'IT';
elseif startsWith(name, 'SC')
    group = 'SC';
elseif startsWith(name, 'SNI')
    group = 'SNI';
elseif startsWith(name, 'TBI')
    group = 'TBI';
else
    group = 'unknown';
end
end