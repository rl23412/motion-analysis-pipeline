function results = run_pipeline(varargin)
% RUN_PIPELINE - Main script to execute the ComBat SNI_TBI embedding pipeline
%
% This is the primary entry point for running the ComBat batch correction
% behavioral embedding pipeline based on custom_embedding_pipeline_SNI_TBI.m
%
% Usage:
%   results = run_pipeline()                    % Run with default settings
%   results = run_pipeline('data_dir', path)   % Specify data directory
%
% Optional Parameters:
%   'data_dir'      - Path to directory containing .mat files
%   'verbose'       - Display progress messages (default: true)
%
% Returns:
%   results - Complete pipeline results structure
%
% Example:
%   % Basic usage
%   results = run_pipeline();
%   
%   % Custom data directory
%   results = run_pipeline('data_dir', '/work/rl349/dannce/mouse14/allData');

%% Setup and initialization
fprintf('=== ComBat SNI_TBI Motion Analysis Pipeline ===\n');
fprintf('Mouse14 format (14 joints) with ComBat batch correction\n');
fprintf('Initializing pipeline...\n\n');

% Add paths to dependencies
setup_paths();

% Validate installation
validate_installation();

%% Parse input arguments
p = inputParser;
addParameter(p, 'data_dir', '/work/rl349/dannce/mouse14/allData', @(x) ischar(x) || isstring(x));
addParameter(p, 'verbose', true, @islogical);
parse(p, varargin{:});

%% Load configuration
config = pipeline_config();
fprintf('Loaded ComBat pipeline configuration (mouse14 format)\n');

% Override config with command line arguments
if ~isempty(p.Results.data_dir)
    config.data_dir = p.Results.data_dir;
end

%% Validate data directory and files
fprintf('Validating data directory: %s\n', config.data_dir);
validate_data_directory(config);

%% Run the ComBat pipeline
fprintf('\nStarting ComBat pipeline execution...\n');
try
    results = custom_embedding_pipeline(...
        'data_dir', config.data_dir, ...
        'config', config, ...
        'verbose', p.Results.verbose);
    
    fprintf('\n=== COMBAT PIPELINE COMPLETED SUCCESSFULLY ===\n');
    
    % Display summary
    display_completion_summary(results);
    
catch ME
    fprintf('\n=== PIPELINE FAILED ===\n');
    fprintf('Error: %s\n', ME.message);
    fprintf('Stack trace:\n');
    for i = 1:length(ME.stack)
        fprintf('  %s (line %d)\n', ME.stack(i).file, ME.stack(i).line);
    end
    rethrow(ME);
end

end

%% Helper Functions

function setup_paths()
% Add all necessary paths for the ComBat pipeline

% Get the directory of this script
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);

% Add source directories
addpath(genpath(fullfile(project_root, 'src')));
addpath(genpath(fullfile(project_root, 'config')));

% Add SocialMapper dependencies (essential for ComBat pipeline)
dependencies_dir = fullfile(project_root, 'dependencies');
addpath(genpath(fullfile(dependencies_dir, 'SocialMapper')));

% Add Combat if available
combat_dir = fullfile(dependencies_dir, 'Combat');
if exist(combat_dir, 'dir')
    addpath(genpath(combat_dir));
end

fprintf('Added ComBat pipeline directories to MATLAB path\n');
end

function validate_installation()
% Validate that all required components are available for ComBat pipeline

fprintf('Validating ComBat pipeline installation...\n');

% Check required functions from SocialMapper (essential for ComBat pipeline)
required_functions = {
    'findWavelets'
    'findTemplatesFromData'  
    'findTDistributedProjections_fmin'
    'findWatershedRegions_v2'
    'setRunParameters'
    'combineCells'
};

missing_functions = {};
for i = 1:length(required_functions)
    if exist(required_functions{i}, 'file') ~= 2
        missing_functions{end+1} = required_functions{i}; %#ok<AGROW>
    end
end

if ~isempty(missing_functions)
    error('Missing required functions for ComBat pipeline: %s', strjoin(missing_functions, ', '));
end

% Check required toolboxes
required_toolboxes = {
    'Statistics and Machine Learning Toolbox'
    'Image Processing Toolbox'
};

for i = 1:length(required_toolboxes)
    if ~license('test', strrep(required_toolboxes{i}, ' ', '_'))
        warning('Required toolbox not available: %s', required_toolboxes{i});
    end
end

% Check for tsne function specifically
if exist('tsne', 'file') ~= 2
    error('tsne function not found. Install Statistics and Machine Learning Toolbox');
end

% Check for ComBat
if exist('combat', 'file') ~= 2
    fprintf('  ⚠ ComBat function not found - using simplified batch correction\n');
else
    fprintf('  ✓ ComBat batch correction available\n');
end

fprintf('ComBat pipeline installation validation completed\n');
end

function validate_data_directory(config)
% Validate that the data directory exists and contains expected files

if ~exist(config.data_dir, 'dir')
    error('Data directory does not exist: %s', config.data_dir);
end

% Check for at least some of the expected files
files_found = 0;
for i = 1:length(config.files.all_files)
    file_path = fullfile(config.data_dir, config.files.all_files{i});
    if exist(file_path, 'file')
        files_found = files_found + 1;
    end
end

if files_found == 0
    error('No expected data files found in: %s', config.data_dir);
end

fprintf('Found %d/%d expected data files\n', files_found, length(config.files.all_files));
end

function display_completion_summary(results)
% Display a summary of ComBat pipeline completion

fprintf('\nCOMBAT PIPELINE SUMMARY:\n');
fprintf('-' * ones(1, 50), '\n');
fprintf('Training files: %d\n', length(results.training_files));
fprintf('Skeleton format: mouse14 (14 joints)\n');
fprintf('ComBat batch correction: Applied\n');
fprintf('PCA components: %d\n', results.nPCA);
fprintf('Watershed regions: %d\n', max(results.LL(:)));

fprintf('\nKey outputs:\n');
fprintf('  - Complete results: complete_embedding_results_SNI_TBI.mat\n');
fprintf('  - Training results: train_SNI_TBI.mat\n');
fprintf('  - Watershed regions: watershed_SNI_TBI.mat\n');
fprintf('  - PCA results: vecsMus_SNI_TBI_training.mat\n');

fprintf('\nComBat SNI_TBI pipeline completed successfully!\n');
end