function results = run_pipeline(varargin)
% RUN_PIPELINE - Main script to execute the motion analysis pipeline
%
% This is the primary entry point for running the complete behavioral
% embedding pipeline with ComBat batch correction and comprehensive analysis
%
% Usage:
%   results = run_pipeline()                    % Run with default settings
%   results = run_pipeline('data_dir', path)   % Specify data directory
%   results = run_pipeline('config', config)   % Use custom configuration
%   results = run_pipeline('skip_training', true)  % Skip training phase
%
% Optional Parameters:
%   'data_dir'      - Path to directory containing .mat files
%   'config'        - Custom configuration structure
%   'output_dir'    - Output directory (default: 'outputs')
%   'skip_training' - Skip training and use existing results (default: false)
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
%   results = run_pipeline('data_dir', '/path/to/my/data');
%   
%   % Skip training (use existing model)
%   results = run_pipeline('skip_training', true);
%
% Dependencies:
%   - All files in src/ directory must be on MATLAB path
%   - SocialMapper utilities in dependencies/SocialMapper
%   - ComBat batch correction (if available)
%   - Required MATLAB toolboxes (see pipeline_config.m)

%% Setup and initialization
fprintf('=== MOTION ANALYSIS PIPELINE ===\n');
fprintf('Initializing pipeline...\n\n');

% Add paths to dependencies
setup_paths();

% Validate installation
validate_installation();

%% Parse input arguments
p = inputParser;
addParameter(p, 'data_dir', [], @(x) ischar(x) || isstring(x));
addParameter(p, 'config', [], @isstruct);
addParameter(p, 'output_dir', 'outputs', @(x) ischar(x) || isstring(x));
addParameter(p, 'skip_training', false, @islogical);
addParameter(p, 'verbose', true, @islogical);
parse(p, varargin{:});

%% Load configuration
if isempty(p.Results.config)
    config = pipeline_config();
    fprintf('Loaded default configuration\n');
else
    config = p.Results.config;
    fprintf('Using provided configuration\n');
end

% Override config with command line arguments
if ~isempty(p.Results.data_dir)
    config.data_dir = p.Results.data_dir;
end

%% Validate data directory and files
fprintf('Validating data directory: %s\n', config.data_dir);
validate_data_directory(config);

%% Create output directory
output_dir = p.Results.output_dir;
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
    fprintf('Created output directory: %s\n', output_dir);
end

%% Run the pipeline
fprintf('\nStarting pipeline execution...\n');
try
    results = custom_embedding_pipeline(...
        'data_dir', config.data_dir, ...
        'config', config, ...
        'output_dir', output_dir, ...
        'skip_training', p.Results.skip_training, ...
        'verbose', p.Results.verbose);
    
    fprintf('\n=== PIPELINE COMPLETED SUCCESSFULLY ===\n');
    
    % Display summary
    display_completion_summary(results, output_dir);
    
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
% Add all necessary paths for the pipeline

% Get the directory of this script
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);

% Add source directories
addpath(genpath(fullfile(project_root, 'src')));
addpath(genpath(fullfile(project_root, 'config')));

% Add dependencies
dependencies_dir = fullfile(project_root, 'dependencies');
addpath(genpath(fullfile(dependencies_dir, 'SocialMapper')));

% Add Combat if available
combat_dir = fullfile(dependencies_dir, 'Combat');
if exist(combat_dir, 'dir')
    addpath(genpath(combat_dir));
end

fprintf('Added pipeline directories to MATLAB path\n');
end

function validate_installation()
% Validate that all required components are available

fprintf('Validating installation...\n');

% Check required functions from SocialMapper
required_functions = {
    'findWavelets'
    'findTemplatesFromData'  
    'findTDistributedProjections_fmin'
    'findWatershedRegions_v2'
    'setRunParameters'
};

missing_functions = {};
for i = 1:length(required_functions)
    if exist(required_functions{i}, 'file') ~= 2
        missing_functions{end+1} = required_functions{i}; %#ok<AGROW>
    end
end

if ~isempty(missing_functions)
    error('Missing required functions: %s', strjoin(missing_functions, ', '));
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

fprintf('Installation validation completed\n');
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

function display_completion_summary(results, output_dir)
% Display a summary of pipeline completion

fprintf('\nPIPELINE SUMMARY:\n');
fprintf('-' * ones(1, 50), '\n');
fprintf('Files processed: %d\n', length(results.reembedding_labels_all));
fprintf('Training files: %d\n', length(results.training_files));
fprintf('Watershed regions: %d\n', max(results.LL(:)));
fprintf('Output directory: %s\n', output_dir);

% Count files by group
group_counts = struct();
for i = 1:length(results.reembedding_metadata_all)
    if ~isempty(results.reembedding_metadata_all{i})
        group = results.reembedding_metadata_all{i}.group;
        field_name = matlab.lang.makeValidName(group);
        if isfield(group_counts, field_name)
            group_counts.(field_name) = group_counts.(field_name) + 1;
        else
            group_counts.(field_name) = 1;
        end
    end
end

fprintf('\nGroup distribution:\n');
group_names = fieldnames(group_counts);
for i = 1:length(group_names)
    fprintf('  %s: %d files\n', strrep(group_names{i}, '_', '-'), group_counts.(group_names{i}));
end

fprintf('\nKey outputs:\n');
fprintf('  - Complete results: complete_embedding_results_SNI_TBI.mat\n');
fprintf('  - Training results: train_SNI_TBI.mat\n');
fprintf('  - Watershed regions: watershed_SNI_TBI.mat\n');
fprintf('  - PCA results: vecsMus_SNI_TBI_training.mat\n');

fprintf('\nAll visualizations and analysis files have been generated.\n');
fprintf('Pipeline completed successfully!\n');
end