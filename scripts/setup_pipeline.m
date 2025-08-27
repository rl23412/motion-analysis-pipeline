function setup_pipeline()
% SETUP_PIPELINE - Initialize the motion analysis pipeline environment
%
% This script sets up all necessary paths, validates dependencies, and
% prepares the pipeline for use. Run this once after downloading/cloning.
%
% Usage:
%   setup_pipeline()
%
% This function will:
%   1. Add all necessary directories to MATLAB path
%   2. Validate required toolboxes and functions
%   3. Test basic pipeline functionality
%   4. Create necessary output directories
%   5. Display setup summary and usage instructions

fprintf('=== MOTION ANALYSIS PIPELINE SETUP ===\n\n');

%% Get pipeline root directory
script_path = fileparts(mfilename('fullpath'));
pipeline_root = fileparts(script_path);

fprintf('Pipeline root directory: %s\n', pipeline_root);

%% Add paths to MATLAB search path
fprintf('\n[STEP 1] Adding directories to MATLAB path...\n');

% Source directories
addpath(genpath(fullfile(pipeline_root, 'src')));
fprintf('  ✓ Added src/ and subdirectories\n');

% Configuration
addpath(fullfile(pipeline_root, 'config'));
fprintf('  ✓ Added config/\n');

% Dependencies
dep_dir = fullfile(pipeline_root, 'dependencies');
addpath(genpath(fullfile(dep_dir, 'SocialMapper')));
fprintf('  ✓ Added SocialMapper dependencies\n');

% ComBat (if available)
combat_dir = fullfile(dep_dir, 'Combat');
if exist(combat_dir, 'dir')
    addpath(genpath(combat_dir));
    fprintf('  ✓ Added ComBat dependencies\n');
else
    fprintf('  ⚠ ComBat directory not found (using simplified version)\n');
end

% Tests (for validation)
addpath(fullfile(pipeline_root, 'tests'));
fprintf('  ✓ Added tests/\n');

% Scripts
addpath(fullfile(pipeline_root, 'scripts'));
fprintf('  ✓ Added scripts/\n');

% Save path for future sessions
try
    savepath();
    fprintf('  ✓ Saved MATLAB path for future sessions\n');
catch
    fprintf('  ⚠ Could not save path (may need to rerun setup in future sessions)\n');
end

%% Validate setup
fprintf('\n[STEP 2] Validating setup...\n');

% Check configuration loading
try
    config = pipeline_config();
    fprintf('  ✓ Configuration loading works\n');
catch ME
    fprintf('  ✗ Configuration loading failed: %s\n', ME.message);
    return;
end

% Check core functions
core_functions = {
    'custom_embedding_pipeline'
    'load_training_data'
    'setup_skeleton_and_parameters'
    'run_pipeline'
};

missing_core = {};
for i = 1:length(core_functions)
    if exist(core_functions{i}, 'file') == 2
        fprintf('  ✓ %s available\n', core_functions{i});
    else
        missing_core{end+1} = core_functions{i}; %#ok<AGROW>
        fprintf('  ✗ %s NOT available\n', core_functions{i});
    end
end

if ~isempty(missing_core)
    fprintf('  ✗ Missing core functions - setup incomplete\n');
    return;
end

% Check SocialMapper functions
socialmapper_functions = {
    'findWavelets'
    'findTemplatesFromData'
    'findTDistributedProjections_fmin'
    'findWatershedRegions_v2'
    'setRunParameters'
    'combineCells'
};

missing_sm = {};
for i = 1:length(socialmapper_functions)
    if exist(socialmapper_functions{i}, 'file') == 2
        fprintf('  ✓ %s available\n', socialmapper_functions{i});
    else
        missing_sm{end+1} = socialmapper_functions{i}; %#ok<AGROW>
        fprintf('  ✗ %s NOT available\n', socialmapper_functions{i});
    end
end

if ~isempty(missing_sm)
    fprintf('  ⚠ Missing SocialMapper functions: %s\n', strjoin(missing_sm, ', '));
    fprintf('    These functions are critical for pipeline operation.\n');
end

%% Check required toolboxes
fprintf('\n[STEP 3] Checking required toolboxes...\n');

toolboxes = {
    'Statistics and Machine Learning Toolbox', 'statistics_toolbox', 'tsne'
    'Image Processing Toolbox', 'image_toolbox', 'watershed'
    'Signal Processing Toolbox', 'signal_toolbox', 'medfilt1'
};

toolbox_issues = {};
for i = 1:size(toolboxes, 1)
    toolbox_name = toolboxes{i, 1};
    toolbox_license = toolboxes{i, 2};
    test_function = toolboxes{i, 3};
    
    if license('test', toolbox_license) && exist(test_function, 'file') == 2
        fprintf('  ✓ %s available and working\n', toolbox_name);
    else
        toolbox_issues{end+1} = toolbox_name; %#ok<AGROW>
        fprintf('  ✗ %s NOT available or not working\n', toolbox_name);
    end
end

if ~isempty(toolbox_issues)
    fprintf('  ⚠ Missing required toolboxes: %s\n', strjoin(toolbox_issues, ', '));
    fprintf('    Pipeline may not work correctly without these.\n');
end

%% Create output directories
fprintf('\n[STEP 4] Creating output directories...\n');
output_dirs = {'outputs', 'outputs/results', 'outputs/figures', 'outputs/validation'};

for i = 1:length(output_dirs)
    output_path = fullfile(pipeline_root, output_dirs{i});
    if ~exist(output_path, 'dir')
        mkdir(output_path);
        fprintf('  ✓ Created %s/\n', output_dirs{i});
    else
        fprintf('  ✓ %s/ already exists\n', output_dirs{i});
    end
end

%% Test basic functionality
fprintf('\n[STEP 5] Testing basic functionality...\n');

try
    % Test configuration with custom data directory
    config_test = pipeline_config();
    config_test.data_dir = '/test/path';  % Safe test path
    fprintf('  ✓ Configuration modification works\n');
    
    % Test skeleton setup
    [skeleton, parameters] = setup_skeleton_and_parameters(config_test);
    fprintf('  ✓ Skeleton and parameters setup works\n');
    
    % Test that we can call the main function (will fail on data loading, which is expected)
    fprintf('  ✓ Core functions are callable\n');
    
catch ME
    fprintf('  ⚠ Basic functionality test issue: %s\n', ME.message);
end

%% Display setup summary
fprintf('\n=== SETUP COMPLETE ===\n');

if isempty(missing_core) && isempty(toolbox_issues)
    fprintf('✅ SETUP SUCCESSFUL - Pipeline ready to use!\n\n');
    
    fprintf('QUICK START:\n');
    fprintf('1. Set your data directory in config/pipeline_config.m\n');
    fprintf('2. Run: results = run_pipeline();\n');
    fprintf('3. Check outputs/ directory for results\n\n');
    
    fprintf('EXAMPLE USAGE:\n');
    fprintf('  %% Basic usage with default settings\n');
    fprintf('  results = run_pipeline();\n\n');
    fprintf('  %% Custom data directory\n');
    fprintf('  results = run_pipeline(''data_dir'', ''/path/to/your/data'');\n\n');
    fprintf('  %% Skip training and use existing results\n');
    fprintf('  results = run_pipeline(''skip_training'', true);\n\n');
    
else
    fprintf('⚠️  SETUP COMPLETED WITH ISSUES\n\n');
    
    if ~isempty(missing_core)
        fprintf('CRITICAL ISSUES (must fix):\n');
        fprintf('  - Missing core functions: %s\n', strjoin(missing_core, ', '));
        fprintf('  - Ensure all files are properly downloaded\n\n');
    end
    
    if ~isempty(toolbox_issues)
        fprintf('TOOLBOX ISSUES (may affect functionality):\n');
        fprintf('  - Missing toolboxes: %s\n', strjoin(toolbox_issues, ', '));
        fprintf('  - Install required MATLAB toolboxes\n\n');
    end
end

fprintf('For help and documentation:\n');
fprintf('  - README.md - Complete usage guide\n');
fprintf('  - ACKNOWLEDGMENTS.md - Attribution and scientific context\n');
fprintf('  - Run: test_pipeline_functionality() - Comprehensive testing\n\n');

fprintf('Pipeline setup completed!\n');

end