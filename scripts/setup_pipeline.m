function setup_results = setup_pipeline()
%% SETUP_PIPELINE Initialize environment for spontaneous pain analysis pipeline
% This script sets up the complete pipeline environment by:
% 1. Adding all necessary paths to MATLAB
% 2. Checking for required dependencies and toolboxes
% 3. Validating MEX files for the current platform
% 4. Creating output directories
% 5. Performing comprehensive system validation
%
% Usage:
%   setup_pipeline()                    % Run setup with default settings
%   results = setup_pipeline()          % Return setup results structure
%
% Outputs:
%   setup_results - Structure containing setup status and validation results

fprintf('\n=== Spontaneous Pain Analysis Pipeline Setup ===\n');
fprintf('Version: 1.4.0\n');
fprintf('Started: %s\n\n', datestr(now));

%% Initialize results structure
setup_results = struct();
setup_results.timestamp = datestr(now);
setup_results.success = true;
setup_results.issues = {};
setup_results.warnings = {};

%% Get current directory and setup paths
currentDir = pwd;
fprintf('Pipeline directory: %s\n', currentDir);

% Define all required paths
pathsToAdd = {
    currentDir,
    fullfile(currentDir, 'src'),
    fullfile(currentDir, 'src', 'core'),
    fullfile(currentDir, 'src', 'utils'),
    fullfile(currentDir, 'src', 'analysis'),
    fullfile(currentDir, 'config'),
    fullfile(currentDir, 'scripts'),
    fullfile(currentDir, 'dependencies'),
    fullfile(currentDir, 'dependencies', 'MotionMapper'),
    fullfile(currentDir, 'dependencies', 'MotionMapper', 'MotionMapperUtilities'),
    fullfile(currentDir, 'dependencies', 'MotionMapper', 'MotionMapperUtilities', 'tSNE'),
    fullfile(currentDir, 'dependencies', 'MotionMapper', 'MotionMapperUtilities', 'utilities'),
    fullfile(currentDir, 'dependencies', 'MotionMapper', 'MotionMapperUtilities', 'metrics'),
    fullfile(currentDir, 'dependencies', 'SocialMapper'),
    fullfile(currentDir, 'dependencies', 'SocialMapper', 'MotionMapperUtilities'),
    fullfile(currentDir, 'dependencies', 'SocialMapper', 'MotionMapperUtilities', 'tSNE'),
    fullfile(currentDir, 'dependencies', 'SocialMapper', 'MotionMapperUtilities', 'utilities'),
    fullfile(currentDir, 'dependencies', 'SocialMapper', 'MotionMapperUtilities', 'metrics'),
    fullfile(currentDir, 'dependencies', 'SocialMapper', 'utilities'),
    fullfile(currentDir, 'dependencies', 'SocialMapper', 'mouseEmbedding'),
    fullfile(currentDir, 'dependencies', 'Animator'),
    fullfile(currentDir, 'tests')
};

% Add paths to MATLAB
fprintf('Adding paths to MATLAB:\n');
addedPaths = {};
missingPaths = {};

for i = 1:length(pathsToAdd)
    if exist(pathsToAdd{i}, 'dir')
        addpath(pathsToAdd{i});
        addedPaths{end+1} = pathsToAdd{i};
        fprintf('  ✓ %s\n', relativePath(pathsToAdd{i}, currentDir));
    else
        missingPaths{end+1} = pathsToAdd{i};
        fprintf('  ✗ %s (not found)\n', relativePath(pathsToAdd{i}, currentDir));
    end
end

setup_results.paths = struct();
setup_results.paths.added = addedPaths;
setup_results.paths.missing = missingPaths;
setup_results.paths.total_added = length(addedPaths);
setup_results.paths.total_missing = length(missingPaths);

if ~isempty(missingPaths)
    setup_results.warnings{end+1} = sprintf('%d dependency paths not found', length(missingPaths));
end

fprintf('\n');

%% Create output directory structure
fprintf('Creating output directory structure:\n');

outputDirs = {
    'outputs',
    'outputs/results',
    'outputs/figures', 
    'outputs/figures/per_video',
    'outputs/figures/group_comparisons',
    'outputs/analysis',
    'outputs/analysis/figures',
    'outputs/analysis/figures/per_video',
    'outputs/analysis/csv',
    'outputs/analysis/csv/frame_indices_per_video',
    'outputs/analysis/csv/sequence_metadata',
    'outputs/analysis/csv/statistical_analysis',
    'outputs/validation',
    'data',
    'data/raw',
    'data/processed',
    'cache'
};

createdDirs = {};
existingDirs = {};

for i = 1:length(outputDirs)
    dirPath = fullfile(currentDir, outputDirs{i});
    if ~exist(dirPath, 'dir')
        try
            mkdir(dirPath);
            createdDirs{end+1} = outputDirs{i};
            fprintf('  ✓ Created: %s\n', outputDirs{i});
        catch ME
            setup_results.issues{end+1} = sprintf('Could not create directory %s: %s', outputDirs{i}, ME.message);
            setup_results.success = false;
            fprintf('  ✗ Failed to create: %s\n', outputDirs{i});
        end
    else
        existingDirs{end+1} = outputDirs{i};
        fprintf('  ✓ Exists: %s\n', outputDirs{i});
    end
end

% Create .gitkeep files for important empty directories
gitkeepDirs = {'outputs/results', 'outputs/figures', 'outputs/validation', 'data/raw', 'data/processed'};
for i = 1:length(gitkeepDirs)
    gitkeepFile = fullfile(currentDir, gitkeepDirs{i}, '.gitkeep');
    if ~exist(gitkeepFile, 'file')
        fid = fopen(gitkeepFile, 'w');
        if fid ~= -1
            fclose(fid);
        end
    end
end

setup_results.directories = struct();
setup_results.directories.created = createdDirs;
setup_results.directories.existing = existingDirs;
setup_results.directories.total_created = length(createdDirs);

fprintf('\n');

%% Check MATLAB toolbox dependencies
fprintf('Checking MATLAB toolbox dependencies:\n');

% Define required and optional toolboxes
toolboxChecks = {
    'Statistics and Machine Learning Toolbox', @check_stats_toolbox, true;
    'Image Processing Toolbox', @check_image_toolbox, true;
    'Signal Processing Toolbox', @check_signal_toolbox, false;
    'Parallel Computing Toolbox', @check_parallel_toolbox, false;
    'Curve Fitting Toolbox', @check_curve_fitting_toolbox, false
};

toolboxResults = struct();
allCriticalAvailable = true;

for i = 1:size(toolboxChecks, 1)
    toolboxName = toolboxChecks{i,1};
    checkFunction = toolboxChecks{i,2};
    isCritical = toolboxChecks{i,3};
    
    [hasToolbox, details] = checkFunction();
    fieldName = matlab.lang.makeValidName(toolboxName);
    
    toolboxResults.(fieldName) = struct();
    toolboxResults.(fieldName).available = hasToolbox;
    toolboxResults.(fieldName).critical = isCritical;
    toolboxResults.(fieldName).details = details;
    
    if hasToolbox
        fprintf('  ✓ %s\n', toolboxName);
        if ~isempty(details)
            fprintf('    %s\n', details);
        end
    else
        if isCritical
            fprintf('  ✗ %s - REQUIRED\n', toolboxName);
            setup_results.issues{end+1} = sprintf('Missing critical toolbox: %s', toolboxName);
            allCriticalAvailable = false;
            setup_results.success = false;
        else
            fprintf('  ! %s - Optional\n', toolboxName);
            setup_results.warnings{end+1} = sprintf('Missing optional toolbox: %s', toolboxName);
        end
        if ~isempty(details)
            fprintf('    %s\n', details);
        end
    end
end

setup_results.toolboxes = toolboxResults;
setup_results.all_critical_toolboxes = allCriticalAvailable;

fprintf('\n');

%% Check custom function dependencies
fprintf('Checking custom function dependencies:\n');

% MotionMapper functions (critical)
motionMapperFunctions = {
    'findWavelets', true, 'Core wavelet analysis';
    'findTemplatesFromData', true, 'Template extraction';
    'findTDistributedProjections_fmin', true, 'Re-embedding algorithm';
    'findWatershedRegions_v2', true, 'Watershed region analysis';
    'findPointDensity', true, 'Density estimation';
    'setRunParameters', true, 'Parameter configuration';
    'returnDist3d', true, '3D distance calculation';
    'combineCells', true, 'Cell array utilities';
    'tsne', false, 't-SNE embedding (may use toolbox version)';
    'Keypoint3DAnimator', false, '3D visualization'
};

functionResults = struct();
allCriticalFunctionsAvailable = true;

for i = 1:size(motionMapperFunctions, 1)
    funcName = motionMapperFunctions{i,1};
    isCritical = motionMapperFunctions{i,2};
    description = motionMapperFunctions{i,3};
    
    fieldName = matlab.lang.makeValidName(funcName);
    functionResults.(fieldName) = struct();
    
    if exist(funcName, 'file')
        funcPath = which(funcName);
        functionResults.(fieldName).available = true;
        functionResults.(fieldName).path = funcPath;
        
        fprintf('  ✓ %s - %s\n', funcName, description);
        
        % Show path for custom functions (not built-in)
        if ~contains(funcPath, matlabroot)
            fprintf('    → %s\n', relativePath(funcPath, currentDir));
        end
    else
        functionResults.(fieldName).available = false;
        functionResults.(fieldName).path = '';
        
        if isCritical
            fprintf('  ✗ %s - CRITICAL MISSING\n', funcName);
            setup_results.issues{end+1} = sprintf('Missing critical function: %s (%s)', funcName, description);
            allCriticalFunctionsAvailable = false;
            setup_results.success = false;
        else
            fprintf('  ! %s - Optional\n', funcName);
            setup_results.warnings{end+1} = sprintf('Missing optional function: %s (%s)', funcName, description);
        end
    end
    
    functionResults.(fieldName).critical = isCritical;
    functionResults.(fieldName).description = description;
end

setup_results.functions = functionResults;
setup_results.all_critical_functions = allCriticalFunctionsAvailable;

fprintf('\n');

%% Check and validate MEX files
fprintf('Checking MEX files for current platform:\n');

mexFiles = {
    'findListDistances', 'Fast distance computation';
    'ba_interp2', 'Bilinear interpolation'
};

computer_arch = computer('arch');
fprintf('Platform: %s (MEX extension: %s)\n', computer_arch, mexext);

mexResults = struct();
for i = 1:length(mexFiles)
    mexName = mexFiles{i,1};
    mexDescription = mexFiles{i,2};
    fieldName = matlab.lang.makeValidName(mexName);
    
    mexResults.(fieldName) = struct();
    mexResults.(fieldName).description = mexDescription;
    
    % Look for MEX file with correct extension
    mexPath = which([mexName '.' mexext]);
    
    if ~isempty(mexPath)
        mexResults.(fieldName).available = true;
        mexResults.(fieldName).path = mexPath;
        
        fprintf('  ✓ %s.%s - %s\n', mexName, mexext, mexDescription);
        fprintf('    → %s\n', relativePath(mexPath, currentDir));
        
        # Test MEX file functionality
        try
            if strcmp(mexName, 'findListDistances')
                % Test with small dummy data
                test_result = feval(mexName, rand(5,3), rand(3,3));
                fprintf('    → MEX test: PASSED\n');
                mexResults.(fieldName).test_passed = true;
            else
                fprintf('    → MEX test: SKIPPED\n');
                mexResults.(fieldName).test_passed = [];
            end
        catch ME
            fprintf('    → MEX test: FAILED (%s)\n', ME.message);
            mexResults.(fieldName).test_passed = false;
            setup_results.warnings{end+1} = sprintf('MEX file %s failed test: %s', mexName, ME.message);
        end
    else
        mexResults.(fieldName).available = false;
        mexResults.(fieldName).path = '';
        mexResults.(fieldName).test_passed = false;
        
        fprintf('  ✗ %s.%s - Missing\n', mexName, mexext);
        setup_results.warnings{end+1} = sprintf('MEX file missing: %s (fallback functions may be used)', mexName);
    end
end

setup_results.mex_files = mexResults;
setup_results.platform = computer_arch;

fprintf('\n');

%% System information and performance checks
fprintf('System information:\n');

% MATLAB version
matlab_version = version;
matlab_release = version('-release');
fprintf('  MATLAB Version: %s (%s)\n', matlab_version, matlab_release);

# Memory information
if ispc
    [~, memInfo] = memory;
    if isfield(memInfo, 'MemAvailableAllArrays')
        available_memory_gb = memInfo.MemAvailableAllArrays / 1e9;
        fprintf('  Available Memory: %.1f GB\n', available_memory_gb);
        
        if available_memory_gb < 8
            setup_results.warnings{end+1} = 'Low available memory (< 8 GB) - may affect performance';
        elseif available_memory_gb < 4
            setup_results.issues{end+1} = 'Very low available memory (< 4 GB) - pipeline may fail';
            setup_results.success = false;
        end
    end
end

% CPU information
if ispc
    [~, cpu_info] = system('wmic cpu get name');
    cpu_lines = strsplit(cpu_info, '\n');
    if length(cpu_lines) > 1
        fprintf('  CPU: %s\n', strtrim(cpu_lines{2}));
    end
end

% Disk space check
try
    current_drive = currentDir(1:2); % Get drive letter (Windows)
    [~, disk_info] = system(['dir /-c ' current_drive '\']);
    % Parse disk info if needed
catch
    % Fallback or skip disk check
end

setup_results.system = struct();
setup_results.system.matlab_version = matlab_version;
setup_results.system.matlab_release = matlab_release;
setup_results.system.platform = computer_arch;

fprintf('\n');

%% Validate input data (if present)
fprintf('Checking for input data:\n');

dataFiles = {
    'data/raw/mouseFileOrder.mat', 'Mouse file order and metadata', true;
    'data/raw/SNI_2.mat', 'Training data 1 (SNI)', false;
    'data/raw/week4-TBI_3.mat', 'Training data 2 (TBI)', false
};

dataResults = struct();
criticalDataPresent = true;

for i = 1:size(dataFiles, 1)
    filename = dataFiles{i,1};
    description = dataFiles{i,2};
    isRequired = dataFiles{i,3};
    
    fieldName = matlab.lang.makeValidName(filename);
    filePath = fullfile(currentDir, filename);
    
    dataResults.(fieldName) = struct();
    dataResults.(fieldName).description = description;
    dataResults.(fieldName).required = isRequired;
    
    if exist(filePath, 'file')
        dataResults.(fieldName).available = true;
        dataResults.(fieldName).path = filePath;
        
        fprintf('  ✓ %s\n', filename);
        fprintf('    %s\n', description);
        
        # Validate structure for key files
        if strcmp(filename, 'data/raw/mouseFileOrder.mat')
            try
                data = load(filePath);
                if isfield(data, 'mouseOrderShort') && isfield(data, 'metadata')
                    fprintf('    → Structure validation: PASSED\n');
                    dataResults.(fieldName).structure_valid = true;
                else
                    fprintf('    → Structure validation: FAILED (missing expected fields)\n');
                    dataResults.(fieldName).structure_valid = false;
                    setup_results.warnings{end+1} = 'mouseFileOrder.mat structure may be invalid';
                end
            catch ME
                fprintf('    → Structure validation: ERROR (%s)\n', ME.message);
                dataResults.(fieldName).structure_valid = false;
                setup_results.warnings{end+1} = sprintf('Could not validate mouseFileOrder.mat: %s', ME.message);
            end
        else
            dataResults.(fieldName).structure_valid = [];
        end
    else
        dataResults.(fieldName).available = false;
        dataResults.(fieldName).path = '';
        dataResults.(fieldName).structure_valid = false;
        
        if isRequired
            fprintf('  ✗ %s - REQUIRED\n', filename);
            setup_results.issues{end+1} = sprintf('Required data file missing: %s', filename);
            criticalDataPresent = false;
            setup_results.success = false;
        else
            fprintf('  - %s - Optional\n', filename);
            fprintf('    %s\n', description);
        end
    end
end

setup_results.data = dataResults;
setup_results.critical_data_present = criticalDataPresent;

fprintf('\n');

%% Final validation and recommendations
fprintf('=== Setup Validation Summary ===\n');

% Overall status
if setup_results.success
    fprintf('✓ Setup completed successfully\n');
else
    fprintf('✗ Setup completed with critical issues\n');
end

% Issue summary
if ~isempty(setup_results.issues)
    fprintf('Issues found: %d\n', length(setup_results.issues));
    for i = 1:length(setup_results.issues)
        fprintf('  • %s\n', setup_results.issues{i});
    end
end

if ~isempty(setup_results.warnings)
    fprintf('Warnings: %d\n', length(setup_results.warnings));
    for i = 1:length(setup_results.warnings)
        fprintf('  • %s\n', setup_results.warnings{i});
    end
end

% Recommendations
fprintf('\nRecommendations:\n');
recommendations = {};

if ~allCriticalAvailable
    recommendations{end+1} = 'Install missing MATLAB toolboxes (Statistics, Image Processing)';
end

if ~allCriticalFunctionsAvailable
    recommendations{end+1} = 'Install MotionMapper toolbox (run scripts/install_dependencies.m)';
end

if ~criticalDataPresent
    recommendations{end+1} = 'Place required data files in data/raw/ directory';
end

if isempty(recommendations)
    recommendations{end+1} = 'Run tests/validate_installation.m to verify setup';
    recommendations{end+1} = 'Use scripts/run_pipeline.m to execute the complete pipeline';
    recommendations{end+1} = 'Check examples/ directory for usage examples';
else
    for i = 1:length(recommendations)
        fprintf('%d. %s\n', i, recommendations{i});
    end
end

setup_results.recommendations = recommendations;

%% Save setup results
setup_results.completed_at = datestr(now);
setup_results.setup_duration = toc;

resultsFile = fullfile(currentDir, 'outputs', 'validation', 'setup_results.mat');
try
    save(resultsFile, 'setup_results');
    fprintf('\nSetup results saved to: %s\n', relativePath(resultsFile, currentDir));
catch ME
    setup_results.warnings{end+1} = sprintf('Could not save setup results: %s', ME.message);
end

fprintf('\n=== Setup Complete ===\n');
fprintf('Duration: %.1f seconds\n\n', setup_results.setup_duration);

end

%% Helper Functions

function [hasToolbox, details] = check_stats_toolbox()
    hasToolbox = exist('tsne', 'file') == 2 && exist('fitlm', 'file') == 2;
    if hasToolbox
        details = 'tsne() and fitlm() functions available';
    else
        details = 'Missing Statistics and Machine Learning Toolbox functions';
    end
end

function [hasToolbox, details] = check_image_toolbox()
    hasToolbox = exist('watershed', 'file') == 2 && exist('bwboundaries', 'file') == 2;
    if hasToolbox
        details = 'watershed() and bwboundaries() functions available';
    else
        details = 'Missing Image Processing Toolbox functions';
    end
end

function [hasToolbox, details] = check_signal_toolbox()
    hasToolbox = exist('medfilt1', 'file') == 2 && exist('smooth', 'file') == 2;
    if hasToolbox
        details = 'medfilt1() and smooth() functions available';
    else
        details = 'Missing Signal Processing Toolbox functions (fallbacks available)';
    end
end

function [hasToolbox, details] = check_parallel_toolbox()
    hasToolbox = license('test', 'Distrib_Computing_Toolbox') && exist('parfor', 'builtin') == 5;
    if hasToolbox
        details = 'Parallel processing available';
    else
        details = 'No parallel processing (will use standard loops)';
    end
end

function [hasToolbox, details] = check_curve_fitting_toolbox()
    hasToolbox = exist('fit', 'file') == 2;
    if hasToolbox
        details = 'Curve fitting functions available';
    else
        details = 'Missing Curve Fitting Toolbox (optional for advanced analysis)';
    end
end

function relPath = relativePath(fullPath, basePath)
    %RELATIVEPATH Get relative path from base path
    try
        relPath = relativepath(fullPath, basePath);
    catch
        # Fallback if relativepath is not available
        if startsWith(fullPath, basePath)
            relPath = fullPath((length(basePath)+2):end); % +2 for filesep
        else
            relPath = fullPath;
        end
    end
end