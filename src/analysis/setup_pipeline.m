function setup_pipeline()
%% Setup Pipeline - Initialize environment and check dependencies
% This script sets up the spontaneous pain analysis pipeline by:
% 1. Adding all necessary paths to MATLAB
% 2. Checking for required dependencies and toolboxes
% 3. Validating MEX files for the current platform
% 4. Creating output directories
% 5. Validating input data if present

fprintf('\n=== Spontaneous Pain Analysis Pipeline Setup ===\n\n');

%% Get current directory and setup paths
currentDir = pwd;
fprintf('Current directory: %s\n', currentDir);

% Add main directories to path
pathsToAdd = {
    currentDir,
    fullfile(currentDir, 'config'),
    fullfile(currentDir, 'utils'),
    fullfile(currentDir, 'dependencies'),
    fullfile(currentDir, 'dependencies', 'MotionMapper'),
    fullfile(currentDir, 'dependencies', 'MotionMapper', 'MotionMapperUtilities'),
    fullfile(currentDir, 'dependencies', 'MotionMapper', 'MotionMapperUtilities', 'tSNE'),
    fullfile(currentDir, 'dependencies', 'MotionMapper', 'MotionMapperUtilities', 'utilities'),
    fullfile(currentDir, 'dependencies', 'MotionMapper', 'MotionMapperUtilities', 'metrics'),
    fullfile(currentDir, 'dependencies', 'SocialMapper-main'),
    fullfile(currentDir, 'dependencies', 'SocialMapper-main', 'MotionMapperUtilities'),
    fullfile(currentDir, 'dependencies', 'SocialMapper-main', 'MotionMapperUtilities', 'tSNE'),
    fullfile(currentDir, 'dependencies', 'SocialMapper-main', 'MotionMapperUtilities', 'utilities'),
    fullfile(currentDir, 'dependencies', 'SocialMapper-main', 'MotionMapperUtilities', 'metrics'),
    fullfile(currentDir, 'dependencies', 'SocialMapper-main', 'utilities'),
    fullfile(currentDir, 'dependencies', 'SocialMapper-main', 'mouseEmbedding'),
    fullfile(currentDir, 'dependencies', 'Animator')
};

fprintf('Adding paths to MATLAB:\n');
for i = 1:length(pathsToAdd)
    if exist(pathsToAdd{i}, 'dir')
        addpath(pathsToAdd{i});
        fprintf('  ✓ %s\n', pathsToAdd{i});
    else
        fprintf('  ✗ %s (not found)\n', pathsToAdd{i});
    end
end
fprintf('\n');

%% Create output directories
outputDirs = {
    'outputs',
    'outputs/results',
    'outputs/figures',
    'outputs/figures/per_video',
    'outputs/figures/group_comparisons',
    'outputs/csv',
    'outputs/csv/frame_indices_per_video',
    'outputs/csv/sequence_metadata',
    'outputs/csv/statistical_analysis',
    'outputs/validation'
};

fprintf('Creating output directories:\n');
for i = 1:length(outputDirs)
    dirPath = fullfile(currentDir, outputDirs{i});
    if ~exist(dirPath, 'dir')
        mkdir(dirPath);
        fprintf('  ✓ Created: %s\n', outputDirs{i});
    else
        fprintf('  ✓ Exists: %s\n', outputDirs{i});
    end
end
fprintf('\n');

%% Check MATLAB toolbox dependencies
fprintf('Checking MATLAB toolbox dependencies:\n');

% Critical toolboxes
toolboxes = {
    'Statistics and Machine Learning Toolbox', 'Statistics_Toolbox', true, @check_stats_toolbox;
    'Image Processing Toolbox', 'Image_Toolbox', true, @check_image_toolbox;
    'Signal Processing Toolbox', 'Signal_Toolbox', false, @check_signal_toolbox;
    'Parallel Computing Toolbox', 'Distrib_Computing_Toolbox', false, @check_parallel_toolbox
};

allCritical = true;
for i = 1:size(toolboxes, 1)
    [hasToolbox, details] = toolboxes{i,4}();
    if hasToolbox
        fprintf('  ✓ %s - Available\n', toolboxes{i,1});
        if ~isempty(details)
            fprintf('    %s\n', details);
        end
    else
        if toolboxes{i,3} % Critical
            fprintf('  ✗ %s - REQUIRED BUT MISSING\n', toolboxes{i,1});
            allCritical = false;
        else
            fprintf('  ! %s - Optional (missing)\n', toolboxes{i,1});
        end
        if ~isempty(details)
            fprintf('    %s\n', details);
        end
    end
end
fprintf('\n');

%% Check custom function dependencies
fprintf('Checking custom function dependencies:\n');

% MotionMapper functions (critical)
motionMapperFunctions = {
    'findWavelets', true;
    'findTemplatesFromData', true;
    'findTDistributedProjections_fmin', true;
    'findWatershedRegions_v2', true;
    'findPointDensity', true;
    'setRunParameters', true;
    'returnDist3d', true;
    'combineCells', true;
    'tsne', false  % May be in toolbox or custom
};

for i = 1:size(motionMapperFunctions, 1)
    funcName = motionMapperFunctions{i,1};
    isCritical = motionMapperFunctions{i,2};
    
    if exist(funcName, 'file')
        funcPath = which(funcName);
        fprintf('  ✓ %s\n', funcName);
        % Show path for custom functions (not built-in)
        if ~contains(funcPath, matlabroot)
            fprintf('    → %s\n', funcPath);
        end
    else
        if isCritical
            fprintf('  ✗ %s - CRITICAL FUNCTION MISSING\n', funcName);
            allCritical = false;
        else
            fprintf('  ! %s - Optional (missing)\n', funcName);
        end
    end
end
fprintf('\n');

%% Check and validate MEX files
fprintf('Checking MEX files for current platform:\n');
mexFiles = {
    'findListDistances';
    'ba_interp2'
};

computer_type = computer('arch');
fprintf('Current platform: %s\n', computer_type);

for i = 1:length(mexFiles)
    mexName = mexFiles{i};
    mexPath = which([mexName '.' mexext]);
    
    if ~isempty(mexPath)
        fprintf('  ✓ %s.%s\n', mexName, mexext);
        
        % Test MEX file
        try
            if strcmp(mexName, 'findListDistances')
                % Test with small dummy data
                test_result = feval(mexName, rand(10,3), rand(5,3));
                fprintf('    → MEX file tested successfully\n');
            elseif strcmp(mexName, 'ba_interp2')
                fprintf('    → MEX file found (not tested)\n');
            end
        catch ME
            fprintf('    ! MEX file found but failed test: %s\n', ME.message);
        end
    else
        fprintf('  ✗ %s.%s - Missing\n', mexName, mexext);
        fprintf('    → Consider recompiling or using fallback functions\n');
    end
end
fprintf('\n');

%% Validate input data if present
fprintf('Checking for input data:\n');
dataFiles = {
    'mouseFileOrder.mat', 'Mouse file order and metadata', true;
    'SNI_2.mat', 'Training data 1 (SNI)', false;
    'week4-TBI_3.mat', 'Training data 2 (TBI)', false
};

dataPresent = true;
for i = 1:size(dataFiles, 1)
    filename = dataFiles{i,1};
    description = dataFiles{i,2};
    isRequired = dataFiles{i,3};
    
    if exist(filename, 'file')
        fprintf('  ✓ %s (%s)\n', filename, description);
        
        % Validate structure for key files
        if strcmp(filename, 'mouseFileOrder.mat')
            try
                data = load(filename);
                if isfield(data, 'mouseOrderShort') && isfield(data, 'metadata')
                    fprintf('    → Valid structure confirmed\n');
                else
                    fprintf('    ! Structure may be invalid - missing expected fields\n');
                end
            catch
                fprintf('    ! Could not validate file structure\n');
            end
        end
    else
        if isRequired
            fprintf('  ✗ %s (%s) - REQUIRED\n', filename, description);
            dataPresent = false;
        else
            fprintf('  - %s (%s) - Not found (will be created or not needed)\n', filename, description);
        end
    end
end
fprintf('\n');

%% Summary and recommendations
fprintf('=== Setup Summary ===\n');
if allCritical
    fprintf('✓ All critical dependencies are available\n');
else
    fprintf('✗ Some critical dependencies are missing\n');
end

if dataPresent
    fprintf('✓ Required input data is present\n');
else
    fprintf('! Some required data files are missing\n');
end

fprintf('\nRecommendations:\n');
if ~allCritical
    fprintf('1. Install missing MATLAB toolboxes before running pipeline\n');
    fprintf('2. Run utils/install_dependencies.m to get MotionMapper\n');
end

fprintf('3. Run validate_full_pipeline.m before first use\n');
fprintf('4. Check config/group_config.m for your experimental setup\n');
fprintf('5. Use run_full_pipeline.m for automated execution\n');

fprintf('\n=== Setup Complete ===\n');

%% Save setup results
setupResults = struct();
setupResults.timestamp = datestr(now);
setupResults.allCritical = allCritical;
setupResults.dataPresent = dataPresent;
setupResults.platform = computer_type;
setupResults.matlabVersion = version;

save(fullfile('outputs', 'validation', 'setup_results.mat'), 'setupResults');
fprintf('Setup results saved to outputs/validation/setup_results.mat\n\n');

end

%% Helper functions for toolbox checking
function [hasToolbox, details] = check_stats_toolbox()
    hasToolbox = exist('tsne', 'file') == 2;
    if hasToolbox
        details = 'tsne() function available';
    else
        details = 'Missing tsne() - install Statistics and Machine Learning Toolbox';
    end
end

function [hasToolbox, details] = check_image_toolbox()
    hasToolbox = exist('watershed', 'file') == 2 && exist('bwboundaries', 'file') == 2;
    if hasToolbox
        details = 'watershed() and bwboundaries() available';
    else
        details = 'Missing watershed/bwboundaries - install Image Processing Toolbox';
    end
end

function [hasToolbox, details] = check_signal_toolbox()
    hasToolbox = exist('medfilt1', 'file') == 2;
    if hasToolbox
        details = 'medfilt1() available';
    else
        details = 'Missing medfilt1() - some filtering may be skipped';
    end
end

function [hasToolbox, details] = check_parallel_toolbox()
    hasToolbox = exist('parfor', 'file') == 2;
    if hasToolbox
        details = 'Parallel processing available';
    else
        details = 'No parallel processing - will use standard for loops';
    end
end