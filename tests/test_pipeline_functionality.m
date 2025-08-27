function test_results = test_pipeline_functionality()
% TEST_PIPELINE_FUNCTIONALITY - Comprehensive test suite for the pipeline
%
% Tests all major components of the motion analysis pipeline to ensure
% proper functionality and dependency availability
%
% Returns:
%   test_results - Structure with test results and summary

fprintf('=== MOTION ANALYSIS PIPELINE TESTS ===\n\n');

test_results = struct();
test_results.tests = {};
test_results.passed = 0;
test_results.failed = 0;
test_results.warnings = 0;

%% Test 1: Configuration Loading
fprintf('[TEST 1] Configuration Loading...\n');
try
    config = pipeline_config();
    assert(isstruct(config), 'Config should be a structure');
    assert(isfield(config, 'data_dir'), 'Config should have data_dir field');
    assert(isfield(config, 'files'), 'Config should have files field');
    assert(isfield(config, 'parameters'), 'Config should have parameters field');
    
    test_results = record_test(test_results, 'Configuration Loading', 'PASS', '');
    fprintf('  ✓ Configuration loaded successfully\n');
catch ME
    test_results = record_test(test_results, 'Configuration Loading', 'FAIL', ME.message);
    fprintf('  ✗ Configuration loading failed: %s\n', ME.message);
end

%% Test 2: Required Toolboxes
fprintf('\n[TEST 2] Required Toolboxes...\n');
required_toolboxes = {
    'Statistics and Machine Learning Toolbox', 'statistics_toolbox'
    'Image Processing Toolbox', 'image_toolbox'
    'Signal Processing Toolbox', 'signal_toolbox'
};

toolbox_warnings = {};
for i = 1:size(required_toolboxes, 1)
    toolbox_name = required_toolboxes{i, 1};
    toolbox_license = required_toolboxes{i, 2};
    
    if license('test', toolbox_license)
        fprintf('  ✓ %s available\n', toolbox_name);
    else
        toolbox_warnings{end+1} = toolbox_name; %#ok<AGROW>
        fprintf('  ⚠ %s NOT available\n', toolbox_name);
    end
end

if isempty(toolbox_warnings)
    test_results = record_test(test_results, 'Required Toolboxes', 'PASS', '');
else
    warning_msg = sprintf('Missing: %s', strjoin(toolbox_warnings, ', '));
    test_results = record_test(test_results, 'Required Toolboxes', 'WARN', warning_msg);
end

%% Test 3: Core Functions Availability
fprintf('\n[TEST 3] SocialMapper Core Functions...\n');
required_functions = {
    'findWavelets'
    'findTemplatesFromData'
    'findTDistributedProjections_fmin' 
    'findWatershedRegions_v2'
    'setRunParameters'
    'tsne'
    'combineCells'
};

missing_functions = {};
for i = 1:length(required_functions)
    func_name = required_functions{i};
    if exist(func_name, 'file') == 2
        fprintf('  ✓ %s found\n', func_name);
    else
        missing_functions{end+1} = func_name; %#ok<AGROW>
        fprintf('  ✗ %s NOT found\n', func_name);
    end
end

if isempty(missing_functions)
    test_results = record_test(test_results, 'SocialMapper Functions', 'PASS', '');
else
    error_msg = sprintf('Missing: %s', strjoin(missing_functions, ', '));
    test_results = record_test(test_results, 'SocialMapper Functions', 'FAIL', error_msg);
end

%% Test 4: ComBat Availability
fprintf('\n[TEST 4] ComBat Batch Correction...\n');
try
    % Test if combat function exists
    if exist('combat', 'file') == 2
        fprintf('  ✓ ComBat function found\n');
        
        % Test basic functionality with dummy data
        X = randn(10, 20);
        batch = [ones(1,10) 2*ones(1,10)];
        X_corrected = combat(X, batch, [], true);
        
        if size(X_corrected, 1) == size(X, 1) && size(X_corrected, 2) == size(X, 2)
            test_results = record_test(test_results, 'ComBat Availability', 'PASS', '');
            fprintf('  ✓ ComBat function working\n');
        else
            test_results = record_test(test_results, 'ComBat Availability', 'WARN', 'ComBat returns wrong size');
            fprintf('  ⚠ ComBat function returns unexpected size\n');
        end
    else
        test_results = record_test(test_results, 'ComBat Availability', 'WARN', 'ComBat not found - using placeholder');
        fprintf('  ⚠ ComBat function not found (will use simplified version)\n');
    end
catch ME
    test_results = record_test(test_results, 'ComBat Availability', 'WARN', ME.message);
    fprintf('  ⚠ ComBat test failed: %s\n', ME.message);
end

%% Test 5: Pipeline Structure
fprintf('\n[TEST 5] Pipeline Structure...\n');
expected_dirs = {'src', 'config', 'dependencies', 'scripts', 'tests'};
missing_dirs = {};

for i = 1:length(expected_dirs)
    dir_name = expected_dirs{i};
    if exist(dir_name, 'dir')
        fprintf('  ✓ %s/ directory exists\n', dir_name);
    else
        missing_dirs{end+1} = dir_name; %#ok<AGROW>
        fprintf('  ✗ %s/ directory missing\n', dir_name);
    end
end

if isempty(missing_dirs)
    test_results = record_test(test_results, 'Pipeline Structure', 'PASS', '');
else
    error_msg = sprintf('Missing directories: %s', strjoin(missing_dirs, ', '));
    test_results = record_test(test_results, 'Pipeline Structure', 'FAIL', error_msg);
end

%% Test 6: Core Pipeline Functions
fprintf('\n[TEST 6] Core Pipeline Functions...\n');
pipeline_functions = {
    'custom_embedding_pipeline'
    'load_training_data'
    'setup_skeleton_and_parameters'
    'pipeline_config'
};

missing_pipeline_funcs = {};
for i = 1:length(pipeline_functions)
    func_name = pipeline_functions{i};
    if exist(func_name, 'file') == 2
        fprintf('  ✓ %s found\n', func_name);
    else
        missing_pipeline_funcs{end+1} = func_name; %#ok<AGROW>
        fprintf('  ✗ %s NOT found\n', func_name);
    end
end

if isempty(missing_pipeline_funcs)
    test_results = record_test(test_results, 'Pipeline Functions', 'PASS', '');
else
    error_msg = sprintf('Missing: %s', strjoin(missing_pipeline_funcs, ', '));
    test_results = record_test(test_results, 'Pipeline Functions', 'FAIL', error_msg);
end

%% Test 7: Memory and Performance Check
fprintf('\n[TEST 7] System Resources...\n');
try
    % Check available memory (rough estimate)
    mem_info = memory;
    if isfield(mem_info, 'MemAvailableAllArrays')
        available_gb = mem_info.MemAvailableAllArrays / 1e9;
        fprintf('  ✓ Available memory: %.1f GB\n', available_gb);
        
        if available_gb < 8
            test_results = record_test(test_results, 'System Resources', 'WARN', 'Low memory (<8GB)');
            fprintf('  ⚠ Low memory available - consider reducing batch sizes\n');
        else
            test_results = record_test(test_results, 'System Resources', 'PASS', '');
        end
    else
        test_results = record_test(test_results, 'System Resources', 'PASS', 'Memory info not available');
        fprintf('  ✓ Memory information not available (continuing)\n');
    end
catch ME
    test_results = record_test(test_results, 'System Resources', 'WARN', ME.message);
    fprintf('  ⚠ Memory check failed: %s\n', ME.message);
end

%% Test 8: Basic Mathematical Operations
fprintf('\n[TEST 8] Basic Mathematical Operations...\n');
try
    % Test basic operations used in pipeline
    X = randn(100, 50);
    
    % PCA-like operations
    mu = mean(X, 1);
    X_centered = bsxfun(@minus, X, mu);
    C = cov(X_centered);
    [vecs, vals] = eig(C);
    
    % t-SNE availability (crucial)
    if exist('tsne', 'file') == 2
        fprintf('  ✓ t-SNE available\n');
    else
        error('t-SNE function not available');
    end
    
    % Distance calculations
    distances = pdist2(X(1:10, :), X(11:20, :));
    
    % Watershed operations
    if exist('watershed', 'file') == 2
        fprintf('  ✓ Watershed function available\n');
    else
        fprintf('  ⚠ Watershed function not found\n');
    end
    
    test_results = record_test(test_results, 'Mathematical Operations', 'PASS', '');
    fprintf('  ✓ Basic mathematical operations working\n');
    
catch ME
    test_results = record_test(test_results, 'Mathematical Operations', 'FAIL', ME.message);
    fprintf('  ✗ Mathematical operations failed: %s\n', ME.message);
end

%% Display Summary
fprintf('\n=== TEST SUMMARY ===\n');
fprintf('Tests passed: %d\n', test_results.passed);
fprintf('Tests failed: %d\n', test_results.failed);
fprintf('Warnings: %d\n', test_results.warnings);

if test_results.failed > 0
    fprintf('\n❌ PIPELINE NOT READY - Fix failed tests before proceeding\n');
    fprintf('Failed tests:\n');
    for i = 1:length(test_results.tests)
        if strcmp(test_results.tests{i}.result, 'FAIL')
            fprintf('  - %s: %s\n', test_results.tests{i}.name, test_results.tests{i}.message);
        end
    end
else
    fprintf('\n✅ PIPELINE READY - All critical tests passed\n');
    if test_results.warnings > 0
        fprintf('Note: %d warnings - pipeline will work but may have reduced functionality\n', test_results.warnings);
    end
end

fprintf('\nPipeline test completed!\n');

end

function test_results = record_test(test_results, test_name, result, message)
% Record a test result

test_entry = struct();
test_entry.name = test_name;
test_entry.result = result;
test_entry.message = message;

test_results.tests{end+1} = test_entry;

switch result
    case 'PASS'
        test_results.passed = test_results.passed + 1;
    case 'FAIL'
        test_results.failed = test_results.failed + 1;
    case 'WARN'
        test_results.warnings = test_results.warnings + 1;
end

end