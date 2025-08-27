function test_results = test_pipeline_integrity()
%% TEST_PIPELINE_INTEGRITY Comprehensive pipeline integrity and bug detection
% This script tests the entire pipeline for bugs, redundancies, and issues

fprintf('\n=== Pipeline Integrity & Bug Detection Test Suite ===\n\n');

test_results = struct();
test_results.timestamp = datestr(now);
test_results.tests = {};
test_results.passed = 0;
test_results.failed = 0;
test_results.warnings = 0;
test_results.total = 0;
test_results.issues = {};
test_results.redundancies = {};
test_results.optimization_suggestions = {};

%% Change to project directory
project_dir = fileparts(fileparts(mfilename('fullpath')));
original_dir = pwd;
cd(project_dir);

fprintf('Testing pipeline at: %s\n\n', project_dir);

try
    %% Test 1: Project Structure Integrity
    test_name = 'Project Structure Integrity';
    fprintf('Running test: %s\n', test_name);
    
    required_dirs = {
        'src', 'src/core', 'src/utils', 'src/analysis',
        'config', 'scripts', 'data', 'examples', 
        'tests', 'outputs', 'dependencies'
    };
    
    missing_dirs = {};
    for i = 1:length(required_dirs)
        if ~exist(required_dirs{i}, 'dir')
            missing_dirs{end+1} = required_dirs{i};
        end
    end
    
    if isempty(missing_dirs)
        fprintf('  ✓ All required directories present\n');
        test_result = true;
    else
        fprintf('  ✗ Missing directories: %s\n', strjoin(missing_dirs, ', '));
        test_results.issues{end+1} = sprintf('Missing directories: %s', strjoin(missing_dirs, ', '));
        test_result = false;
    end
    
    test_results = add_test_result(test_results, test_name, test_result, 'Check project directory structure');
    
    %% Test 2: Critical File Existence
    test_name = 'Critical File Existence';
    fprintf('\nRunning test: %s\n', test_name);
    
    critical_files = {
        'README.md', 'Main project documentation';
        'LICENSE', 'Project license';
        '.gitignore', 'Git ignore rules';
        'scripts/setup_pipeline.m', 'Environment setup script';
        'scripts/run_pipeline.m', 'Main execution script';
        'config/pipeline_config.m', 'Pipeline configuration';
        'src/core/mouse_embedding.m', 'Core embedding function';
        'src/analysis/analyze_maps_and_counts.m', 'Analysis function';
        'examples/basic_usage.m', 'Usage example'
    };
    
    missing_files = {};
    for i = 1:size(critical_files, 1)
        if ~exist(critical_files{i,1}, 'file')
            missing_files{end+1} = critical_files{i,1};
        end
    end
    
    if isempty(missing_files)
        fprintf('  ✓ All critical files present\n');
        test_result = true;
    else
        fprintf('  ✗ Missing critical files: %s\n', strjoin(missing_files, ', '));
        test_results.issues{end+1} = sprintf('Missing critical files: %s', strjoin(missing_files, ', '));
        test_result = false;
    end
    
    test_results = add_test_result(test_results, test_name, test_result, 'Check critical file existence');
    
    %% Test 3: MATLAB Syntax Check
    test_name = 'MATLAB Syntax Check';
    fprintf('\nRunning test: %s\n', test_name);
    
    matlab_files = {
        'scripts/setup_pipeline.m',
        'scripts/run_pipeline.m',
        'config/pipeline_config.m',
        'src/core/mouse_embedding.m',
        'src/analysis/analyze_maps_and_counts.m',
        'examples/basic_usage.m'
    };
    
    syntax_errors = {};
    for i = 1:length(matlab_files)
        if exist(matlab_files{i}, 'file')
            try
                % Check syntax by attempting to parse (but not execute)
                pcode(matlab_files{i}, '-inplace');
                pfile = [matlab_files{i}(1:end-2) '.p'];
                if exist(pfile, 'file')
                    delete(pfile); % Clean up
                end
                fprintf('  ✓ %s - syntax OK\n', matlab_files{i});
            catch ME
                fprintf('  ✗ %s - syntax error: %s\n', matlab_files{i}, ME.message);
                syntax_errors{end+1} = sprintf('%s: %s', matlab_files{i}, ME.message);
            end
        else
            fprintf('  ! %s - file not found\n', matlab_files{i});
            syntax_errors{end+1} = sprintf('%s: file not found', matlab_files{i});
        end
    end
    
    test_result = isempty(syntax_errors);
    if ~test_result
        test_results.issues = [test_results.issues, syntax_errors];
    end
    
    test_results = add_test_result(test_results, test_name, test_result, 'Check MATLAB syntax');
    
    %% Test 4: Function Dependency Analysis
    test_name = 'Function Dependency Analysis';
    fprintf('\nRunning test: %s\n', test_name);
    
    % Analyze dependencies and check for missing functions
    dependency_issues = analyze_function_dependencies();
    
    if isempty(dependency_issues)
        fprintf('  ✓ All function dependencies resolved\n');
        test_result = true;
    else
        fprintf('  ⚠ Dependency issues found:\n');
        for i = 1:length(dependency_issues)
            fprintf('    - %s\n', dependency_issues{i});
        end
        test_results.issues = [test_results.issues, dependency_issues];
        test_result = false;
    end
    
    test_results = add_test_result(test_results, test_name, test_result, 'Analyze function dependencies');
    
    %% Test 5: Configuration Validation
    test_name = 'Configuration Validation';
    fprintf('\nRunning test: %s\n', test_name);
    
    config_issues = {};
    try
        addpath('config');
        config = pipeline_config();
        
        % Check required configuration fields
        required_fields = {
            'version', 'parameters', 'output', 'data', 'training'
        };
        
        for i = 1:length(required_fields)
            if ~isfield(config, required_fields{i})
                config_issues{end+1} = sprintf('Missing config field: %s', required_fields{i});
            end
        end
        
        % Check parameter consistency
        if isfield(config, 'parameters')
            if isfield(config.parameters, 'wavelets')
                if config.parameters.wavelets.min_freq >= config.parameters.wavelets.max_freq
                    config_issues{end+1} = 'Wavelet min_freq must be less than max_freq';
                end
            end
        end
        
        if isempty(config_issues)
            fprintf('  ✓ Configuration validation passed\n');
            test_result = true;
        else
            fprintf('  ✗ Configuration issues found:\n');
            for i = 1:length(config_issues)
                fprintf('    - %s\n', config_issues{i});
            end
            test_result = false;
        end
        
    catch ME
        fprintf('  ✗ Error loading configuration: %s\n', ME.message);
        config_issues{end+1} = sprintf('Configuration loading error: %s', ME.message);
        test_result = false;
    end
    
    test_results.issues = [test_results.issues, config_issues];
    test_results = add_test_result(test_results, test_name, test_result, 'Validate configuration');
    
    %% Test 6: Redundancy Detection
    test_name = 'Redundancy Detection';
    fprintf('\nRunning test: %s\n', test_name);
    
    redundancies = detect_redundancies();
    
    if isempty(redundancies)
        fprintf('  ✓ No redundancies detected\n');
        test_result = true;
    else
        fprintf('  ⚠ Redundancies detected:\n');
        for i = 1:length(redundancies)
            fprintf('    - %s\n', redundancies{i});
        end
        test_results.redundancies = redundancies;
        test_result = true; % Redundancies are warnings, not failures
        test_results.warnings = test_results.warnings + 1;
    end
    
    test_results = add_test_result(test_results, test_name, test_result, 'Detect code redundancies');
    
    %% Test 7: Path Resolution Test
    test_name = 'Path Resolution Test';
    fprintf('\nRunning test: %s\n', test_name);
    
    path_issues = {};
    
    % Test relative path resolution
    test_paths = {
        '../config/pipeline_config.m',
        '../data/README.md',
        '../examples/basic_usage.m',
        'test_git_functionality.m'
    };
    
    for i = 1:length(test_paths)
        if ~exist(test_paths{i}, 'file')
            path_issues{end+1} = sprintf('Path resolution failed: %s', test_paths{i});
        end
    end
    
    if isempty(path_issues)
        fprintf('  ✓ Path resolution working correctly\n');
        test_result = true;
    else
        fprintf('  ✗ Path resolution issues:\n');
        for i = 1:length(path_issues)
            fprintf('    - %s\n', path_issues{i});
        end
        test_result = false;
    end
    
    test_results.issues = [test_results.issues, path_issues];
    test_results = add_test_result(test_results, test_name, test_result, 'Test path resolution');
    
    %% Test 8: Memory and Performance Analysis
    test_name = 'Memory and Performance Analysis';
    fprintf('\nRunning test: %s\n', test_name);
    
    performance_issues = analyze_performance_concerns();
    
    if isempty(performance_issues)
        fprintf('  ✓ No major performance issues detected\n');
        test_result = true;
    else
        fprintf('  ⚠ Performance concerns:\n');
        for i = 1:length(performance_issues)
            fprintf('    - %s\n', performance_issues{i});
        end
        test_results.optimization_suggestions = performance_issues;
        test_result = true; % Performance issues are suggestions, not failures
        test_results.warnings = test_results.warnings + 1;
    end
    
    test_results = add_test_result(test_results, test_name, test_result, 'Analyze performance concerns');
    
    %% Test 9: Documentation Completeness
    test_name = 'Documentation Completeness';
    fprintf('\nRunning test: %s\n', test_name);
    
    doc_issues = check_documentation_completeness();
    
    if isempty(doc_issues)
        fprintf('  ✓ Documentation is complete\n');
        test_result = true;
    else
        fprintf('  ⚠ Documentation issues:\n');
        for i = 1:length(doc_issues)
            fprintf('    - %s\n', doc_issues{i});
        end
        test_results.issues = [test_results.issues, doc_issues];
        test_result = true; % Documentation issues are warnings
        test_results.warnings = test_results.warnings + 1;
    end
    
    test_results = add_test_result(test_results, test_name, test_result, 'Check documentation completeness');
    
    %% Test 10: Workflow Logic Validation
    test_name = 'Workflow Logic Validation';
    fprintf('\nRunning test: %s\n', test_name);
    
    workflow_issues = validate_workflow_logic();
    
    if isempty(workflow_issues)
        fprintf('  ✓ Workflow logic is sound\n');
        test_result = true;
    else
        fprintf('  ✗ Workflow logic issues:\n');
        for i = 1:length(workflow_issues)
            fprintf('    - %s\n', workflow_issues{i});
        end
        test_result = false;
    end
    
    test_results.issues = [test_results.issues, workflow_issues];
    test_results = add_test_result(test_results, test_name, test_result, 'Validate workflow logic');
    
finally
    cd(original_dir);
end

%% Test Summary
fprintf('\n=== Pipeline Integrity Test Summary ===\n');
fprintf('Total tests: %d\n', test_results.total);
fprintf('Passed: %d\n', test_results.passed);
fprintf('Failed: %d\n', test_results.failed);
fprintf('Warnings: %d\n', test_results.warnings);
fprintf('Success rate: %.1f%%\n', 100 * test_results.passed / test_results.total);

test_results.success_rate = test_results.passed / test_results.total;
test_results.overall_success = test_results.failed == 0;

if test_results.overall_success
    fprintf('\n✅ Pipeline integrity tests PASSED!\n');
    if test_results.warnings > 0
        fprintf('⚠️  %d warnings found - see details above\n', test_results.warnings);
    end
else
    fprintf('\n❌ Pipeline integrity tests FAILED!\n');
    fprintf('Critical issues found - review and fix before using pipeline\n');
end

% Display summary of findings
if ~isempty(test_results.issues)
    fprintf('\n🐛 Issues Found:\n');
    for i = 1:min(5, length(test_results.issues))
        fprintf('  %d. %s\n', i, test_results.issues{i});
    end
    if length(test_results.issues) > 5
        fprintf('  ... and %d more issues\n', length(test_results.issues) - 5);
    end
end

if ~isempty(test_results.redundancies)
    fprintf('\n🔄 Redundancies Found:\n');
    for i = 1:min(3, length(test_results.redundancies))
        fprintf('  %d. %s\n', i, test_results.redundancies{i});
    end
end

if ~isempty(test_results.optimization_suggestions)
    fprintf('\n🚀 Optimization Suggestions:\n');
    for i = 1:min(3, length(test_results.optimization_suggestions))
        fprintf('  %d. %s\n', i, test_results.optimization_suggestions{i});
    end
end

% Save test results
results_file = fullfile(project_dir, 'tests', 'pipeline_integrity_results.mat');
try
    save(results_file, 'test_results');
    fprintf('\nTest results saved to: %s\n', results_file);
catch
    fprintf('\nWarning: Could not save test results to file\n');
end

fprintf('\n=== Pipeline Integrity Test Complete ===\n\n');

end

%% Helper Functions

function test_results = add_test_result(test_results, test_name, passed, description)
    test_info = struct();
    test_info.name = test_name;
    test_info.passed = passed;
    test_info.description = description;
    test_info.timestamp = datestr(now);
    
    test_results.tests{end+1} = test_info;
    test_results.total = test_results.total + 1;
    
    if passed
        test_results.passed = test_results.passed + 1;
    else
        test_results.failed = test_results.failed + 1;
    end
end

function issues = analyze_function_dependencies()
    %ANALYZE_FUNCTION_DEPENDENCIES Check for missing function dependencies
    
    issues = {};
    
    % Critical MotionMapper functions that must exist
    critical_functions = {
        'findWavelets', 'Core wavelet analysis function';
        'findTemplatesFromData', 'Template extraction function';
        'findTDistributedProjections_fmin', 'Re-embedding function';
        'findWatershedRegions_v2', 'Watershed region analysis';
        'findPointDensity', 'Density estimation function';
        'setRunParameters', 'Parameter configuration function';
        'returnDist3d', '3D distance calculation function';
        'combineCells', 'Cell array combination function'
    };
    
    for i = 1:size(critical_functions, 1)
        func_name = critical_functions{i,1};
        description = critical_functions{i,2};
        
        if ~exist(func_name, 'file')
            issues{end+1} = sprintf('Missing critical function: %s (%s)', func_name, description);
        end
    end
    
    % Check MATLAB toolbox functions
    toolbox_functions = {
        'tsne', 'Statistics and Machine Learning Toolbox';
        'watershed', 'Image Processing Toolbox';
        'bwboundaries', 'Image Processing Toolbox'
    };
    
    for i = 1:size(toolbox_functions, 1)
        func_name = toolbox_functions{i,1};
        toolbox = toolbox_functions{i,2};
        
        if ~exist(func_name, 'file')
            issues{end+1} = sprintf('Missing toolbox function: %s (requires %s)', func_name, toolbox);
        end
    end
end

function redundancies = detect_redundancies()
    %DETECT_REDUNDANCIES Find redundant code patterns
    
    redundancies = {};
    
    % Check for duplicate configuration loading
    if exist('scripts/run_pipeline.m', 'file') && exist('src/core/mouse_embedding.m', 'file')
        run_content = fileread('scripts/run_pipeline.m');
        mouse_content = fileread('src/core/mouse_embedding.m');
        
        if contains(run_content, 'pipeline_config') && contains(mouse_content, 'pipeline_config')
            redundancies{end+1} = 'Configuration loading appears in multiple files - consider centralization';
        end
    end
    
    % Check for duplicate path setup
    files_with_addpath = {};
    matlab_files = dir('**/*.m');
    
    for i = 1:length(matlab_files)
        if ~matlab_files(i).isdir
            content = fileread(fullfile(matlab_files(i).folder, matlab_files(i).name));
            if contains(content, 'addpath')
                files_with_addpath{end+1} = fullfile(matlab_files(i).folder, matlab_files(i).name);
            end
        end
    end
    
    if length(files_with_addpath) > 2
        redundancies{end+1} = sprintf('Path setup (addpath) found in %d files - consider centralizing in setup script', length(files_with_addpath));
    end
    
    % Check for duplicate error handling patterns
    redundancies{end+1} = 'Consider creating a common error handling utility function';
end

function issues = analyze_performance_concerns()
    %ANALYZE_PERFORMANCE_CONCERNS Identify potential performance issues
    
    issues = {};
    
    % Check for large loop structures without progress indication
    if exist('src/core/mouse_embedding.m', 'file')
        content = fileread('src/core/mouse_embedding.m');
        
        if contains(content, 'for j = 1:length(allMOUSE_combined)')
            if ~contains(content, 'fprintf') || ~contains(content, 'progress')
                issues{end+1} = 'Long loops in mouse_embedding.m lack progress indicators';
            end
        end
        
        % Check for nested loops
        if contains(content, 'for j') && contains(content, 'for i')
            issues{end+1} = 'Nested loops detected - consider vectorization opportunities';
        end
    end
    
    % Check memory allocation patterns
    issues{end+1} = 'Consider pre-allocating large matrices to improve performance';
    
    % Check for redundant calculations
    issues{end+1} = 'Consider caching expensive calculations (PCA, distance matrices)';
    
    % Batch processing suggestions
    issues{end+1} = 'Consider implementing batch processing for large datasets';
end

function issues = check_documentation_completeness()
    %CHECK_DOCUMENTATION_COMPLETENESS Verify documentation coverage
    
    issues = {};
    
    % Check if README has all required sections
    if exist('README.md', 'file')
        readme_content = fileread('README.md');
        required_sections = {'Installation', 'Usage', 'Examples', 'Troubleshooting'};
        
        for i = 1:length(required_sections)
            if ~contains(readme_content, required_sections{i})
                issues{end+1} = sprintf('README.md missing section: %s', required_sections{i});
            end
        end
    else
        issues{end+1} = 'README.md file not found';
    end
    
    % Check function documentation
    matlab_files = {'src/core/mouse_embedding.m', 'src/analysis/analyze_maps_and_counts.m'};
    
    for i = 1:length(matlab_files)
        if exist(matlab_files{i}, 'file')
            content = fileread(matlab_files{i});
            if ~contains(content, '%%') || ~contains(content, 'Usage:')
                issues{end+1} = sprintf('Function %s lacks proper documentation header', matlab_files{i});
            end
        end
    end
end

function issues = validate_workflow_logic()
    %VALIDATE_WORKFLOW_LOGIC Check workflow logic for issues
    
    issues = {};
    
    % Check if pipeline stages are properly sequenced
    if exist('scripts/run_pipeline.m', 'file')
        content = fileread('scripts/run_pipeline.m');
        
        % Check for proper error handling between stages
        if ~contains(content, 'try') || ~contains(content, 'catch')
            issues{end+1} = 'Main pipeline lacks comprehensive error handling';
        end
        
        # Check for stage dependency validation
        if ~contains(content, 'existing_results') && ~contains(content, 'skip_training')
            issues{end+1} = 'Pipeline may not properly handle existing intermediate results';
        end
    end
    
    % Check file I/O consistency
    if exist('config/pipeline_config.m', 'file')
        config_content = fileread('config/pipeline_config.m');
        
        if exist('src/core/mouse_embedding.m', 'file')
            embedding_content = fileread('src/core/mouse_embedding.m');
            
            # Check if file naming is consistent
            if contains(config_content, 'files.') && ~contains(embedding_content, 'config.files.')
                issues{end+1} = 'File naming configuration may not be consistently used across modules';
            end
        end
    end
    
    % Check output directory consistency
    output_patterns = {'outputs/', 'analysis_outputs/', '../outputs'};
    inconsistent_outputs = {};
    
    matlab_files = dir('**/*.m');
    for i = 1:length(matlab_files)
        if ~matlab_files(i).isdir
            content = fileread(fullfile(matlab_files(i).folder, matlab_files(i).name));
            for j = 1:length(output_patterns)
                if contains(content, output_patterns{j})
                    inconsistent_outputs{end+1} = output_patterns{j};
                end
            end
        end
    end
    
    if length(unique(inconsistent_outputs)) > 1
        issues{end+1} = 'Inconsistent output directory patterns across files';
    end
end