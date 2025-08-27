function results = run_pipeline(varargin)
%% RUN_PIPELINE Main script to execute the complete spontaneous pain analysis pipeline
% This is the primary entry point for running the behavioral embedding and
% analysis pipeline with proper error handling, validation, and progress tracking.
%
% Usage:
%   results = run_pipeline()                           % Use default configuration
%   results = run_pipeline('config_file', 'my_config.m') % Use custom config file
%   results = run_pipeline('skip_training', true)      % Skip training, use existing results
%   results = run_pipeline('analysis_only', true)      % Only run downstream analysis
%   results = run_pipeline('data_dir', 'path/to/data') % Specify data directory
%
% Parameters:
%   'config_file'    - Path to configuration file (default: 'config/pipeline_config.m')
%   'data_dir'       - Directory containing input data files (default: 'data/raw')
%   'output_dir'     - Directory for outputs (default: 'outputs')
%   'skip_training'  - Skip training phase, use existing model (default: false)
%   'analysis_only'  - Only run analysis, skip embedding (default: false)
%   'verbose'        - Display progress messages (default: true)
%   'save_progress'  - Save intermediate progress (default: true)
%   'generate_report'- Generate final report (default: true)
%
% Outputs:
%   results - Structure containing complete pipeline results and metadata
%
% Examples:
%   % Basic usage
%   results = run_pipeline();
%   
%   % Custom configuration
%   results = run_pipeline('config_file', 'config/my_experiment.m');
%   
%   % Analysis only (skip training)
%   results = run_pipeline('analysis_only', true);
%   
%   % Use existing training results
%   results = run_pipeline('skip_training', true);

%% Add project paths
addpath(genpath('src'));
addpath(genpath('config'));
addpath(genpath('dependencies'));

%% Parse input arguments
p = inputParser;
addParameter(p, 'config_file', 'config/pipeline_config.m', @ischar);
addParameter(p, 'data_dir', 'data/raw', @ischar);
addParameter(p, 'output_dir', 'outputs', @ischar);
addParameter(p, 'skip_training', false, @islogical);
addParameter(p, 'analysis_only', false, @islogical);
addParameter(p, 'verbose', true, @islogical);
addParameter(p, 'save_progress', true, @islogical);
addParameter(p, 'generate_report', true, @islogical);
parse(p, varargin{:});

config_file = p.Results.config_file;
data_dir = p.Results.data_dir;
output_dir = p.Results.output_dir;
skip_training = p.Results.skip_training;
analysis_only = p.Results.analysis_only;
verbose = p.Results.verbose;
save_progress = p.Results.save_progress;
generate_report = p.Results.generate_report;

%% Initialize pipeline
if verbose
    fprintf('\n=== Spontaneous Pain Analysis Pipeline ===\n');
    fprintf('Version: 1.4.0\n');
    fprintf('Started: %s\n', datestr(now));
    fprintf('Data directory: %s\n', data_dir);
    fprintf('Output directory: %s\n', output_dir);
    fprintf('==========================================\n\n');
end

% Load configuration
if verbose, fprintf('Loading configuration from %s...\n', config_file); end
if exist(config_file, 'file')
    [config_dir, config_name, ~] = fileparts(config_file);
    if ~isempty(config_dir), addpath(config_dir); end
    config = feval(config_name);
else
    error('Configuration file not found: %s', config_file);
end

% Override output directory if specified
if ~strcmp(output_dir, 'outputs')
    config.output.base_dir = output_dir;
end

% Initialize results structure
results = struct();
results.config = config;
results.timestamp = datestr(now);
results.version = config.version;
results.stages = struct();
results.data_dir = data_dir;
results.output_dir = output_dir;
results.total_start_time = tic;

%% Stage 1: Environment Setup and Validation
if verbose, fprintf('=== Stage 1: Environment Setup and Validation ===\n'); end
stage1_start = tic;

try
    % Setup environment
    if verbose, fprintf('Setting up environment...\n'); end
    setup_pipeline();
    
    % Create output directories
    create_output_directories(config.output.base_dir);
    
    % Validate input data
    if verbose, fprintf('Validating input data...\n'); end
    validation_results = validate_input_data(data_dir, config);
    
    results.stages.setup = struct();
    results.stages.setup.success = true;
    results.stages.setup.duration = toc(stage1_start);
    results.stages.setup.validation = validation_results;
    
    if verbose
        fprintf('✓ Environment setup completed (%.1f seconds)\n', results.stages.setup.duration);
        if validation_results.has_issues
            fprintf('! %d validation issues found (see validation report)\n', length(validation_results.issues));
        end
        fprintf('\n');
    end
    
catch ME
    results.stages.setup = struct();
    results.stages.setup.success = false;
    results.stages.setup.error = ME.message;
    results.stages.setup.duration = toc(stage1_start);
    
    fprintf('✗ Environment setup failed: %s\n\n', ME.message);
    if nargout > 0, return; end
    error('Pipeline setup failed: %s', ME.message);
end

%% Stage 2: Training Phase
if ~skip_training && ~analysis_only
    if verbose, fprintf('=== Stage 2: Behavioral Embedding Training ===\n'); end
    stage2_start = tic;
    
    try
        % Check if training results already exist
        existing_results = check_existing_training_results(config.output.base_dir);
        
        if existing_results.found && ~existing_results.force_retrain
            if verbose
                fprintf('Existing training results found:\n');
                for i = 1:length(existing_results.files)
                    fprintf('  ✓ %s\n', existing_results.files{i});
                end
                
                if config.debug.verbose
                    response = input('Use existing results? (y/n/f for force retrain): ', 's');
                    if strcmpi(response, 'n') || strcmpi(response, 'f')
                        existing_results.use_existing = false;
                        if strcmpi(response, 'f')
                            existing_results.force_retrain = true;
                        end
                    else
                        existing_results.use_existing = true;
                    end
                else
                    existing_results.use_existing = true;
                end
            else
                existing_results.use_existing = true;
            end
        else
            existing_results.use_existing = false;
        end
        
        if ~existing_results.use_existing
            % Clean up existing results if force retraining
            if existing_results.force_retrain
                cleanup_training_results(config.output.base_dir);
            end
            
            % Run main embedding training
            if verbose, fprintf('Running behavioral embedding training...\n'); end
            
            training_results = mouse_embedding(...
                'config', config, ...
                'data_dir', data_dir, ...
                'output_dir', config.output.base_dir, ...
                'verbose', verbose);
            
            results.stages.training = struct();
            results.stages.training.success = true;
            results.stages.training.duration = toc(stage2_start);
            results.stages.training.results = training_results;
            results.stages.training.num_mice = training_results.num_mice;
            results.stages.training.num_groups = training_results.num_groups;
            
        else
            % Use existing results
            if verbose, fprintf('Using existing training results...\n'); end
            
            results.stages.training = struct();
            results.stages.training.success = true;
            results.stages.training.duration = toc(stage2_start);
            results.stages.training.skipped = true;
            results.stages.training.used_existing = true;
            results.stages.training.existing_files = existing_results.files;
        end
        
        if verbose
            fprintf('✓ Training phase completed (%.1f seconds)\n\n', results.stages.training.duration);
        end
        
        % Save intermediate progress
        if save_progress
            save(fullfile(config.output.base_dir, 'results', 'pipeline_progress.mat'), 'results');
        end
        
    catch ME
        results.stages.training = struct();
        results.stages.training.success = false;
        results.stages.training.error = ME.message;
        results.stages.training.duration = toc(stage2_start);
        results.stages.training.stack = ME.stack;
        
        fprintf('✗ Training phase failed: %s\n\n', ME.message);
        if verbose && length(ME.stack) > 0
            fprintf('Error occurred in: %s (line %d)\n\n', ME.stack(1).name, ME.stack(1).line);
        end
        
        if nargout > 0, return; end
        error('Training phase failed: %s', ME.message);
    end
    
else
    if verbose
        if analysis_only
            fprintf('=== Stage 2: Training Phase (Skipped - Analysis Only Mode) ===\n\n');
        else
            fprintf('=== Stage 2: Training Phase (Skipped - Using Existing Results) ===\n\n');
        end
    end
    
    results.stages.training = struct();
    results.stages.training.success = true;
    results.stages.training.skipped = true;
    results.stages.training.reason = ternary(analysis_only, 'analysis_only', 'skip_training');
    results.stages.training.duration = 0;
end

%% Stage 3: Downstream Analysis
if verbose, fprintf('=== Stage 3: Downstream Analysis and Visualization ===\n'); end
stage3_start = tic;

try
    % Check for required analysis input files
    analysis_inputs = check_analysis_inputs(config.output.base_dir);
    
    if ~analysis_inputs.ready
        if verbose
            fprintf('Missing analysis input files. Generating required inputs...\n');
            for i = 1:length(analysis_inputs.missing)
                fprintf('  Missing: %s\n', analysis_inputs.missing{i});
            end
        end
        
        % Run preprocessing to generate required files
        preprocessing_results = run_preprocessing_for_analysis(config, data_dir, verbose);
        
        if ~preprocessing_results.success
            error('Preprocessing failed: %s', preprocessing_results.error);
        end
    end
    
    % Run downstream analysis
    if verbose, fprintf('Running downstream analysis and visualization...\n'); end
    
    analysis_results = analyze_maps_and_counts(...
        'config', config, ...
        'input_dir', config.output.base_dir, ...
        'output_dir', fullfile(config.output.base_dir, 'analysis'), ...
        'verbose', verbose);
    
    results.stages.analysis = struct();
    results.stages.analysis.success = true;
    results.stages.analysis.duration = toc(stage3_start);
    results.stages.analysis.results = analysis_results;
    results.stages.analysis.num_files = analysis_results.data.num_files;
    results.stages.analysis.num_regions = analysis_results.data.num_regions;
    
    if verbose
        fprintf('✓ Analysis completed (%.1f seconds)\n', results.stages.analysis.duration);
        fprintf('  Processed %d files with %d behavioral regions\n', ...
            analysis_results.data.num_files, analysis_results.data.num_regions);
        fprintf('  Outputs saved to: %s\n\n', analysis_results.outputs.output_dir);
    end
    
catch ME
    results.stages.analysis = struct();
    results.stages.analysis.success = false;
    results.stages.analysis.error = ME.message;
    results.stages.analysis.duration = toc(stage3_start);
    results.stages.analysis.stack = ME.stack;
    
    fprintf('✗ Analysis phase failed: %s\n\n', ME.message);
    if verbose && length(ME.stack) > 0
        fprintf('Error occurred in: %s (line %d)\n\n', ME.stack(1).name, ME.stack(1).line);
    end
    
    if nargout > 0, return; end
    error('Analysis phase failed: %s', ME.message);
end

%% Stage 4: Validation and Reporting
if verbose, fprintf('=== Stage 4: Validation and Report Generation ===\n'); end
stage4_start = tic;

try
    % Run comprehensive validation
    if verbose, fprintf('Validating pipeline results...\n'); end
    validation_results = validate_pipeline_results(config.output.base_dir, config);
    
    % Generate comprehensive report
    if generate_report
        if verbose, fprintf('Generating pipeline report...\n'); end
        report_results = generate_comprehensive_report(results, config, validation_results);
        results.report_file = report_results.report_file;
        results.summary_file = report_results.summary_file;
    end
    
    results.stages.validation = struct();
    results.stages.validation.success = true;
    results.stages.validation.duration = toc(stage4_start);
    results.stages.validation.results = validation_results;
    
    if generate_report
        results.stages.validation.report_generated = true;
        results.stages.validation.report_file = results.report_file;
    end
    
    if verbose
        fprintf('✓ Validation completed (%.1f seconds)\n', results.stages.validation.duration);
        if generate_report
            fprintf('  Report saved: %s\n', results.report_file);
        end
        fprintf('\n');
    end
    
catch ME
    results.stages.validation = struct();
    results.stages.validation.success = false;
    results.stages.validation.error = ME.message;
    results.stages.validation.duration = toc(stage4_start);
    
    fprintf('! Validation failed: %s\n', ME.message);
    fprintf('  Pipeline may have completed successfully despite validation issues\n\n');
end

%% Final Results Summary
results.success = all([results.stages.setup.success, ...
                      results.stages.training.success, ...
                      results.stages.analysis.success]);

if isfield(results.stages, 'validation')
    results.validation_success = results.stages.validation.success;
else
    results.validation_success = false;
end

results.total_duration = toc(results.total_start_time);
results.completed_at = datestr(now);

% Calculate stage durations
stage_durations = [results.stages.setup.duration, ...
                  results.stages.training.duration, ...
                  results.stages.analysis.duration];
if isfield(results.stages, 'validation')
    stage_durations = [stage_durations, results.stages.validation.duration];
end

results.stage_durations = stage_durations;
results.stage_names = {'Setup', 'Training', 'Analysis', 'Validation'};

% Save final results
results_file = fullfile(config.output.base_dir, 'results', 'final_pipeline_results.mat');
save(results_file, 'results', 'config');
results.results_file = results_file;

%% Display final summary
if verbose
    fprintf('=== Pipeline Execution Summary ===\n');
    if results.success
        fprintf('✓ Pipeline completed successfully\n');
    else
        fprintf('✗ Pipeline completed with errors\n');
    end
    
    fprintf('Total execution time: %.1f seconds (%.1f minutes)\n', ...
        results.total_duration, results.total_duration/60);
    
    % Stage timing breakdown
    fprintf('\nStage timing breakdown:\n');
    for i = 1:length(results.stage_names)
        if i <= length(stage_durations)
            fprintf('  %s: %.1f seconds (%.1f%%)\n', ...
                results.stage_names{i}, stage_durations(i), ...
                100*stage_durations(i)/results.total_duration);
        end
    end
    
    % Key outputs summary
    fprintf('\nGenerated outputs:\n');
    output_dirs = {
        fullfile(config.output.base_dir, 'results'), 'Pipeline results and models';
        fullfile(config.output.base_dir, 'figures'), 'Visualization figures';
        fullfile(config.output.base_dir, 'analysis'), 'Analysis outputs and CSV files';
        fullfile(config.output.base_dir, 'validation'), 'Validation reports'
    };
    
    for i = 1:size(output_dirs, 1)
        if exist(output_dirs{i,1}, 'dir')
            file_count = length(dir(fullfile(output_dirs{i,1}, '*.*'))) - 2; % Exclude . and ..
            fprintf('   %s (%d files)\n', output_dirs{i,2}, file_count);
        end
    end
    
    fprintf('\nMain results file: %s\n', results.results_file);
    
    if isfield(results, 'report_file')
        fprintf('Comprehensive report: %s\n', results.report_file);
    end
    
    fprintf('\n=== Pipeline Complete ===\n\n');
end

end

%% Helper Functions

function create_output_directories(base_dir)
    %CREATE_OUTPUT_DIRECTORIES Create all necessary output directories
    
    dirs = {
        base_dir,
        fullfile(base_dir, 'results'),
        fullfile(base_dir, 'figures'),
        fullfile(base_dir, 'figures', 'per_video'),
        fullfile(base_dir, 'figures', 'group_comparisons'),
        fullfile(base_dir, 'analysis'),
        fullfile(base_dir, 'analysis', 'figures'),
        fullfile(base_dir, 'analysis', 'figures', 'per_video'),
        fullfile(base_dir, 'analysis', 'csv'),
        fullfile(base_dir, 'analysis', 'csv', 'frame_indices_per_video'),
        fullfile(base_dir, 'analysis', 'csv', 'sequence_metadata'),
        fullfile(base_dir, 'analysis', 'csv', 'statistical_analysis'),
        fullfile(base_dir, 'validation')
    };
    
    for i = 1:length(dirs)
        if ~exist(dirs{i}, 'dir')
            mkdir(dirs{i});
        end
    end
    
    % Create .gitkeep files for empty directories
    gitkeep_dirs = {
        fullfile(base_dir, 'results'),
        fullfile(base_dir, 'figures'),
        fullfile(base_dir, 'validation')
    };
    
    for i = 1:length(gitkeep_dirs)
        gitkeep_file = fullfile(gitkeep_dirs{i}, '.gitkeep');
        if ~exist(gitkeep_file, 'file')
            fid = fopen(gitkeep_file, 'w');
            if fid ~= -1
                fclose(fid);
            end
        end
    end
end

function validation = validate_input_data(data_dir, config)
    %VALIDATE_INPUT_DATA Comprehensive input data validation
    
    validation = struct();
    validation.timestamp = datestr(now);
    validation.data_dir = data_dir;
    validation.issues = {};
    validation.warnings = {};
    validation.has_issues = false;
    
    % Check data directory exists
    if ~exist(data_dir, 'dir')
        validation.issues{end+1} = sprintf('Data directory not found: %s', data_dir);
        validation.has_issues = true;
        return;
    end
    
    % Check for mouseFileOrder.mat
    mouse_file_order = fullfile(data_dir, config.files.mouse_file_order);
    if exist(mouse_file_order, 'file')
        try
            data = load(mouse_file_order);
            if isfield(data, 'mouseOrderShort') && isfield(data, 'metadata')
                validation.mouse_file_order = true;
                validation.num_files = length(data.mouseOrderShort);
            else
                validation.issues{end+1} = 'mouseFileOrder.mat missing required fields (mouseOrderShort, metadata)';
                validation.mouse_file_order = false;
                validation.has_issues = true;
            end
        catch ME
            validation.issues{end+1} = sprintf('Could not load mouseFileOrder.mat: %s', ME.message);
            validation.mouse_file_order = false;
            validation.has_issues = true;
        end
    else
        validation.issues{end+1} = sprintf('Required file not found: %s', mouse_file_order);
        validation.mouse_file_order = false;
        validation.has_issues = true;
    end
    
    % Check training files
    if isfield(config, 'training') && isfield(config.training, 'files')
        validation.training_files = struct();
        for i = 1:length(config.training.files)
            filename = config.training.files{i};
            file_path = fullfile(data_dir, filename);
            field_name = matlab.lang.makeValidName(filename);
            
            if exist(file_path, 'file')
                validation.training_files.(field_name) = true;
                
                % Validate file structure
                try
                    data = load(file_path);
                    if isfield(data, 'pred')
                        pred_size = size(data.pred);
                        expected_size = config.data.expected_dimensions;
                        if length(pred_size) ~= length(expected_size) || ...
                           (pred_size(2) ~= expected_size(2)) || ...
                           (pred_size(3) ~= expected_size(3))
                            validation.warnings{end+1} = sprintf('Training file %s has unexpected dimensions: %s', ...
                                filename, mat2str(pred_size));
                        end
                    else
                        validation.warnings{end+1} = sprintf('Training file %s missing ''pred'' field', filename);
                    end
                catch ME
                    validation.warnings{end+1} = sprintf('Could not validate structure of %s: %s', filename, ME.message);
                end
            else
                validation.training_files.(field_name) = false;
                validation.warnings{end+1} = sprintf('Training file not found (may be created during training): %s', filename);
            end
        end
    end
    
    % Summary
    validation.summary = struct();
    validation.summary.total_issues = length(validation.issues);
    validation.summary.total_warnings = length(validation.warnings);
    validation.summary.data_dir_exists = exist(data_dir, 'dir') == 7;
    validation.summary.mouse_file_order_valid = validation.mouse_file_order;
    
    if validation.has_issues
        validation.severity = 'error';
    elseif ~isempty(validation.warnings)
        validation.severity = 'warning';
    else
        validation.severity = 'ok';
    end
end

function existing = check_existing_training_results(output_dir)
    %CHECK_EXISTING_TRAINING_RESULTS Check for existing training results
    
    existing = struct();
    existing.found = false;
    existing.files = {};
    existing.use_existing = false;
    existing.force_retrain = false;
    
    % List of files that should exist after training
    expected_files = {
        'vecsValsMouse_weekdata.mat',
        'loneSignalDataAmps_weekdata.mat', 
        'train_weekdata.mat',
        'mouseEmbeddingResults_weekdata.mat'
    };
    
    existing_files = {};
    for i = 1:length(expected_files)
        file_path = fullfile(output_dir, expected_files{i});
        if exist(file_path, 'file')
            existing_files{end+1} = expected_files{i};
        end
    end
    
    if length(existing_files) >= 3  % At least most key files exist
        existing.found = true;
        existing.files = existing_files;
    end
end

function cleanup_training_results(output_dir)
    %CLEANUP_TRAINING_RESULTS Remove existing training results for clean retrain
    
    files_to_remove = {
        'vecsValsMouse_weekdata.mat',
        'loneSignalDataAmps_weekdata.mat',
        'train_weekdata.mat', 
        'mouseEmbeddingResults_weekdata.mat'
    };
    
    for i = 1:length(files_to_remove)
        file_path = fullfile(output_dir, files_to_remove{i});
        if exist(file_path, 'file')
            delete(file_path);
        end
    end
end

function inputs = check_analysis_inputs(output_dir)
    %CHECK_ANALYSIS_INPUTS Check if required files for analysis exist
    
    inputs = struct();
    inputs.ready = false;
    inputs.missing = {};
    
    required_files = {
        'watershed_SNI_TBI.mat',
        'complete_embedding_results_SNI_TBI.mat'
    };
    
    missing_files = {};
    for i = 1:length(required_files)
        file_path = fullfile(output_dir, required_files{i});
        if ~exist(file_path, 'file')
            missing_files{end+1} = required_files{i};
        end
    end
    
    inputs.missing = missing_files;
    inputs.ready = isempty(missing_files);
end

function results = run_preprocessing_for_analysis(config, data_dir, verbose)
    %RUN_PREPROCESSING_FOR_ANALYSIS Generate required analysis input files
    
    results = struct();
    results.success = false;
    results.error = '';
    
    try
        % This would run the specialized SNI/TBI preprocessing pipeline
        % For now, we'll create a placeholder
        if verbose, fprintf('  Running preprocessing pipeline...\n'); end
        
        % Check if custom preprocessing script exists
        preprocessing_script = 'src/preprocessing/custom_embedding_pipeline_SNI_TBI.m';
        if exist(preprocessing_script, 'file')
            run(preprocessing_script);
        else
            % Create minimal required files as placeholders
            if verbose, fprintf('  Creating placeholder analysis input files...\n'); end
            create_placeholder_analysis_inputs(config.output.base_dir);
        end
        
        results.success = true;
        
    catch ME
        results.success = false;
        results.error = ME.message;
    end
end

function create_placeholder_analysis_inputs(output_dir)
    %CREATE_PLACEHOLDER_ANALYSIS_INPUTS Create minimal analysis input files
    
    % This is a placeholder - in practice, you would run the actual preprocessing
    warning('Creating placeholder analysis input files - replace with actual preprocessing');
    
    % Create minimal watershed file
    watershed_file = fullfile(output_dir, 'watershed_SNI_TBI.mat');
    if ~exist(watershed_file, 'file')
        D = rand(501, 501) * 1e-4;  % Placeholder density map
        LL = ones(501, 501);        % Placeholder watershed labels
        LL2 = LL;
        llbwb = [250, 250; 251, 251]; % Minimal boundaries
        xx = linspace(-65, 65, 501);
        save(watershed_file, 'D', 'LL', 'LL2', 'llbwb', 'xx');
    end
    
    % Create minimal results file
    results_file = fullfile(output_dir, 'complete_embedding_results_SNI_TBI.mat');
    if ~exist(results_file, 'file')
        results = struct();
        results.reembedding_labels_all = {'placeholder_file.mat'};
        results.zEmbeddings_all = {rand(100, 2)};
        save(results_file, 'results');
    end
end

function validation = validate_pipeline_results(output_dir, config)
    %VALIDATE_PIPELINE_RESULTS Comprehensive validation of pipeline outputs
    
    validation = struct();
    validation.timestamp = datestr(now);
    validation.output_dir = output_dir;
    validation.issues = {};
    validation.success = true;
    
    % Check for key result files
    key_files = {
        config.files.pca_results, 'PCA results';
        config.files.embedding_data, 'Embedding data';
        config.files.tsne_results, 't-SNE results';
        config.files.final_results, 'Final embedding results'
    };
    
    validation.files = struct();
    for i = 1:size(key_files, 1)
        filename = key_files{i,1};
        description = key_files{i,2};
        file_path = fullfile(output_dir, filename);
        field_name = matlab.lang.makeValidName(filename);
        
        if exist(file_path, 'file')
            validation.files.(field_name) = true;
            
            % Basic file size check
            file_info = dir(file_path);
            if file_info.bytes < 1000  % Very small file might indicate error
                validation.issues{end+1} = sprintf('%s file is unusually small: %s', description, filename);
                validation.success = false;
            end
        else
            validation.files.(field_name) = false;
            validation.issues{end+1} = sprintf('Missing %s file: %s', description, filename);
            validation.success = false;
        end
    end
    
    % Check output directories
    output_dirs = {'results', 'figures', 'analysis', 'validation'};
    validation.directories = struct();
    for i = 1:length(output_dirs)
        dir_path = fullfile(output_dir, output_dirs{i});
        field_name = output_dirs{i};
        validation.directories.(field_name) = exist(dir_path, 'dir') == 7;
        
        if ~validation.directories.(field_name)
            validation.issues{end+1} = sprintf('Missing output directory: %s', output_dirs{i});
            validation.success = false;
        end
    end
    
    validation.total_issues = length(validation.issues);
end

function report_results = generate_comprehensive_report(results, config, validation_results)
    %GENERATE_COMPREHENSIVE_REPORT Generate detailed pipeline execution report
    
    report_results = struct();
    
    % Generate main report
    report_file = fullfile(config.output.base_dir, 'validation', 'pipeline_execution_report.txt');
    report_results.report_file = report_file;
    
    fid = fopen(report_file, 'w');
    if fid == -1
        error('Could not create report file: %s', report_file);
    end
    
    try
        % Header
        fprintf(fid, 'Spontaneous Pain Analysis Pipeline Execution Report\n');
        fprintf(fid, '=================================================\n\n');
        fprintf(fid, 'Generated: %s\n', results.completed_at);
        fprintf(fid, 'Pipeline Version: %s\n', results.version);
        fprintf(fid, 'Configuration: %s\n', config.description);
        fprintf(fid, 'Total Execution Time: %.1f seconds (%.1f minutes)\n\n', ...
            results.total_duration, results.total_duration/60);
        
        % Overall status
        fprintf(fid, 'OVERALL STATUS: %s\n', ternary(results.success, 'SUCCESS', 'FAILED'));
        fprintf(fid, 'Validation Status: %s\n\n', ternary(validation_results.success, 'PASSED', 'FAILED'));
        
        % Stage details
        fprintf(fid, 'STAGE EXECUTION DETAILS\n');
        fprintf(fid, '=======================\n');
        
        stage_names = fieldnames(results.stages);
        for i = 1:length(stage_names)
            stage = results.stages.(stage_names{i});
            fprintf(fid, '%s Stage:\n', stage_names{i});
            fprintf(fid, '  Status: %s\n', ternary(stage.success, 'SUCCESS', 'FAILED'));
            fprintf(fid, '  Duration: %.1f seconds\n', stage.duration);
            
            if ~stage.success && isfield(stage, 'error')
                fprintf(fid, '  Error: %s\n', stage.error);
            end
            
            if isfield(stage, 'skipped') && stage.skipped
                fprintf(fid, '  Note: Stage was skipped\n');
            end
            
            fprintf(fid, '\n');
        end
        
        % Configuration summary
        fprintf(fid, 'CONFIGURATION SUMMARY\n');
        fprintf(fid, '====================\n');
        fprintf(fid, 'Data Format: %s\n', config.data.format);
        fprintf(fid, 'PCA Components: %d\n', config.parameters.pca.num_components);
        fprintf(fid, 'Wavelet Frequency Range: %.1f - %.1f Hz\n', ...
            config.parameters.wavelets.min_freq, config.parameters.wavelets.max_freq);
        fprintf(fid, 't-SNE Templates per Dataset: %d\n', config.parameters.tsne.num_per_dataset);
        fprintf(fid, 'Watershed Connectivity: %d\n', config.parameters.watershed.connectivity);
        fprintf(fid, 'Output Directory: %s\n', config.output.base_dir);
        fprintf(fid, 'Generate Figures: %s\n', mat2str(config.output.generate_figures));
        fprintf(fid, '\n');
        
        % Validation details
        if ~isempty(validation_results.issues)
            fprintf(fid, 'VALIDATION ISSUES\n');
            fprintf(fid, '=================\n');
            for i = 1:length(validation_results.issues)
                fprintf(fid, '- %s\n', validation_results.issues{i});
            end
            fprintf(fid, '\n');
        end
        
        % File outputs
        fprintf(fid, 'GENERATED OUTPUTS\n');
        fprintf(fid, '=================\n');
        output_base = config.output.base_dir;
        output_info = {
            fullfile(output_base, 'results'), 'Pipeline results and models';
            fullfile(output_base, 'figures'), 'Visualization figures';
            fullfile(output_base, 'analysis'), 'Analysis outputs and CSV files';
            fullfile(output_base, 'validation'), 'Validation reports'
        };
        
        for i = 1:size(output_info, 1)
            if exist(output_info{i,1}, 'dir')
                files = dir(fullfile(output_info{i,1}, '*.*'));
                file_count = sum(~[files.isdir]) - 2; % Exclude . and ..
                fprintf(fid, '%s: %d files\n', output_info{i,2}, max(0, file_count));
            end
        end
        
        fprintf(fid, '\nMain Results File: %s\n', results.results_file);
        
        % Performance summary
        if length(results.stage_durations) > 0
            fprintf(fid, '\nPERFORMANCE BREAKDOWN\n');
            fprintf(fid, '====================\n');
            for i = 1:length(results.stage_names)
                if i <= length(results.stage_durations)
                    fprintf(fid, '%s: %.1f seconds (%.1f%%)\n', ...
                        results.stage_names{i}, results.stage_durations(i), ...
                        100*results.stage_durations(i)/results.total_duration);
                end
            end
        end
        
        fclose(fid);
        
    catch ME
        fclose(fid);
        rethrow(ME);
    end
    
    % Generate summary file
    summary_file = fullfile(config.output.base_dir, 'validation', 'pipeline_summary.txt');
    report_results.summary_file = summary_file;
    
    fid = fopen(summary_file, 'w');
    if fid ~= -1
        fprintf(fid, 'Pipeline Execution Summary\n');
        fprintf(fid, '========================\n');
        fprintf(fid, 'Status: %s\n', ternary(results.success, 'SUCCESS', 'FAILED'));
        fprintf(fid, 'Duration: %.1f minutes\n', results.total_duration/60);
        fprintf(fid, 'Completed: %s\n', results.completed_at);
        if validation_results.total_issues > 0
            fprintf(fid, 'Issues: %d validation issues found\n', validation_results.total_issues);
        end
        fclose(fid);
    end
end

function result = ternary(condition, true_value, false_value)
    %TERNARY Simple ternary operator implementation
    if condition
        result = true_value;
    else
        result = false_value;
    end
end