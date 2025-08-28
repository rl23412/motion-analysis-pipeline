function results = run_full_pipeline(varargin)
%% Run Full Pipeline - Complete spontaneous pain analysis pipeline
% This script runs the complete spontaneous pain behavioral analysis pipeline
% with proper error handling, validation, and progress tracking.
%
% Usage:
%   run_full_pipeline()                    % Use default configuration
%   run_full_pipeline('config', config)    % Use custom configuration
%   run_full_pipeline('skipTraining', true) % Skip training, use existing results
%   run_full_pipeline('analysisOnly', true) % Only run downstream analysis
%
% Outputs:
%   results - Structure containing pipeline results and metadata

%% Parse input arguments
p = inputParser;
addParameter(p, 'config', [], @isstruct);
addParameter(p, 'skipTraining', false, @islogical);
addParameter(p, 'analysisOnly', false, @islogical);
addParameter(p, 'verbose', true, @islogical);
addParameter(p, 'saveProgress', true, @islogical);
parse(p, varargin{:});

config = p.Results.config;
skipTraining = p.Results.skipTraining;
analysisOnly = p.Results.analysisOnly;
verbose = p.Results.verbose;
saveProgress = p.Results.saveProgress;

%% Initialize pipeline
if verbose
    fprintf('\n=== Spontaneous Pain Analysis Pipeline ===\n');
    fprintf('Started: %s\n\n', datestr(now));
end

% Load configuration if not provided
if isempty(config)
    if verbose, fprintf('Loading default configuration...\n'); end
    config = group_config();
end

% Initialize results structure
results = struct();
results.config = config;
results.timestamp = datestr(now);
results.version = config.version;
results.stages = struct();

%% Stage 1: Environment Setup and Validation
if verbose, fprintf('=== Stage 1: Environment Setup ===\n'); end
stage1_start = tic;

try
    % Setup environment (paths, directories, etc.)
    if verbose, fprintf('Setting up environment...\n'); end
    setup_pipeline();
    
    % Validate input data
    if verbose, fprintf('Validating input data...\n'); end
    validation_results = validate_input_data(config);
    
    results.stages.setup = struct();
    results.stages.setup.success = true;
    results.stages.setup.duration = toc(stage1_start);
    results.stages.setup.validation = validation_results;
    
    if verbose
        fprintf('✓ Environment setup completed (%.1f seconds)\n\n', results.stages.setup.duration);
    end
    
catch ME
    results.stages.setup = struct();
    results.stages.setup.success = false;
    results.stages.setup.error = ME.message;
    results.stages.setup.duration = toc(stage1_start);
    
    fprintf('✗ Environment setup failed: %s\n\n', ME.message);
    if nargout > 0, return; end
    error('Pipeline setup failed');
end

%% Stage 2: Training Phase (Skip if requested or analyzing existing results)
if ~skipTraining && ~analysisOnly
    if verbose, fprintf('=== Stage 2: Training Phase ===\n'); end
    stage2_start = tic;
    
    try
        % Run main embedding training
        if verbose, fprintf('Running main embedding pipeline (mouseEmbedding.m)...\n'); end
        
        % Check if training results already exist
        if exist('mouseEmbeddingResults_weekdata.mat', 'file') && ~skipTraining
            if verbose
                fprintf('Existing training results found.\n');
                response = input('Use existing results? (y/n): ', 's');
                if strcmpi(response, 'n')
                    skipTraining = false;
                else
                    skipTraining = true;
                end
            end
        end
        
        if ~skipTraining
            % Clear previous results to ensure clean training
            if exist('vecsValsMouse_weekdata.mat', 'file')
                delete('vecsValsMouse_weekdata.mat');
            end
            if exist('loneSignalDataAmps_weekdata.mat', 'file')
                delete('loneSignalDataAmps_weekdata.mat');
            end
            if exist('train_weekdata.mat', 'file')
                delete('train_weekdata.mat');
            end
            if exist('mouseEmbeddingResults_weekdata.mat', 'file')
                delete('mouseEmbeddingResults_weekdata.mat');
            end
            
            % Run the main embedding pipeline
            mouseEmbedding();
        end
        
        % Load and validate training results
        if exist('mouseEmbeddingResults_weekdata.mat', 'file')
            trainingData = load('mouseEmbeddingResults_weekdata.mat');
            results.stages.training = struct();
            results.stages.training.success = true;
            results.stages.training.duration = toc(stage2_start);
            results.stages.training.numMice = length(trainingData.wrFINE) / 2; % Divide by 2 for original + flipped
            results.stages.training.resultsFile = 'mouseEmbeddingResults_weekdata.mat';
            
            if verbose
                fprintf('✓ Training completed (%.1f seconds)\n', results.stages.training.duration);
                fprintf('  Processed %d mice (original + flipped)\n\n', results.stages.training.numMice);
            end
        else
            error('Training results not found after pipeline execution');
        end
        
        % Save intermediate progress
        if saveProgress
            save(fullfile('outputs', 'results', 'pipeline_progress.mat'), 'results');
        end
        
    catch ME
        results.stages.training = struct();
        results.stages.training.success = false;
        results.stages.training.error = ME.message;
        results.stages.training.duration = toc(stage2_start);
        
        fprintf('✗ Training phase failed: %s\n\n', ME.message);
        if nargout > 0, return; end
        error('Training phase failed');
    end
else
    if verbose
        if analysisOnly
            fprintf('=== Stage 2: Training Phase (Skipped - Analysis Only) ===\n\n');
        else
            fprintf('=== Stage 2: Training Phase (Skipped - Using Existing Results) ===\n\n');
        end
    end
    results.stages.training = struct();
    results.stages.training.success = true;
    results.stages.training.skipped = true;
    results.stages.training.duration = 0;
end

%% Stage 3: Downstream Analysis
if verbose, fprintf('=== Stage 3: Downstream Analysis ===\n'); end
stage3_start = tic;

try
    % Check for required analysis input files
    analysisInputs = {
        'watershed_SNI_TBI.mat', 'Watershed regions and density map';
        'complete_embedding_results_SNI_TBI.mat', 'Complete embedding results'
    };
    
    missingInputs = {};
    for i = 1:size(analysisInputs, 1)
        if ~exist(analysisInputs{i,1}, 'file')
            missingInputs{end+1} = analysisInputs{i,1};
        end
    end
    
    if ~isempty(missingInputs)
        if verbose
            fprintf('Missing analysis input files. Running preprocessing...\n');
            for i = 1:length(missingInputs)
                fprintf('  Missing: %s\n', missingInputs{i});
            end
        end
        
        % Run the specialized SNI/TBI pipeline to generate required files
        if exist('custom_embedding_pipeline_SNI_TBI.m', 'file')
            if verbose, fprintf('Running custom embedding pipeline for SNI/TBI...\n'); end
            custom_embedding_pipeline_SNI_TBI();
        else
            error('Required preprocessing script not found: custom_embedding_pipeline_SNI_TBI.m');
        end
    end
    
    % Run downstream analysis
    if verbose, fprintf('Running downstream analysis (analyze_saved_maps_and_counts.m)...\n'); end
    analyze_saved_maps_and_counts();
    
    % Validate analysis outputs
    expectedOutputs = {
        fullfile('analysis_outputs', 'figures', 'behavioral_map_with_indices.png');
        fullfile('analysis_outputs', 'csv', 'per_file_region_counts.csv');
        fullfile('analysis_outputs', 'csv', 'frame_indices_summary.csv')
    };
    
    outputsGenerated = 0;
    for i = 1:length(expectedOutputs)
        if exist(expectedOutputs{i}, 'file')
            outputsGenerated = outputsGenerated + 1;
        end
    end
    
    results.stages.analysis = struct();
    results.stages.analysis.success = true;
    results.stages.analysis.duration = toc(stage3_start);
    results.stages.analysis.outputsGenerated = outputsGenerated;
    results.stages.analysis.expectedOutputs = length(expectedOutputs);
    results.stages.analysis.outputDir = 'analysis_outputs';
    
    if verbose
        fprintf('✓ Analysis completed (%.1f seconds)\n', results.stages.analysis.duration);
        fprintf('  Generated %d/%d expected outputs\n\n', outputsGenerated, length(expectedOutputs));
    end
    
catch ME
    results.stages.analysis = struct();
    results.stages.analysis.success = false;
    results.stages.analysis.error = ME.message;
    results.stages.analysis.duration = toc(stage3_start);
    
    fprintf('✗ Analysis phase failed: %s\n\n', ME.message);
    if nargout > 0, return; end
    error('Analysis phase failed');
end

%% Stage 4: Validation and Reporting
if verbose, fprintf('=== Stage 4: Validation and Reporting ===\n'); end
stage4_start = tic;

try
    % Run pipeline validation
    if verbose, fprintf('Validating pipeline results...\n'); end
    validation = validate_pipeline_results();
    
    % Generate summary report
    if verbose, fprintf('Generating summary report...\n'); end
    reportFile = generate_pipeline_report(results);
    
    results.stages.validation = struct();
    results.stages.validation.success = true;
    results.stages.validation.duration = toc(stage4_start);
    results.stages.validation.results = validation;
    results.stages.validation.reportFile = reportFile;
    
    if verbose
        fprintf('✓ Validation completed (%.1f seconds)\n', results.stages.validation.duration);
        fprintf('  Report saved: %s\n\n', reportFile);
    end
    
catch ME
    results.stages.validation = struct();
    results.stages.validation.success = false;
    results.stages.validation.error = ME.message;
    results.stages.validation.duration = toc(stage4_start);
    
    fprintf('! Validation failed: %s\n', ME.message);
    fprintf('  Pipeline may have completed but validation could not be performed\n\n');
end

%% Final Results and Summary
results.success = all([results.stages.setup.success, ...
                      results.stages.training.success, ...
                      results.stages.analysis.success]);

if isfield(results.stages, 'validation')
    results.success = results.success && results.stages.validation.success;
end

results.totalDuration = sum([results.stages.setup.duration, ...
                            results.stages.training.duration, ...
                            results.stages.analysis.duration]);

if isfield(results.stages, 'validation')
    results.totalDuration = results.totalDuration + results.stages.validation.duration;
end

results.completedAt = datestr(now);

% Save final results
resultsFile = fullfile('outputs', 'results', 'pipeline_results.mat');
save(resultsFile, 'results');

if verbose
    fprintf('=== Pipeline Summary ===\n');
    if results.success
        fprintf('✓ Pipeline completed successfully\n');
    else
        fprintf('✗ Pipeline completed with errors\n');
    end
    fprintf('Total duration: %.1f seconds (%.1f minutes)\n', results.totalDuration, results.totalDuration/60);
    fprintf('Results saved: %s\n', resultsFile);
    
    % Display key outputs
    fprintf('\nKey outputs:\n');
    if exist('analysis_outputs', 'dir')
        fprintf('  📊 Analysis outputs: analysis_outputs/\n');
        fprintf('  📈 Figures: analysis_outputs/figures/\n');
        fprintf('  📋 CSV files: analysis_outputs/csv/\n');
    end
    if exist('outputs', 'dir')
        fprintf('  📁 Pipeline results: outputs/results/\n');
        fprintf('  ✅ Validation: outputs/validation/\n');
    end
    
    fprintf('\n=== Pipeline Complete ===\n\n');
end

end

%% Helper Functions

function validation = validate_input_data(config)
%VALIDATE_INPUT_DATA Validate input data files and format
    validation = struct();
    validation.success = true;
    validation.issues = {};
    
    % Check for mouseFileOrder.mat
    if exist('mouseFileOrder.mat', 'file')
        try
            data = load('mouseFileOrder.mat');
            if isfield(data, 'mouseOrderShort') && isfield(data, 'metadata')
                validation.mouseFileOrder = true;
            else
                validation.mouseFileOrder = false;
                validation.issues{end+1} = 'mouseFileOrder.mat missing required fields';
                validation.success = false;
            end
        catch
            validation.mouseFileOrder = false;
            validation.issues{end+1} = 'Could not load mouseFileOrder.mat';
            validation.success = false;
        end
    else
        validation.mouseFileOrder = false;
        validation.issues{end+1} = 'mouseFileOrder.mat not found';
        validation.success = false;
    end
    
    % Check training files
    validation.trainingFiles = struct();
    for i = 1:length(config.training.files)
        filename = config.training.files{i};
        if exist(filename, 'file')
            validation.trainingFiles.(matlab.lang.makeValidName(filename)) = true;
        else
            validation.trainingFiles.(matlab.lang.makeValidName(filename)) = false;
            validation.issues{end+1} = sprintf('Training file not found: %s', filename);
        end
    end
end

function validation = validate_pipeline_results()
%VALIDATE_PIPELINE_RESULTS Validate that pipeline generated expected outputs
    validation = struct();
    validation.timestamp = datestr(now);
    
    % Check for key result files
    keyFiles = {
        'mouseEmbeddingResults_weekdata.mat', 'Main embedding results';
        'watershed_SNI_TBI.mat', 'Watershed regions';
        'complete_embedding_results_SNI_TBI.mat', 'Complete results'
    };
    
    validation.files = struct();
    for i = 1:size(keyFiles, 1)
        filename = keyFiles{i,1};
        validation.files.(matlab.lang.makeValidName(filename)) = exist(filename, 'file') == 2;
    end
    
    % Check analysis outputs
    validation.analysisOutputs = struct();
    validation.analysisOutputs.figuresDir = exist(fullfile('analysis_outputs', 'figures'), 'dir') == 7;
    validation.analysisOutputs.csvDir = exist(fullfile('analysis_outputs', 'csv'), 'dir') == 7;
    validation.analysisOutputs.mainFigure = exist(fullfile('analysis_outputs', 'figures', 'behavioral_map_with_indices.png'), 'file') == 2;
    validation.analysisOutputs.countsCsv = exist(fullfile('analysis_outputs', 'csv', 'per_file_region_counts.csv'), 'file') == 2;
    
    validation.success = all(struct2array(validation.files)) && ...
                        all(struct2array(validation.analysisOutputs));
end

function reportFile = generate_pipeline_report(results)
%GENERATE_PIPELINE_REPORT Generate a summary report of pipeline execution
    reportFile = fullfile('outputs', 'validation', 'pipeline_report.txt');
    
    fid = fopen(reportFile, 'w');
    if fid == -1
        error('Could not create report file: %s', reportFile);
    end
    
    try
        % Header
        fprintf(fid, 'Spontaneous Pain Analysis Pipeline Report\n');
        fprintf(fid, '==========================================\n\n');
        fprintf(fid, 'Generated: %s\n', results.completedAt);
        fprintf(fid, 'Version: %s\n', results.version);
        fprintf(fid, 'Total Duration: %.1f seconds (%.1f minutes)\n\n', results.totalDuration, results.totalDuration/60);
        
        % Overall success
        fprintf(fid, 'Overall Status: %s\n\n', ternary(results.success, 'SUCCESS', 'FAILED'));
        
        % Stage details
        fprintf(fid, 'Stage Details:\n');
        fprintf(fid, '--------------\n');
        
        stages = fieldnames(results.stages);
        for i = 1:length(stages)
            stage = results.stages.(stages{i});
            fprintf(fid, '%s: %s (%.1f seconds)\n', stages{i}, ...
                ternary(stage.success, 'SUCCESS', 'FAILED'), stage.duration);
            
            if ~stage.success && isfield(stage, 'error')
                fprintf(fid, '  Error: %s\n', stage.error);
            end
        end
        
        fprintf(fid, '\n');
        
        % Configuration summary
        fprintf(fid, 'Configuration Summary:\n');
        fprintf(fid, '---------------------\n');
        if isfield(results.config, 'groups')
            weeks = fieldnames(results.config.groups);
            totalGroups = 0;
            for w = 1:length(weeks)
                groupNames = fieldnames(results.config.groups.(weeks{w}));
                totalGroups = totalGroups + length(groupNames);
            end
            fprintf(fid, 'Total Groups: %d\n', totalGroups);
            fprintf(fid, 'Training Files: %s\n', strjoin(results.config.training.files, ', '));
        end
        
        % File outputs
        fprintf(fid, '\nGenerated Outputs:\n');
        fprintf(fid, '------------------\n');
        if exist('analysis_outputs', 'dir')
            fprintf(fid, 'Analysis outputs directory: analysis_outputs/\n');
        end
        if exist('outputs', 'dir')
            fprintf(fid, 'Pipeline results directory: outputs/\n');
        end
        
        fclose(fid);
        
    catch ME
        fclose(fid);
        rethrow(ME);
    end
end

function result = ternary(condition, trueValue, falseValue)
%TERNARY Simple ternary operator implementation
    if condition
        result = trueValue;
    else
        result = falseValue;
    end
end