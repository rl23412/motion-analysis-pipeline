%% Basic Usage Example - Spontaneous Pain Analysis Pipeline
% This script demonstrates the basic usage of the spontaneous pain
% behavioral analysis pipeline with default settings.

%% Clear workspace and setup
clear; close all; clc;

% Add pipeline to path (if not already done)
if ~exist('setup_pipeline', 'file')
    addpath('../scripts');
end

fprintf('=== Basic Pipeline Usage Example ===\n\n');

%% Step 1: Setup Environment
fprintf('Step 1: Setting up environment...\n');

% Run setup to initialize paths and check dependencies
setup_results = setup_pipeline();

if ~setup_results.success
    error('Pipeline setup failed. Please resolve the issues and try again.');
end

fprintf('✓ Environment setup completed successfully\n\n');

%% Step 2: Check Data Availability
fprintf('Step 2: Checking for input data...\n');

% Check if sample data exists
data_dir = '../data/raw';
mouse_file = fullfile(data_dir, 'mouseFileOrder.mat');

if exist(mouse_file, 'file')
    fprintf('✓ Found mouseFileOrder.mat\n');
    
    % Load and inspect the data structure
    data_info = load(mouse_file);
    fprintf('  - Number of files: %d\n', length(data_info.mouseOrderShort));
    
    if isfield(data_info, 'metadata')
        fprintf('  - Metadata structure found\n');
        
        % Display group information
        if isfield(data_info.metadata, 'week1_indices')
            week1_groups = fieldnames(data_info.metadata.week1_indices);
            fprintf('  - Week 1 groups: %s\n', strjoin(week1_groups, ', '));
        end
        
        if isfield(data_info.metadata, 'week2_indices')
            week2_groups = fieldnames(data_info.metadata.week2_indices);
            fprintf('  - Week 2 groups: %s\n', strjoin(week2_groups, ', '));
        end
    end
    
    fprintf('\n');
else
    fprintf('! No input data found. Using sample data generation...\n');
    
    % Generate sample data for demonstration
    fprintf('Generating sample data...\n');
    generate_sample_data(data_dir);
    fprintf('✓ Sample data generated\n\n');
end

%% Step 3: Load Configuration
fprintf('Step 3: Loading pipeline configuration...\n');

% Use default configuration
config_file = '../config/pipeline_config.m';
if exist(config_file, 'file')
    addpath('../config');
    config = pipeline_config();
    fprintf('✓ Configuration loaded from %s\n', config_file);
else
    % Use minimal default config
    config = create_minimal_config();
    fprintf('✓ Using minimal default configuration\n');
end

% Display key configuration parameters
fprintf('Configuration summary:\n');
fprintf('  - Data format: %s\n', config.data.format);
fprintf('  - PCA components: %d\n', config.parameters.pca.num_components);
fprintf('  - Wavelet frequency range: %.1f - %.1f Hz\n', ...
    config.parameters.wavelets.min_freq, config.parameters.wavelets.max_freq);
fprintf('  - Output directory: %s\n', config.output.base_dir);

fprintf('\n');

%% Step 4: Run Complete Pipeline
fprintf('Step 4: Running complete pipeline...\n');
fprintf('This may take several minutes depending on data size...\n\n');

% Run the complete pipeline with default settings
try
    % Execute pipeline
    results = run_pipeline(...
        'data_dir', data_dir, ...
        'output_dir', '../outputs', ...
        'verbose', true);
    
    if results.success
        fprintf('✓ Pipeline completed successfully!\n');
        fprintf('Total execution time: %.1f minutes\n', results.total_duration / 60);
        
        % Display key results
        fprintf('\nPipeline Results Summary:\n');
        fprintf('========================\n');
        
        if isfield(results.stages, 'training') && isfield(results.stages.training, 'num_mice')
            fprintf('- Processed %d mice\n', results.stages.training.num_mice);
        end
        
        if isfield(results.stages, 'analysis') && isfield(results.stages.analysis, 'num_regions')
            fprintf('- Identified %d behavioral regions\n', results.stages.analysis.num_regions);
        end
        
        % List generated outputs
        fprintf('\nGenerated Outputs:\n');
        output_dir = results.output_dir;
        
        if exist(fullfile(output_dir, 'results'), 'dir')
            result_files = dir(fullfile(output_dir, 'results', '*.mat'));
            fprintf('- Results: %d .mat files\n', length(result_files));
        end
        
        if exist(fullfile(output_dir, 'figures'), 'dir')
            figure_files = dir(fullfile(output_dir, 'figures', '*.png'));
            fprintf('- Figures: %d .png files\n', length(figure_files));
        end
        
        if exist(fullfile(output_dir, 'analysis', 'csv'), 'dir')
            csv_files = dir(fullfile(output_dir, 'analysis', 'csv', '*.csv'));
            fprintf('- CSV exports: %d files\n', length(csv_files));
        end
        
    else
        fprintf('✗ Pipeline failed. Check error messages above.\n');
        
        % Display stage-specific errors
        stages = fieldnames(results.stages);
        for i = 1:length(stages)
            stage = results.stages.(stages{i});
            if ~stage.success && isfield(stage, 'error')
                fprintf('Error in %s stage: %s\n', stages{i}, stage.error);
            end
        end
    end
    
catch ME
    fprintf('✗ Pipeline execution failed with error:\n');
    fprintf('%s\n', ME.message);
    
    if length(ME.stack) > 0
        fprintf('Error occurred in: %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
    end
end

fprintf('\n');

%% Step 5: Explore Results (if successful)
if exist('results', 'var') && results.success
    fprintf('Step 5: Exploring results...\n');
    
    % Load main results file
    main_results_file = fullfile(results.output_dir, 'results', 'final_pipeline_results.mat');
    if exist(main_results_file, 'file')
        fprintf('Loading main results from: %s\n', main_results_file);
        loaded_results = load(main_results_file);
        
        % Display basic statistics
        if isfield(loaded_results, 'results')
            res = loaded_results.results;
            fprintf('✓ Results loaded successfully\n');
            
            % Show timing breakdown
            if isfield(res, 'stage_durations')
                fprintf('\nTiming breakdown:\n');
                for i = 1:length(res.stage_names)
                    if i <= length(res.stage_durations)
                        fprintf('  %s: %.1f seconds\n', res.stage_names{i}, res.stage_durations(i));
                    end
                end
            end
        end
    end
    
    % Show sample figures if available
    figure_dir = fullfile(results.output_dir, 'figures');
    if exist(figure_dir, 'dir')
        figure_files = dir(fullfile(figure_dir, '*.png'));
        if length(figure_files) > 0
            fprintf('\nSample figures generated:\n');
            for i = 1:min(3, length(figure_files))  % Show first 3 figures
                fprintf('  - %s\n', figure_files(i).name);
            end
            
            % Display a figure if requested
            response = input('\nDisplay first figure? (y/n): ', 's');
            if strcmpi(response, 'y')
                first_figure = fullfile(figure_dir, figure_files(1).name);
                if exist('imshow', 'file')
                    figure('Name', 'Pipeline Output Example');
                    img = imread(first_figure);
                    imshow(img);
                    title(strrep(figure_files(1).name, '_', '\_'));
                else
                    fprintf('Image viewing not available (Image Processing Toolbox required)\n');
                end
            end
        end
    end
    
    % Show CSV exports
    csv_dir = fullfile(results.output_dir, 'analysis', 'csv');
    if exist(csv_dir, 'dir')
        csv_files = dir(fullfile(csv_dir, '*.csv'));
        if length(csv_files) > 0
            fprintf('\nCSV exports available:\n');
            for i = 1:min(3, length(csv_files))  % Show first 3 CSV files
                fprintf('  - %s\n', csv_files(i).name);
            end
            
            % Show content of first CSV
            response = input('\nShow content of first CSV file? (y/n): ', 's');
            if strcmpi(response, 'y')
                first_csv = fullfile(csv_dir, csv_files(1).name);
                if exist('readtable', 'file')
                    try
                        data_table = readtable(first_csv);
                        fprintf('\nFirst few rows of %s:\n', csv_files(1).name);
                        disp(head(data_table));
                    catch
                        fprintf('Could not read CSV file\n');
                    end
                else
                    fprintf('CSV reading not available\n');
                end
            end
        end
    end
    
else
    fprintf('Step 5: Results exploration skipped (pipeline not successful)\n');
end

fprintf('\n');

%% Step 6: Next Steps and Recommendations
fprintf('Step 6: Next steps and recommendations\n');
fprintf('=====================================\n');

if exist('results', 'var') && results.success
    fprintf('✓ Basic pipeline execution completed successfully!\n\n');
    
    fprintf('Recommended next steps:\n');
    fprintf('1. Explore the generated figures in: %s/figures/\n', results.output_dir);
    fprintf('2. Analyze the CSV exports in: %s/analysis/csv/\n', results.output_dir);
    fprintf('3. Read the pipeline report: %s/validation/pipeline_execution_report.txt\n', results.output_dir);
    fprintf('4. Try custom configuration: examples/custom_analysis.m\n');
    fprintf('5. Add your own experimental groups: config/group_config.m\n');
    
else
    fprintf('Pipeline execution encountered issues.\n\n');
    
    fprintf('Troubleshooting steps:\n');
    fprintf('1. Check setup results: run setup_pipeline() again\n');
    fprintf('2. Verify input data format: validate your .mat files\n');
    fprintf('3. Check dependencies: ensure all required toolboxes are installed\n');
    fprintf('4. Review error messages above\n');
    fprintf('5. Consult troubleshooting guide: docs/troubleshooting.md\n');
end

fprintf('\n=== Basic Usage Example Complete ===\n');

%% Helper Functions

function generate_sample_data(data_dir)
    %GENERATE_SAMPLE_DATA Create minimal sample data for demonstration
    
    % Create data directory if it doesn't exist
    if ~exist(data_dir, 'dir')
        mkdir(data_dir);
    end
    
    % Generate sample mouse file order
    mouseOrderShort = {
        'sample_DRG_1.mat';
        'sample_SC_1.mat';
        'sample_SNI_1.mat'
    };
    
    % Create sample metadata
    metadata = struct();
    metadata.week1_indices = struct();
    metadata.week1_indices.DRG = 1;
    metadata.week1_indices.SC = 2;
    metadata.week2_indices = struct();
    metadata.week2_indices.SNI = 3;
    
    % Save mouse file order
    mouse_file = fullfile(data_dir, 'mouseFileOrder.mat');
    save(mouse_file, 'mouseOrderShort', 'metadata');
    
    % Generate sample pose data files
    for i = 1:length(mouseOrderShort)
        filename = fullfile(data_dir, mouseOrderShort{i});
        
        % Generate random pose data (500 frames, 3 coords, 23 joints)
        n_frames = 500;
        n_coords = 3;
        n_joints = 23;
        
        % Create realistic-looking pose data
        pred = zeros(n_frames, n_coords, n_joints);
        
        for joint = 1:n_joints
            % Generate smooth trajectories with some noise
            base_x = 100 + 50 * sin(2*pi*(1:n_frames)/100) + 10*randn(1,n_frames);
            base_y = 150 + 30 * cos(2*pi*(1:n_frames)/80) + 8*randn(1,n_frames);
            base_z = 50 + 20 * sin(2*pi*(1:n_frames)/60) + 5*randn(1,n_frames);
            
            pred(:, 1, joint) = base_x + joint*5;  % Spread joints spatially
            pred(:, 2, joint) = base_y + joint*3;
            pred(:, 3, joint) = base_z + joint*2;
        end
        
        % Save sample data
        save(filename, 'pred');
    end
    
    % Create training data files
    training_files = {'SNI_2.mat', 'week4-TBI_3.mat'};
    for i = 1:length(training_files)
        filename = fullfile(data_dir, training_files{i});
        
        % Generate larger training dataset
        n_frames = 1000;
        pred = zeros(n_frames, n_coords, n_joints);
        
        for joint = 1:n_joints
            base_x = 100 + 50 * sin(2*pi*(1:n_frames)/100) + 10*randn(1,n_frames);
            base_y = 150 + 30 * cos(2*pi*(1:n_frames)/80) + 8*randn(1,n_frames);
            base_z = 50 + 20 * sin(2*pi*(1:n_frames)/60) + 5*randn(1,n_frames);
            
            pred(:, 1, joint) = base_x + joint*5;
            pred(:, 2, joint) = base_y + joint*3;
            pred(:, 3, joint) = base_z + joint*2;
        end
        
        save(filename, 'pred');
    end
end

function config = create_minimal_config()
    %CREATE_MINIMAL_CONFIG Create minimal configuration for demo
    
    config = struct();
    config.version = '1.4.0';
    config.description = 'Minimal demo configuration';
    
    % Data configuration
    config.data = struct();
    config.data.format = 'rat23';
    config.data.expected_dimensions = [NaN, 3, 23];
    
    % Parameters
    config.parameters = struct();
    config.parameters.pca = struct();
    config.parameters.pca.num_components = 15;
    config.parameters.pca.batch_size = 30000;
    
    config.parameters.wavelets = struct();
    config.parameters.wavelets.min_freq = 0.5;
    config.parameters.wavelets.max_freq = 20;
    config.parameters.wavelets.pca_modes = 20;
    config.parameters.wavelets.sampling_freq = 50;
    
    config.parameters.tsne = struct();
    config.parameters.tsne.num_per_dataset = 320;
    
    config.parameters.watershed = struct();
    config.parameters.watershed.connectivity = 18;
    config.parameters.watershed.density_range = [-65 65];
    config.parameters.watershed.grid_size = 501;
    
    % Output configuration
    config.output = struct();
    config.output.base_dir = '../outputs';
    config.output.generate_figures = true;
    config.output.export_csv = true;
    
    % Files
    config.files = struct();
    config.files.mouse_file_order = 'mouseFileOrder.mat';
    config.files.final_results = 'mouseEmbeddingResults_weekdata.mat';
    
    % Training
    config.training = struct();
    config.training.files = {'SNI_2.mat', 'week4-TBI_3.mat'};
end