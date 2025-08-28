%% Analyze differences between original and flipped mouse embeddings for different groups within weeks
% This script loads the saved embedding data and quantifies differences
% Compares between groups within the same week 
% MODIFIED FOR MOUSE14 FORMAT: 14 joints instead of 23
% UPDATED FOR SNI_2 and week4-TBI_3 training data structure

clear; close all; clc;

%% Setup output directories and files
outputDir = 'flip_analysis_output';
plotsDir = fullfile(outputDir, 'plots');
csvDir = fullfile(outputDir, 'csv_results');

% Create directories
if ~exist(outputDir, 'dir'), mkdir(outputDir); end
if ~exist(plotsDir, 'dir'), mkdir(plotsDir); end
if ~exist(csvDir, 'dir'), mkdir(csvDir); end

fprintf('Output directory: %s\n', outputDir);
fprintf('Plots will be saved to: %s\n', plotsDir);
fprintf('CSV files will be saved to: %s\n', csvDir);

% Add required helper functions
returnDist3d = @(x,y) sqrt(sum((x-y).^2,2));
combineCells = @(x) vertcat(x{:});

% Check if smooth function exists, otherwise provide simple alternative
if ~exist('smooth', 'file')
    smooth = @(x, n) movmean(x, n);
end

% Check for imgaussfilt
if ~exist('imgaussfilt', 'file')
    if exist('imfilter', 'file') && exist('fspecial', 'file')
        imgaussfilt = @(img, sigma) imfilter(img, fspecial('gaussian', ceil(6*sigma), sigma));
    else
        % Simple box filter as fallback
        imgaussfilt = @(img, sigma) conv2(img, ones(ceil(6*sigma))/(ceil(6*sigma)^2), 'same');
    end
end

%% Load embedding results from custom SNI_TBI pipeline
fprintf('Loading saved embedding data...\n');

try 
    load('complete_embedding_results_SNI_TBI.mat', 'results');
    
    % Extract data from results structure
    zEmbeddings = results.zEmbeddings_all;
    reembedding_labels = results.reembedding_labels_all; 
    reembedding_metadata = results.reembedding_metadata_all;
    D = results.D;
    LL = results.LL;
    LL2 = results.LL2;
    llbwb = results.llbwb;
    
    fprintf('✓ Successfully loaded embedding results\n');
    fprintf('  Found embeddings for %d datasets\n', length(zEmbeddings));
    
catch ME
    fprintf('❌ Error loading embedding results: %s\n', ME.message);
    fprintf('Make sure you have run custom_embedding_pipeline_SNI_TBI.m first\n');
    return;
end

%% Load raw data from your data directory
fprintf('Loading raw DANNCE data...\n');

data_dir = '/work/rl349/dannce/mouse14/allData';
all_files = {
    'DRG_1.mat', 'DRG_2.mat', 'DRG_3.mat', 'DRG_4.mat', 'DRG_5.mat', ...
    'IT_1.mat', 'IT_2.mat', ...
    'SC_1.mat', 'SC_2.mat', 'SC_3.mat', 'SC_4.mat', 'SC_5.mat', 'SC_6.mat', ...
    'SNI_1.mat', 'SNI_2.mat', 'SNI_3.mat', ...
    'week4-DRG_1.mat', 'week4-DRG_2.mat', 'week4-DRG_3.mat', ...
    'week4-SC_1.mat', 'week4-SC_2.mat', 'week4-SC_3.mat', ...
    'week4-SNI_1.mat', 'week4-SNI_2.mat', 'week4-SNI_3.mat', ...
    'week4-TBI_1.mat', 'week4-TBI_2.mat', 'week4-TBI_3.mat', 'week4-TBI_4.mat'
};

% Load raw data
allMOUSE = cell(length(all_files), 1);
mouseFiles = cell(length(all_files), 1);
mouseGroups = cell(length(all_files), 1);
mouseWeeks = cell(length(all_files), 1);

numValidFiles = 0;
for i = 1:length(all_files)
    file_path = fullfile(data_dir, all_files{i});
    
    if exist(file_path, 'file')
        try
            data = load(file_path);
            if isfield(data, 'pred')
                numValidFiles = numValidFiles + 1;
                allMOUSE{numValidFiles} = data.pred;
                mouseFiles{numValidFiles} = all_files{i};
                
                % Parse filename to get group and week
                [group, week] = parseFilename(all_files{i});
                mouseGroups{numValidFiles} = group;
                mouseWeeks{numValidFiles} = week;
                
                fprintf('✓ Loaded %s (%s, %s): %d frames\n', all_files{i}, group, week, size(data.pred, 1));
            else
                fprintf('⚠ %s: No pred field found\n', all_files{i});
            end
        catch ME
            fprintf('❌ Error loading %s: %s\n', all_files{i}, ME.message);
        end
    else
        fprintf('⚠ File not found: %s\n', all_files{i});
    end
end

% Trim empty cells
allMOUSE = allMOUSE(1:numValidFiles);
mouseFiles = mouseFiles(1:numValidFiles);
mouseGroups = mouseGroups(1:numValidFiles);
mouseWeeks = mouseWeeks(1:numValidFiles);

fprintf('Successfully loaded %d data files\n', numValidFiles);

%% Create mouse labels for analysis
mouseLabels = cell(numValidFiles, 1);
groupCounts = containers.Map();

for i = 1:numValidFiles
    group = mouseGroups{i};
    if isKey(groupCounts, group)
        groupCounts(group) = groupCounts(group) + 1;
    else
        groupCounts(group) = 1;
    end
    mouseLabels{i} = sprintf('%s%d', group, groupCounts(group));
end

%% Initialize results structure with mouse14-specific parameters
flip_results = struct();
flip_results.mouseFiles = mouseFiles;
flip_results.mouseLabels = mouseLabels; 
flip_results.mouseGroups = mouseGroups;
flip_results.mouseWeeks = mouseWeeks;
flip_results.centroidDist = zeros(numValidFiles, 1);
flip_results.densityOverlap = zeros(numValidFiles, 1);
flip_results.densityCorr = zeros(numValidFiles, 1);
flip_results.avgDisplacement = zeros(numValidFiles, 1);
flip_results.biomechanicalAsymmetry = zeros(numValidFiles, 1);
flip_results.frontLimbAsymmetry = zeros(numValidFiles, 1);
flip_results.hindLimbAsymmetry = zeros(numValidFiles, 1);

%% Main analysis loop - analyze each mouse
fprintf('\nAnalyzing flip differences for mouse14 format...\n');

for m = 1:numValidFiles
    fprintf('Processing mouse %d/%d (%s)...\n', m, numValidFiles, mouseLabels{m});
    
    % Find corresponding embeddings (original and flipped)
    % The embeddings are stored as: [original_files; flipped_files]
    original_idx = m;  % Index for original mouse
    flipped_idx = m + numValidFiles;  % Index for flipped version
    
    if original_idx <= length(zEmbeddings) && flipped_idx <= length(zEmbeddings)
        z_orig = zEmbeddings{original_idx};
        z_flip = zEmbeddings{flipped_idx};
        
        % Transform to image coordinates (using dynamic bounds)
        if ~isempty(z_orig) && ~isempty(z_flip)
            % Simple transformation to positive coordinates
            z_orig_img = (z_orig + 65) * 501 / 130;
            z_flip_img = (z_flip + 65) * 501 / 130;
            
            % 1. Calculate centroid distance
            centroid_orig = mean(z_orig_img, 1);
            centroid_flip = mean(z_flip_img, 1);
            flip_results.centroidDist(m) = norm(centroid_orig - centroid_flip);
            
            % 2. Calculate density overlap using 2D histogram
            try
                % Create combined bounds
                all_points = [z_orig_img; z_flip_img];
                x_range = [min(all_points(:,1)), max(all_points(:,1))];
                y_range = [min(all_points(:,2)), max(all_points(:,2))];
                
                % Create histograms
                n_bins = 50;
                [N_orig, ~, ~] = histcounts2(z_orig_img(:,1), z_orig_img(:,2), ...
                    linspace(x_range(1), x_range(2), n_bins), ...
                    linspace(y_range(1), y_range(2), n_bins));
                [N_flip, ~, ~] = histcounts2(z_flip_img(:,1), z_flip_img(:,2), ...
                    linspace(x_range(1), x_range(2), n_bins), ...
                    linspace(y_range(1), y_range(2), n_bins));
                
                % Normalize
                N_orig_norm = N_orig / sum(N_orig(:));
                N_flip_norm = N_flip / sum(N_flip(:));
                
                % Calculate Jaccard similarity
                intersection = min(N_orig_norm, N_flip_norm);
                union = max(N_orig_norm, N_flip_norm);
                flip_results.densityOverlap(m) = sum(intersection(:)) / sum(union(:));
                
                % Calculate correlation
                flip_results.densityCorr(m) = corr(N_orig_norm(:), N_flip_norm(:));
                
            catch
                flip_results.densityOverlap(m) = NaN;
                flip_results.densityCorr(m) = NaN;
            end
            
            % 3. Calculate average displacement
            if size(z_orig_img, 1) == size(z_flip_img, 1)
                displacements = sqrt(sum((z_orig_img - z_flip_img).^2, 2));
                flip_results.avgDisplacement(m) = mean(displacements);
            else
                flip_results.avgDisplacement(m) = NaN;
            end
        end
    end
    
    % 4. Biomechanical analysis using mouse14 joint structure
    if m <= length(allMOUSE) && ~isempty(allMOUSE{m})
        mouseData = allMOUSE{m}; % frames x 3 x 14
        
        % Validate dimensions
        if size(mouseData, 3) ~= 14
            fprintf('⚠ Warning: Expected 14 joints, found %d in %s\n', size(mouseData, 3), mouseFiles{m});
            continue;
        end
        
        % Define mouse14 joint indices (1-indexed)
        % 1:Snout, 2:EarL, 3:EarR, 4:SpineF, 5:SpineM, 6:Tail(base)
        % 7:ForShdL, 8:ForepawL, 9:ForeShdR, 10:ForepawR
        % 11:HindShdL, 12:HindpawL, 13:HindShdR, 14:HindpawR
        
        spineF = 4; spineM = 5; 
        forShdL = 7; forepawL = 8; foreShdR = 9; forepawR = 10;
        hindShdL = 11; hindpawL = 12; hindShdR = 13; hindpawR = 14;
        
        numFrames = min(size(mouseData, 1), size(z_orig, 1));
        
        % Calculate front limb asymmetry
        frontLimbAsymmetries = zeros(numFrames, 1);
        hindLimbAsymmetries = zeros(numFrames, 1);
        
        for f = 1:numFrames
            try
                % Front limb angles: SpineF -> ForShdL/R -> ForepawL/R
                spineF_pos = squeeze(mouseData(f, :, spineF))';  % 1x3
                forShdL_pos = squeeze(mouseData(f, :, forShdL))';
                forepawL_pos = squeeze(mouseData(f, :, forepawL))';
                foreShdR_pos = squeeze(mouseData(f, :, foreShdR))';
                forepawR_pos = squeeze(mouseData(f, :, forepawR))';
                
                % Calculate front limb angles (shoulder-elbow-paw)
                leftFrontAngle = calculateAngle3D(spineF_pos, forShdL_pos, forepawL_pos);
                rightFrontAngle = calculateAngle3D(spineF_pos, foreShdR_pos, forepawR_pos);
                
                frontLimbAsymmetries(f) = abs(leftFrontAngle - rightFrontAngle);
                
                % Hind limb angles: SpineM -> HindShdL/R -> HindpawL/R  
                spineM_pos = squeeze(mouseData(f, :, spineM))';
                hindShdL_pos = squeeze(mouseData(f, :, hindShdL))';
                hindpawL_pos = squeeze(mouseData(f, :, hindpawL))';
                hindShdR_pos = squeeze(mouseData(f, :, hindShdR))';
                hindpawR_pos = squeeze(mouseData(f, :, hindpawR))';
                
                % Calculate hind limb angles
                leftHindAngle = calculateAngle3D(spineM_pos, hindShdL_pos, hindpawL_pos);
                rightHindAngle = calculateAngle3D(spineM_pos, hindShdR_pos, hindpawR_pos);
                
                hindLimbAsymmetries(f) = abs(leftHindAngle - rightHindAngle);
                
            catch ME
                % Handle any calculation errors
                frontLimbAsymmetries(f) = NaN;
                hindLimbAsymmetries(f) = NaN;
            end
        end
        
        % Store average asymmetries
        flip_results.frontLimbAsymmetry(m) = nanmean(frontLimbAsymmetries);
        flip_results.hindLimbAsymmetry(m) = nanmean(hindLimbAsymmetries);
        flip_results.biomechanicalAsymmetry(m) = nanmean([frontLimbAsymmetries; hindLimbAsymmetries]);
        
    end
end

%% Display results by group and week
fprintf('\n=== FLIP DIFFERENCE ANALYSIS RESULTS (Mouse14 Format) ===\n');

unique_weeks = unique(mouseWeeks);
for w = 1:length(unique_weeks)
    week = unique_weeks{w};
    weekIdx = strcmp(mouseWeeks, week);
    
    if any(weekIdx)
        fprintf('\n--- %s ---\n', upper(week));
        fprintf('Mouse\t\tGroup\tCentroid\tDensity\tCorr\tDispl\tBiomech\tFront\tHind\n');
        fprintf('\t\t\tDist\tOverlap\t\t\tAsym\tAsym\tAsym\n');
        
        weekIndices = find(weekIdx);
        for i = weekIndices'
            fprintf('%s\t%s\t%.2f\t%.3f\t%.3f\t%.2f\t%.2f\t%.2f\t%.2f\n', ...
                mouseLabels{i}, mouseGroups{i}, ...
                flip_results.centroidDist(i), flip_results.densityOverlap(i), ...
                flip_results.densityCorr(i), flip_results.avgDisplacement(i), ...
                flip_results.biomechanicalAsymmetry(i), ...
                flip_results.frontLimbAsymmetry(i), flip_results.hindLimbAsymmetry(i));
        end
    end
end

%% Create visualization plots
fprintf('\n=== CREATING VISUALIZATIONS ===\n');

% Define colors for each group
groupColors = struct();
groupColors.DRG = [1 0 0];         % Red
groupColors.SC = [0 0 1];          % Blue  
groupColors.IT = [0 1 0];          % Green
groupColors.SNI = [1 0.5 0];       % Orange
groupColors.TBI = [0.5 0 1];       % Purple

% Create group comparison plots
createGroupComparisonPlots(flip_results, groupColors, unique_weeks, plotsDir);

% Create individual mouse density plots
createIndividualMousePlots(flip_results, zEmbeddings, D, llbwb, groupColors, plotsDir);

% Create biomechanical analysis plots
createBiomechanicalPlots(flip_results, groupColors, unique_weeks, plotsDir);

%% Export results to CSV
fprintf('\n=== EXPORTING RESULTS TO CSV ===\n');

% Create comprehensive results table
try
    resultsTable = table(mouseFiles, mouseLabels, mouseGroups, mouseWeeks, ...
        flip_results.centroidDist, flip_results.densityOverlap, flip_results.densityCorr, ...
        flip_results.avgDisplacement, flip_results.biomechanicalAsymmetry, ...
        flip_results.frontLimbAsymmetry, flip_results.hindLimbAsymmetry, ...
        'VariableNames', {'MouseFile', 'MouseLabel', 'Group', 'Week', ...
        'CentroidDistance', 'DensityOverlap', 'DensityCorrelation', ...
        'AverageDisplacement', 'BiomechanicalAsymmetry', ...
        'FrontLimbAsymmetry', 'HindLimbAsymmetry'});
catch
    % Fallback for older MATLAB versions
    resultsTable = table();
    resultsTable.MouseFile = mouseFiles;
    resultsTable.MouseLabel = mouseLabels;
    resultsTable.Group = mouseGroups;
    resultsTable.Week = mouseWeeks;
    resultsTable.CentroidDistance = flip_results.centroidDist;
    resultsTable.DensityOverlap = flip_results.densityOverlap;
    resultsTable.DensityCorrelation = flip_results.densityCorr;
    resultsTable.AverageDisplacement = flip_results.avgDisplacement;
    resultsTable.BiomechanicalAsymmetry = flip_results.biomechanicalAsymmetry;
    resultsTable.FrontLimbAsymmetry = flip_results.frontLimbAsymmetry;
    resultsTable.HindLimbAsymmetry = flip_results.hindLimbAsymmetry;
end

csvFile = fullfile(csvDir, 'flip_analysis_results_mouse14.csv');
writetable(resultsTable, csvFile);
fprintf('✓ Exported results to: %s\n', csvFile);

% Create group summary statistics
fprintf('Creating group summary statistics...\n');
createGroupSummaryStats(flip_results, csvDir);

%% Save complete results
fprintf('\nSaving complete analysis results...\n');
save(fullfile(outputDir, 'flip_analysis_results_mouse14.mat'), 'flip_results', 'resultsTable');

fprintf('\n=== ANALYSIS COMPLETE ===\n');
fprintf('Results saved to: %s\n', outputDir);
fprintf('- CSV files: %s\n', csvDir);
fprintf('- Plots: %s\n', plotsDir);

%% HELPER FUNCTIONS

function [group, week] = parseFilename(filename)
    % Parse filename to extract group and week information
    name = filename(1:end-4); % Remove .mat extension
    
    if startsWith(name, 'week4-')
        week = 'week4';
        name = name(7:end); % Remove 'week4-' prefix
    else
        week = 'week1';
    end
    
    % Extract group
    if startsWith(name, 'DRG')
        group = 'DRG';
    elseif startsWith(name, 'IT')
        group = 'IT';
    elseif startsWith(name, 'SC')
        group = 'SC';
    elseif startsWith(name, 'SNI')
        group = 'SNI';
    elseif startsWith(name, 'TBI')
        group = 'TBI';
    else
        group = 'unknown';
    end
end

function angle = calculateAngle3D(p1, p2, p3)
    % Calculate angle at p2 formed by p1-p2-p3 in 3D space
    % Input: p1, p2, p3 are 1x3 vectors
    
    % Convert to column vectors if needed
    if size(p1, 1) == 1, p1 = p1'; end
    if size(p2, 1) == 1, p2 = p2'; end
    if size(p3, 1) == 1, p3 = p3'; end
    
    v1 = p1 - p2;
    v2 = p3 - p2;
    
    cos_angle = dot(v1, v2) / (norm(v1) * norm(v2) + eps);
    cos_angle = max(-1, min(1, cos_angle)); % Clamp to [-1, 1]
    angle = acos(cos_angle) * 180 / pi;
end

function createGroupComparisonPlots(flip_results, groupColors, unique_weeks, plotsDir)
    % Create group comparison plots for different metrics
    
    metrics = {'centroidDist', 'densityOverlap', 'densityCorr', 'avgDisplacement', ...
               'biomechanicalAsymmetry', 'frontLimbAsymmetry', 'hindLimbAsymmetry'};
    metricNames = {'Centroid Distance', 'Density Overlap', 'Density Correlation', ...
                   'Average Displacement', 'Biomechanical Asymmetry', ...
                   'Front Limb Asymmetry', 'Hind Limb Asymmetry'};
    
    for w = 1:length(unique_weeks)
        week = unique_weeks{w};
        weekIdx = strcmp(flip_results.mouseWeeks, week);
        weekGroups = unique(flip_results.mouseGroups(weekIdx));
        
        if length(weekGroups) < 2, continue; end
        
        figure('Name', sprintf('%s - Group Comparisons Mouse14', week), ...
            'Position', [100 100 1800 1200]);
        
        for m = 1:length(metrics)
            subplot(3, 3, m);
            
            metric = metrics{m};
            metricName = metricNames{m};
            
            % Get data for this week and metric
            weekData = flip_results.(metric)(weekIdx);
            weekGroupLabels = flip_results.mouseGroups(weekIdx);
            
            % Create grouped data
            groupData = struct();
            for g = 1:length(weekGroups)
                group = weekGroups{g};
                groupIdx = strcmp(weekGroupLabels, group);
                groupData.(group) = weekData(groupIdx);
            end
            
            % Create box plot
            hold on;
            positions = [];
            all_data = [];
            group_labels = {};
            
            for g = 1:length(weekGroups)
                group = weekGroups{g};
                values = groupData.(group);
                
                if ~isempty(values) && any(~isnan(values))
                    % Add jitter and plot points
                    x_pos = g + 0.1 * (rand(length(values), 1) - 0.5);
                    
                    if isfield(groupColors, group)
                        color = groupColors.(group);
                    else
                        color = [0.5 0.5 0.5];
                    end
                    
                    scatter(x_pos, values, 60, color, 'filled', 'MarkerEdgeColor', 'k', ...
                        'MarkerEdgeAlpha', 0.3, 'MarkerFaceAlpha', 0.7);
                    
                    % Add mean and error bars
                    errorbar(g, nanmean(values), nanstd(values) / sqrt(sum(~isnan(values))), ...
                        'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k', 'LineWidth', 2);
                end
            end
            
            xlim([0.5, length(weekGroups) + 0.5]);
            set(gca, 'XTick', 1:length(weekGroups), 'XTickLabel', weekGroups);
            ylabel(metricName);
            title(sprintf('%s: %s', week, metricName));
            grid on;
            box on;
        end
        
        sgtitle(sprintf('%s: Group Comparisons (Mouse14 Format)', week), 'FontSize', 16);
        
        % Save plot
        savePath = fullfile(plotsDir, sprintf('%s_group_comparisons_mouse14.png', week));
        saveas(gcf, savePath);
        fprintf('✓ Saved: %s\n', savePath);
    end
end

function createIndividualMousePlots(flip_results, zEmbeddings, D, llbwb, groupColors, plotsDir)
    % Create individual mouse density plots
    
    unique_groups = unique(flip_results.mouseGroups);
    
    for g = 1:length(unique_groups)
        group = unique_groups{g};
        groupIdx = strcmp(flip_results.mouseGroups, group);
        
        if sum(groupIdx) == 0, continue; end
        
        figure('Name', sprintf('%s Individual Mice Mouse14', group), ...
            'Position', [100 100 1800 1000]);
        
        mice_in_group = find(groupIdx);
        n_mice = length(mice_in_group);
        n_cols = min(6, n_mice);
        n_rows = ceil(n_mice / n_cols);
        
        for m = 1:n_mice
            mouseIdx = mice_in_group(m);
            
            subplot(n_rows, n_cols, m);
            
            % Background
            imagesc(D);
            hold on;
            
            % Get embedding data
            if mouseIdx <= length(zEmbeddings) && ~isempty(zEmbeddings{mouseIdx})
                z = zEmbeddings{mouseIdx};
                z_img = (z + 65) * 501 / 130;
                
                % Create density map
                if size(z_img, 1) > 10
                    [N, xedges, yedges] = histcounts2(z_img(:,1), z_img(:,2), ...
                        linspace(1, 501, 40), linspace(1, 501, 40));
                    
                    if exist('imgaussfilt', 'file')
                        N = imgaussfilt(N', 2);
                    else
                        N = conv2(N', ones(3)/9, 'same');
                    end
                    
                    % Get color
                    if isfield(groupColors, group)
                        color = groupColors.(group);
                    else
                        color = [0.5 0.5 0.5];
                    end
                    
                    % Create overlay
                    overlay = zeros(size(N,1), size(N,2), 3);
                    for c = 1:3
                        overlay(:,:,c) = color(c);
                    end
                    
                    % Display
                    if max(N(:)) > 0
                        h = imagesc(xedges(1:end-1), yedges(1:end-1), overlay);
                        set(h, 'AlphaData', N/max(N(:))*0.8);
                    end
                end
            end
            
            % Add watershed boundaries
            scatter(llbwb(:,2), llbwb(:,1), 0.5, '.', 'k', 'MarkerEdgeAlpha', 0.3);
            
            axis equal off;
            colormap(gca, flipud(gray));
            caxis([0 max(D(:))*0.8]);
            
            % Create title
            title(strrep(flip_results.mouseLabels{mouseIdx}, '_', '\_'), 'FontSize', 8);
        end
        
        sgtitle(sprintf('%s - Individual Mouse Density Maps (Mouse14)', group), 'FontSize', 14);
        
        % Save plot
        savePath = fullfile(plotsDir, sprintf('%s_individual_mice_mouse14.png', group));
        saveas(gcf, savePath);
        fprintf('✓ Saved: %s\n', savePath);
    end
end

function createBiomechanicalPlots(flip_results, groupColors, unique_weeks, plotsDir)
    % Create biomechanical asymmetry analysis plots
    
    figure('Name', 'Biomechanical Asymmetry Analysis Mouse14', 'Position', [100 100 1500 900]);
    
    % Plot 1: Front vs Hind Limb Asymmetry
    subplot(2, 3, 1);
    scatter(flip_results.frontLimbAsymmetry, flip_results.hindLimbAsymmetry, 60, 'filled');
    xlabel('Front Limb Asymmetry (degrees)');
    ylabel('Hind Limb Asymmetry (degrees)');
    title('Front vs Hind Limb Asymmetry');
    grid on;
    
    % Plot 2: Asymmetry by Group
    subplot(2, 3, 2);
    unique_groups = unique(flip_results.mouseGroups);
    hold on;
    
    for g = 1:length(unique_groups)
        group = unique_groups{g};
        groupIdx = strcmp(flip_results.mouseGroups, group);
        
        if sum(groupIdx) > 0
            asym_values = flip_results.biomechanicalAsymmetry(groupIdx);
            x_pos = g + 0.1 * (rand(sum(groupIdx), 1) - 0.5);
            
            if isfield(groupColors, group)
                color = groupColors.(group);
            else
                color = [0.5 0.5 0.5];
            end
            
            scatter(x_pos, asym_values, 60, color, 'filled');
        end
    end
    
    xlim([0.5, length(unique_groups) + 0.5]);
    set(gca, 'XTick', 1:length(unique_groups), 'XTickLabel', unique_groups);
    ylabel('Biomechanical Asymmetry (degrees)');
    title('Asymmetry by Group');
    grid on;
    
    % Plot 3: Asymmetry by Week
    subplot(2, 3, 3);
    hold on;
    
    for w = 1:length(unique_weeks)
        week = unique_weeks{w};
        weekIdx = strcmp(flip_results.mouseWeeks, week);
        
        if sum(weekIdx) > 0
            asym_values = flip_results.biomechanicalAsymmetry(weekIdx);
            x_pos = w + 0.1 * (rand(sum(weekIdx), 1) - 0.5);
            
            scatter(x_pos, asym_values, 60, 'filled');
        end
    end
    
    xlim([0.5, length(unique_weeks) + 0.5]);
    set(gca, 'XTick', 1:length(unique_weeks), 'XTickLabel', unique_weeks);
    ylabel('Biomechanical Asymmetry (degrees)');
    title('Asymmetry by Week');
    grid on;
    
    % Plot 4: Correlation matrix
    subplot(2, 3, 4);
    metrics_data = [flip_results.centroidDist, flip_results.densityOverlap, ...
                   flip_results.avgDisplacement, flip_results.biomechanicalAsymmetry];
    corrMatrix = corr(metrics_data, 'rows', 'complete');
    
    imagesc(corrMatrix);
    colorbar;
    colormap('RdBu');
    caxis([-1 1]);
    
    metricLabels = {'Centroid Dist', 'Density Overlap', 'Avg Displacement', 'Biomech Asym'};
    set(gca, 'XTick', 1:4, 'XTickLabel', metricLabels, 'XTickLabelRotation', 45);
    set(gca, 'YTick', 1:4, 'YTickLabel', metricLabels);
    title('Metric Correlations');
    
    % Plot 5: Distribution of all asymmetry measures
    subplot(2, 3, 5);
    hold on;
    histogram(flip_results.frontLimbAsymmetry(~isnan(flip_results.frontLimbAsymmetry)), ...
        'Normalization', 'probability', 'FaceAlpha', 0.7, 'DisplayName', 'Front Limb');
    histogram(flip_results.hindLimbAsymmetry(~isnan(flip_results.hindLimbAsymmetry)), ...
        'Normalization', 'probability', 'FaceAlpha', 0.7, 'DisplayName', 'Hind Limb');
    
    xlabel('Asymmetry (degrees)');
    ylabel('Probability');
    title('Asymmetry Distributions');
    legend;
    grid on;
    
    sgtitle('Biomechanical Asymmetry Analysis (Mouse14 Format)', 'FontSize', 16);
    
    % Save plot
    savePath = fullfile(plotsDir, 'biomechanical_analysis_mouse14.png');
    saveas(gcf, savePath);
    fprintf('✓ Saved: %s\n', savePath);
end

function createGroupSummaryStats(flip_results, csvDir)
    % Create and export group summary statistics
    
    unique_groups = unique(flip_results.mouseGroups);
    unique_weeks = unique(flip_results.mouseWeeks);
    
    summaryStats = [];
    
    for w = 1:length(unique_weeks)
        week = unique_weeks{w};
        weekIdx = strcmp(flip_results.mouseWeeks, week);
        weekGroups = unique(flip_results.mouseGroups(weekIdx));
        
        for g = 1:length(weekGroups)
            group = weekGroups{g};
            groupIdx = weekIdx & strcmp(flip_results.mouseGroups, group);
            
            if sum(groupIdx) > 0
                stats = struct();
                stats.Week = {week};
                stats.Group = {group};
                stats.N_Mice = sum(groupIdx);
                
                % Calculate statistics for each metric
                metrics = {'centroidDist', 'densityOverlap', 'densityCorr', 'avgDisplacement', ...
                          'biomechanicalAsymmetry', 'frontLimbAsymmetry', 'hindLimbAsymmetry'};
                
                for m = 1:length(metrics)
                    metric = metrics{m};
                    data = flip_results.(metric)(groupIdx);
                    
                    stats.([metric '_mean']) = nanmean(data);
                    stats.([metric '_std']) = nanstd(data);
                    stats.([metric '_sem']) = nanstd(data) / sqrt(sum(~isnan(data)));
                    stats.([metric '_median']) = nanmedian(data);
                end
                
                if isempty(summaryStats)
                    summaryStats = stats;
                else
                    summaryStats = [summaryStats; stats];
                end
            end
        end
    end
    
    % Convert to table and export
    if ~isempty(summaryStats)
        summaryTable = struct2table(summaryStats);
        csvFile = fullfile(csvDir, 'group_summary_statistics_mouse14.csv');
        writetable(summaryTable, csvFile);
        fprintf('✓ Exported group summary: %s\n', csvFile);
    end
end
