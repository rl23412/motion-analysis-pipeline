%% Analyze differences between original and flipped mouse embeddings for different groups within weeks
% This script loads the saved embedding data and quantifies differences
% Compares between groups within the same week (e.g., Week1: DRG vs SC, Week2: DRG vs IT vs SC vs SNI)
% Also creates individual mouse density plots
% ENHANCED VERSION: Saves all plots, exports CSV results, calculates leg angles and asymmetries

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

%% Load necessary data
fprintf('Loading saved data...\n');
load('mouseEmbeddingResults_weekdata.mat', 'wrFINE', 'zEmbeddings', 'LL', 'LL2', 'D', 'xx', ...
    'mouseGroups', 'mouseWeeks', 'mouseFiles', 'metadata');
load('mouseFileOrder.mat', 'mouseOrderShort');

% Convert to cell array if needed
if ~iscell(mouseOrderShort)
    if ischar(mouseOrderShort)
        mouseOrderShort = cellstr(mouseOrderShort);
    elseif isstring(mouseOrderShort)
        mouseOrderShort = cellstr(mouseOrderShort);
    end
end

% Get valid files
validIdx = ~cellfun(@isempty, mouseOrderShort);
validFiles = mouseOrderShort(validIdx);
numMice = length(validFiles);
mouseFiles = validFiles;

% Create simple mouse labels (Group + Index)
mouseLabels = cell(numMice, 1);
groupCounts = containers.Map();
for i = 1:numMice
    group = mouseGroups{i};
    if isKey(groupCounts, group)
        groupCounts(group) = groupCounts(group) + 1;
    else
        groupCounts(group) = 1;
    end
    mouseLabels{i} = sprintf('%s%d', group, groupCounts(group));
end

% Define colors for each group
groupColors = struct();
groupColors.DRG = [1 0 0];     % Red
groupColors.SC = [0 0 1];      % Blue
groupColors.IT = [0 1 0];      % Green
groupColors.SNI = [1 0.5 0];   % Orange

%% Calculate quantitative differences between original and flipped by week and group
% Using density-based metrics (not just area)

% Initialize enhanced results structure
results = struct();
results.mouseID = mouseFiles;
results.mouseLabel = mouseLabels;
results.group = mouseGroups;
results.week = mouseWeeks;
results.centroidDist = zeros(numMice, 1);
results.densityOverlap = zeros(numMice, 1);           % Density-based overlap (Jaccard)
results.densityCorr = zeros(numMice, 1);              % Density correlation
results.avgDisplacement = zeros(numMice, 1);
results.densityKLDiv = zeros(numMice, 1);             % KL divergence between densities
results.densityBhattacharyya = zeros(numMice, 1);     % Bhattacharyya distance

% NEW: Enhanced analysis results
results.regionFrequency = cell(numMice, 1);          % Time spent in each watershed region
results.leftKneeAngle = cell(numMice, 1);            % SpineL->HipL->KneeL angle time series
results.rightKneeAngle = cell(numMice, 1);           % SpineL->HipR->KneeR angle time series  
results.leftAngularSpeed = cell(numMice, 1);         % Angular speed of left leg
results.rightAngularSpeed = cell(numMice, 1);        % Angular speed of right leg
results.legAsymmetry = cell(numMice, 1);              % Left-right leg asymmetry time series
results.comSpeed = cell(numMice, 1);                  % Center of mass speed time series

% Region-based summaries will be calculated later
regionResults = struct();
regionResults.regionID = [];
regionResults.groupCounts = [];                       % Frequency by group
regionResults.individualCounts = [];                  % Frequency by individual mouse
regionResults.avgLeftKneeAngle = [];                  % Average left knee angle per region
regionResults.avgRightKneeAngle = [];                 % Average right knee angle per region
regionResults.avgAngularSpeed = [];                   % Average angular speed per region
regionResults.avgLegAsymmetry = [];                   % Average leg asymmetry per region

%% Load individual mouse data for biomechanical analysis
fprintf('\nLoading individual mouse data for biomechanical analysis...\n');
allMOUSE = cell(numMice, 1);

for i = 1:numMice
    try
        data = load(mouseFiles{i});
        if isfield(data, 'pred')
            allMOUSE{i} = data.pred;
        else
            fprintf('Warning: No pred field in %s\n', mouseFiles{i});
            allMOUSE{i} = [];
        end
    catch ME
        fprintf('Warning: Could not load %s: %s\n', mouseFiles{i}, ME.message);
        allMOUSE{i} = [];
    end
end

fprintf('\nCalculating differences between original and flipped embeddings...\n');

for i = 1:numMice
    fprintf('Processing mouse %d/%d (%s)...\n', i, numMice, mouseLabels{i});
    
    % Get original and flipped embeddings
    z_orig = zEmbeddings{i};
    z_flip = zEmbeddings{i + numMice};
    
    % Load original mouse data for biomechanical analysis
    mouseData = [];
    if i <= length(allMOUSE) && ~isempty(allMOUSE{i})
        mouseData = allMOUSE{i}; % frames x 3 x 23
    else
        % Try to load from file if not in allMOUSE
        try
            data = load(mouseFiles{i});
            if isfield(data, 'pred')
                mouseData = data.pred;
            end
        catch
            fprintf('Warning: Could not load mouse data for %s\n', mouseFiles{i});
        end
    end
    
    % Transform to image coordinates
    z_orig_img = (z_orig + 65) * 501 / 130;
    z_flip_img = (z_flip + 65) * 501 / 130;
    
    % 1. Calculate centroid distance
    centroid_orig = mean(z_orig_img, 1);
    centroid_flip = mean(z_flip_img, 1);
    results.centroidDist(i) = norm(centroid_orig - centroid_flip);
    
    % 2. Calculate high-resolution density maps for better overlap analysis
    gridRes = 120; % Higher resolution to capture asymmetries better
    [N_orig, xedges, yedges] = histcounts2(z_orig_img(:,1), z_orig_img(:,2), ...
        linspace(1, 501, gridRes), linspace(1, 501, gridRes));
    [N_flip, ~, ~] = histcounts2(z_flip_img(:,1), z_flip_img(:,2), ...
        linspace(1, 501, gridRes), linspace(1, 501, gridRes));
    
    % Apply minimal Gaussian smoothing to preserve asymmetry differences
    sigma = 1; % Reduced smoothing parameter
    N_orig_smooth = imgaussfilt(N_orig', sigma);
    N_flip_smooth = imgaussfilt(N_flip', sigma);
    
    % Normalize to probability densities
    N_orig_norm = N_orig_smooth / sum(N_orig_smooth(:));
    N_flip_norm = N_flip_smooth / sum(N_flip_smooth(:));
    
    % 3. Calculate density-based overlap metrics
    
    % a) Jaccard similarity (intersection over union for densities)
    intersection = min(N_orig_norm, N_flip_norm);
    union = max(N_orig_norm, N_flip_norm);
    results.densityOverlap(i) = sum(intersection(:)) / sum(union(:));
    
    % b) Density correlation
    if exist('corr', 'file')
        results.densityCorr(i) = corr(N_orig_norm(:), N_flip_norm(:));
    else
        % Manual correlation calculation
        x = N_orig_norm(:);
        y = N_flip_norm(:);
        mx = mean(x);
        my = mean(y);
        results.densityCorr(i) = sum((x - mx).*(y - my)) / ...
            (sqrt(sum((x - mx).^2)) * sqrt(sum((y - my).^2)));
    end
    
    % c) KL divergence (symmetric)
    epsilon = 1e-10; % Small value to avoid log(0)
    p = N_orig_norm(:) + epsilon;
    q = N_flip_norm(:) + epsilon;
    p = p / sum(p);
    q = q / sum(q);
    kl_pq = sum(p .* log(p ./ q));
    kl_qp = sum(q .* log(q ./ p));
    results.densityKLDiv(i) = 0.5 * (kl_pq + kl_qp); % Symmetric KL divergence
    
    % d) Bhattacharyya distance
    results.densityBhattacharyya(i) = -log(sum(sqrt(p .* q)));
    
    % 4. Calculate average point-wise displacement (using time alignment)
    displacements = sqrt(sum((z_orig_img - z_flip_img).^2, 2));
    results.avgDisplacement(i) = mean(displacements);
    
    % 5. NEW: Biomechanical analysis
    if ~isempty(mouseData) && size(mouseData, 1) >= length(z_orig)
        % Define joint indices (based on analyze_full_leg_asymmetry.m)
        spineL = 6; hipL = 16; kneeL = 17; ankleL = 18; footL = 19;
        spineR = 6; hipR = 20; kneeR = 21; ankleR = 22; footR = 23;
        
        numFrames = min(size(mouseData, 1), length(z_orig));
        
        % Initialize arrays
        leftKneeAngles = zeros(numFrames, 1);
        rightKneeAngles = zeros(numFrames, 1);
        legAsymmetries = zeros(numFrames, 1);
        comSpeeds = zeros(numFrames, 1);
        regionFreq = containers.Map('KeyType', 'int32', 'ValueType', 'int32');
        
        % Calculate COM (Center of Mass) positions for speed analysis
        % Using key body points: Snout(1), SpineF(4), SpineM(5), SpineL(6), TailBase(7)
        comPositions = zeros(numFrames, 3);
        for f = 1:numFrames
            % Extract joint positions and ensure they are row vectors [1x3]
            snout_pos = squeeze(mouseData(f, :, 1));
            spineF_pos = squeeze(mouseData(f, :, 4));
            spineM_pos = squeeze(mouseData(f, :, 5));
            spineL_pos = squeeze(mouseData(f, :, 6));
            tailBase_pos = squeeze(mouseData(f, :, 7));
            
            % Ensure positions are row vectors
            if size(snout_pos, 1) > size(snout_pos, 2)
                snout_pos = snout_pos';
            end
            if size(spineF_pos, 1) > size(spineF_pos, 2)
                spineF_pos = spineF_pos';
            end
            if size(spineM_pos, 1) > size(spineM_pos, 2)
                spineM_pos = spineM_pos';
            end
            if size(spineL_pos, 1) > size(spineL_pos, 2)
                spineL_pos = spineL_pos';
            end
            if size(tailBase_pos, 1) > size(tailBase_pos, 2)
                tailBase_pos = tailBase_pos';
            end
            
            % Weighted COM calculation (more weight on central spine)
            weights = [0.15, 0.2, 0.3, 0.25, 0.1]; % Snout, SpineF, SpineM, SpineL, TailBase
            positions = [snout_pos; spineF_pos; spineM_pos; spineL_pos; tailBase_pos]; % 5x3 matrix
            
            % Calculate weighted COM - positions is 5x3, weights is 1x5
            % We want weights as column vector for proper broadcasting
            weightCol = weights'; % 5x1
            comPositions(f, :) = sum(positions .* weightCol, 1) / sum(weights);
        end
        
        % Calculate COM speeds with smoothing
        if numFrames > 1
            % Calculate raw velocities
            rawSpeeds = [0; sqrt(sum(diff(comPositions).^2, 2))];
            
            % Apply median filter to remove outliers, then smooth
            if numFrames > 5
                comSpeeds = medfilt1(rawSpeeds, 5); % Remove outliers
                comSpeeds = smooth(comSpeeds, min(15, floor(numFrames/10))); % Smooth with adaptive window
            else
                comSpeeds = rawSpeeds;
            end
            
            % Convert to appropriate units (assuming 50Hz frame rate)
            frameRate = 50; % frames per second
            comSpeeds = comSpeeds * frameRate; % Convert to units per second
        end
        
        % Process each frame
        for f = 1:numFrames
            % Get joint positions and ensure proper dimensions
            spineL_pos = squeeze(mouseData(f, :, spineL));
            hipL_pos = squeeze(mouseData(f, :, hipL));
            kneeL_pos = squeeze(mouseData(f, :, kneeL));
            ankleL_pos = squeeze(mouseData(f, :, ankleL));
            footL_pos = squeeze(mouseData(f, :, footL));
            
            hipR_pos = squeeze(mouseData(f, :, hipR));
            kneeR_pos = squeeze(mouseData(f, :, kneeR));
            ankleR_pos = squeeze(mouseData(f, :, ankleR));
            footR_pos = squeeze(mouseData(f, :, footR));
            
            % Ensure all positions are row vectors [1x3]
            if size(spineL_pos, 1) > size(spineL_pos, 2), spineL_pos = spineL_pos'; end
            if size(hipL_pos, 1) > size(hipL_pos, 2), hipL_pos = hipL_pos'; end
            if size(kneeL_pos, 1) > size(kneeL_pos, 2), kneeL_pos = kneeL_pos'; end
            if size(ankleL_pos, 1) > size(ankleL_pos, 2), ankleL_pos = ankleL_pos'; end
            if size(footL_pos, 1) > size(footL_pos, 2), footL_pos = footL_pos'; end
            if size(hipR_pos, 1) > size(hipR_pos, 2), hipR_pos = hipR_pos'; end
            if size(kneeR_pos, 1) > size(kneeR_pos, 2), kneeR_pos = kneeR_pos'; end
            if size(ankleR_pos, 1) > size(ankleR_pos, 2), ankleR_pos = ankleR_pos'; end
            if size(footR_pos, 1) > size(footR_pos, 2), footR_pos = footR_pos'; end
            
            % Calculate left knee angle (SpineL->HipL->KneeL)
            v1_left = spineL_pos - hipL_pos;
            v2_left = kneeL_pos - hipL_pos;
            leftKneeAngles(f) = calculateAngle(v1_left, v2_left);
            
            % Calculate right knee angle (SpineL->HipR->KneeR) 
            v1_right = spineL_pos - hipR_pos;
            v2_right = kneeR_pos - hipR_pos;
            rightKneeAngles(f) = calculateAngle(v1_right, v2_right);
            
            % Calculate leg asymmetry using multiple metrics
            % 1. Height asymmetry (foot height difference)
            heightAsym = footL_pos(3) - footR_pos(3);
            
            % 2. Leg length asymmetry
            leftLegLength = norm(hipL_pos - kneeL_pos) + norm(kneeL_pos - ankleL_pos) + norm(ankleL_pos - footL_pos);
            rightLegLength = norm(hipR_pos - kneeR_pos) + norm(kneeR_pos - ankleR_pos) + norm(ankleR_pos - footR_pos);
            lengthAsym = leftLegLength - rightLegLength;
            
            % 3. Angle asymmetry
            angleAsym = leftKneeAngles(f) - rightKneeAngles(f);
            
            % Combined asymmetry metric (normalized)
            legAsymmetries(f) = heightAsym + 0.1*lengthAsym + 0.01*angleAsym;
            
            % Track watershed region frequency
            if f <= size(z_orig_img, 1)
                x = round(z_orig_img(f, 1));
                y = round(z_orig_img(f, 2));
                
                if x >= 1 && x <= 501 && y >= 1 && y <= 501
                    if LL2(y, x) > 0  % Valid behavioral region
                        regionID = LL(y, x);
                        if regionID > 0
                            if isKey(regionFreq, regionID)
                                regionFreq(regionID) = regionFreq(regionID) + 1;
                            else
                                regionFreq(regionID) = 1;
                            end
                        end
                    end
                end
            end
        end
        
        % Calculate angular speeds (derivative of angles)
        leftAngularSpeeds = [0; abs(diff(leftKneeAngles))];
        rightAngularSpeeds = [0; abs(diff(rightKneeAngles))];
        
        % Apply smoothing
        leftKneeAngles = medfilt1(leftKneeAngles, 5);
        rightKneeAngles = medfilt1(rightKneeAngles, 5);
        leftAngularSpeeds = medfilt1(leftAngularSpeeds, 5);
        rightAngularSpeeds = medfilt1(rightAngularSpeeds, 5);
        legAsymmetries = medfilt1(legAsymmetries, 5);
        
        % Store results
        results.leftKneeAngle{i} = leftKneeAngles;
        results.rightKneeAngle{i} = rightKneeAngles;
        results.leftAngularSpeed{i} = leftAngularSpeeds;
        results.rightAngularSpeed{i} = rightAngularSpeeds;
        results.legAsymmetry{i} = legAsymmetries;
        results.comSpeed{i} = comSpeeds;
        results.regionFrequency{i} = regionFreq;
        
    else
        fprintf('  Warning: No valid mouse data for biomechanical analysis\n');
        results.leftKneeAngle{i} = [];
        results.rightKneeAngle{i} = [];
        results.leftAngularSpeed{i} = [];
        results.rightAngularSpeed{i} = [];
        results.legAsymmetry{i} = [];
        results.comSpeed{i} = [];
        results.regionFrequency{i} = containers.Map('KeyType', 'int32', 'ValueType', 'int32');
    end
end

%% Display quantitative results by week
weeks = unique(mouseWeeks);
for w = 1:length(weeks)
    week = weeks{w};
    weekIdx = strcmp(mouseWeeks, week);
    
    fprintf('\n=== %s QUANTITATIVE ANALYSIS RESULTS ===\n', upper(week));
    fprintf('Mouse\t\tGroup\tCentroid\tDensity\tCorr\tKL-Div\tBhatt\tDispl\n');
    fprintf('\t\t\tDist\tOverlap\t\t\t\n');
    
    weekIndices = find(weekIdx);
    for i = weekIndices'
        fprintf('%s\t%s\t%.2f\t%.3f\t%.3f\t%.3f\t%.3f\t%.2f\n', ...
            mouseLabels{i}, mouseGroups{i}, ...
            results.centroidDist(i), results.densityOverlap(i), ...
            results.densityCorr(i), results.densityKLDiv(i), ...
            results.densityBhattacharyya(i), results.avgDisplacement(i));
    end
end

%% Create comprehensive dot plots with statistical testing for group comparisons
fprintf('\n=== CREATING DOT PLOTS WITH STATISTICAL TESTING ===\n');

% Define metrics for analysis
metrics = {'centroidDist', 'densityOverlap', 'densityCorr', 'densityKLDiv', 'densityBhattacharyya', 'avgDisplacement'};
metricNames = {'Centroid Distance', 'Density Overlap', 'Density Correlation', 'KL Divergence', 'Bhattacharyya Distance', 'Avg Displacement'};
metricUnits = {'pixels', 'ratio', 'correlation', 'nats', 'distance', 'pixels'};

for w = 1:length(weeks)
    week = weeks{w};
    weekIdx = strcmp(mouseWeeks, week);
    weekGroups = unique(mouseGroups(weekIdx));
    
    if length(weekGroups) < 2
        continue; % Skip if less than 2 groups
    end
    
    fprintf('\nCreating dot plots for %s...\n', week);
    
    % Create figure for this week's comparisons
    figure('Name', sprintf('%s - Group Comparisons with Statistics', week), ...
        'Position', [50 50 1800 1200]);
    
    % Calculate subplot layout
    nMetrics = length(metrics);
    nCols = 3;
    nRows = ceil(nMetrics / nCols);
    
    for m = 1:nMetrics
        subplot(nRows, nCols, m);
        
        metric = metrics{m};
        metricName = metricNames{m};
        unit = metricUnits{m};
        
        % Get data for this metric and week
        weekData = results.(metric)(weekIdx);
        weekGroupLabels = mouseGroups(weekIdx);
        weekMouseLabels = mouseLabels(weekIdx);
        
        % Create group data structure
        groupData = struct();
        groupPositions = [];
        allValues = [];
        allGroupNames = {};
        
        for g = 1:length(weekGroups)
            group = weekGroups{g};
            groupIdx = strcmp(weekGroupLabels, group);
            groupData.(group) = weekData(groupIdx);
            
            % Store for plotting
            groupPositions = [groupPositions; g * ones(sum(groupIdx), 1)];
            allValues = [allValues; groupData.(group)];
            allGroupNames = [allGroupNames; repmat({group}, sum(groupIdx), 1)];
        end
        
        % Create dot plot
        hold on;
        
        % Plot individual points
        for g = 1:length(weekGroups)
            group = weekGroups{g};
            values = groupData.(group);
            color = groupColors.(group);
            
            % Add jitter for better visualization
            jitter = 0.1 * (rand(length(values), 1) - 0.5);
            x_pos = g + jitter;
            
            scatter(x_pos, values, 60, color, 'filled', 'MarkerEdgeColor', 'k', ...
                'MarkerEdgeAlpha', 0.3, 'MarkerFaceAlpha', 0.7);
        end
        
        % Add mean and error bars
        groupMeans = zeros(length(weekGroups), 1);
        groupSEMs = zeros(length(weekGroups), 1);
        
        for g = 1:length(weekGroups)
            group = weekGroups{g};
            values = groupData.(group);
            groupMeans(g) = mean(values);
            groupSEMs(g) = std(values) / sqrt(length(values));
            
            % Plot mean as larger circle
            scatter(g, groupMeans(g), 120, groupColors.(group), 's', 'filled', ...
                'MarkerEdgeColor', 'k', 'MarkerEdgeAlpha', 1, 'LineWidth', 2);
            
            % Plot error bars (SEM)
            errorbar(g, groupMeans(g), groupSEMs(g), 'k-', 'LineWidth', 2, 'CapSize', 8);
        end
        
        % Pairwise t-tests between all groups
        if length(weekGroups) >= 2
            fprintf('\n  Pairwise t-tests for %s:\n', metricName);
            ylims = ylim;
            yRange = ylims(2) - ylims(1);
            
            % Calculate all pairwise comparisons
            pairCount = 0;
            for g1 = 1:length(weekGroups)-1
                for g2 = g1+1:length(weekGroups)
                    group1Data = groupData.(weekGroups{g1});
                    group2Data = groupData.(weekGroups{g2});
                    
                    if length(group1Data) >= 2 && length(group2Data) >= 2
                        % Perform t-test
                        if exist('ttest2', 'file')
                            try
                                [~, pValue] = ttest2(group1Data, group2Data);
                                testName = 't-test';
                            catch
                                % Fallback to simple comparison
                                pValue = permutationTest(group1Data, group2Data, 1000);
                                testName = 'permutation';
                            end
                        else
                            % Simple permutation test
                            pValue = permutationTest(group1Data, group2Data, 1000);
                            testName = 'permutation';
                        end
                        
                        % Determine significance
                        if pValue < 0.001
                            sigText = '***';
                        elseif pValue < 0.01
                            sigText = '**';
                        elseif pValue < 0.05
                            sigText = '*';
                        else
                            sigText = 'ns';
                        end
                        
                        fprintf('    %s vs %s: p=%.3f (%s)\n', weekGroups{g1}, weekGroups{g2}, pValue, sigText);
                        
                        % Add significance bar for significant comparisons
                        if pValue < 0.05
                            pairCount = pairCount + 1;
                            ypos = ylims(2) - (0.08 + 0.04 * pairCount) * yRange;
                            
                            % Draw significance bar
                            plot([g1 g2], [ypos ypos], 'k-', 'LineWidth', 1.5);
                            plot([g1 g1], [ypos-0.005*yRange ypos+0.005*yRange], 'k-', 'LineWidth', 1.5);
                            plot([g2 g2], [ypos-0.005*yRange ypos+0.005*yRange], 'k-', 'LineWidth', 1.5);
                            
                            % Add significance text
                            text((g1+g2)/2, ypos + 0.015*yRange, sigText, ...
                                'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 10);
                        end
                    end
                end
            end
        end
        
        % Formatting
        xlim([0.5, length(weekGroups) + 0.5]);
        set(gca, 'XTick', 1:length(weekGroups), 'XTickLabel', weekGroups);
        ylabel(sprintf('%s (%s)', metricName, unit));
        title(sprintf('%s: %s', week, metricName));
        grid on;
        box on;
        
        % Add sample sizes
        ylims = ylim;
        for g = 1:length(weekGroups)
            group = weekGroups{g};
            n = length(groupData.(group));
            text(g, ylims(1) + 0.02 * (ylims(2) - ylims(1)), ...
                sprintf('n=%d', n), 'HorizontalAlignment', 'center', 'FontSize', 8);
        end
    end
    
    % Add overall title
    sgtitle(sprintf('%s: Group Comparisons - Density-Based Metrics', week), 'FontSize', 16, 'FontWeight', 'bold');
    
    % Save the plot
    plotFileName = fullfile(plotsDir, sprintf('%s_group_comparisons_dotplots.png', week));
    saveas(gcf, plotFileName);
    fprintf('Saved plot: %s\n', plotFileName);
end

%% Load watershed boundaries from the main analysis
% Get the same boundaries used in the original analysis
load('train_weekdata.mat', 'Y', 'allD');

% Check if findPointDensity is available, otherwise use simple alternative
if exist('findPointDensity', 'file') == 2
    [xx2, d] = findPointDensity(Y, 1, 501, [-65 65]);
else
    % Simple density calculation if function not available
    fprintf('Using simple density calculation...\n');
    gridSize = 501;
    bounds = [-65 65];
    x = linspace(bounds(1), bounds(2), gridSize);
    [xx2, yy] = meshgrid(x, x);
    
    % Calculate density
    d = zeros(size(xx2));
    h = 1 * (bounds(2) - bounds(1)) / gridSize;
    
    for i = 1:size(Y, 1)
        dist2 = (xx2 - Y(i,1)).^2 + (yy - Y(i,2)).^2;
        d = d + exp(-dist2 / (2*h^2));
    end
    
    d = d / (size(Y,1) * 2*pi*h^2);
end
LL = watershed(-d, 18);
LL2 = LL;
LL2(d < 1e-6) = -1;
LLBW = LL2 == 0;

% Get boundaries
if exist('bwboundaries', 'file') == 2
    LLBWB = bwboundaries(LLBW);
    llbwb = LLBWB(2:end);
    llbwb = combineCells(llbwb');
else
    % Simple boundary detection
    fprintf('Using simple boundary detection...\n');
    [y, x] = find(LLBW);
    llbwb = [x, y];
end

%% Create individual mouse density plots by week
for w = 1:length(weeks)
    week = weeks{w};
    weekIdx = strcmp(mouseWeeks, week);
    weekMice = find(weekIdx);
    
    if isempty(weekMice)
        continue;
    end
    
    figure('Name', sprintf('%s - Individual Mouse Density Maps', week), 'Position', [50 50 1800 1000]);
    
    % Determine subplot layout
    nMice = length(weekMice);
    nCols = 6;
    nRows = ceil(nMice * 2 / nCols);
    
    plotIdx = 1;
    for idx = 1:length(weekMice)
        i = weekMice(idx);
        
        % Original mouse
        subplot(nRows, nCols, plotIdx);
        
        % Background
        imagesc(D);
        hold on;
        
        % Get embedding data
        z_orig = zEmbeddings{i};
        z_orig_img = (z_orig + 65) * 501 / 130;
        
        % Create density map
        [N, xedges, yedges] = histcounts2(z_orig_img(:,1), z_orig_img(:,2), ...
            linspace(1, 501, 80), linspace(1, 501, 80));
        N = imgaussfilt(N', 3);
        
        % Create colored overlay
        color = groupColors.(mouseGroups{i});
        overlay = zeros(size(N,1), size(N,2), 3);
        for c = 1:3
            overlay(:,:,c) = color(c);
        end
        
        % Display density
        h = imagesc(xedges(1:end-1), yedges(1:end-1), overlay);
        set(h, 'AlphaData', N/max(N(:))*0.8);
        
        % Add watershed boundaries
        scatter(llbwb(:,2), llbwb(:,1), 0.5, '.', 'k', 'MarkerEdgeAlpha', 0.3);
        
        axis equal off;
        colormap(gca, flipud(gray));
        caxis([0 6e-4]);
        title(sprintf('%s - Original', mouseLabels{i}), 'Interpreter', 'none');
        
        % Flipped mouse
        subplot(nRows, nCols, plotIdx + 1);
        
        % Background
        imagesc(D);
        hold on;
        
        % Get flipped embedding data
        z_flip = zEmbeddings{i + numMice};
        z_flip_img = (z_flip + 65) * 501 / 130;
        
        % Create density map
        [N, xedges, yedges] = histcounts2(z_flip_img(:,1), z_flip_img(:,2), ...
            linspace(1, 501, 80), linspace(1, 501, 80));
        N = imgaussfilt(N', 3);
        
        % Create colored overlay (darker for flipped)
        overlay = zeros(size(N,1), size(N,2), 3);
        for c = 1:3
            overlay(:,:,c) = color(c) * 0.6;
        end
        
        % Display density
        h = imagesc(xedges(1:end-1), yedges(1:end-1), overlay);
        set(h, 'AlphaData', N/max(N(:))*0.8);
        
        % Add watershed boundaries
        scatter(llbwb(:,2), llbwb(:,1), 0.5, '.', 'k', 'MarkerEdgeAlpha', 0.3);
        
        axis equal off;
        colormap(gca, flipud(gray));
        caxis([0 6e-4]);
        title(sprintf('%s - Flipped', mouseLabels{i}), 'Interpreter', 'none');
        
        plotIdx = plotIdx + 2;
    end
    
    % Save the individual mouse density plots
    plotFileName = fullfile(plotsDir, sprintf('%s_individual_mouse_density_maps.png', week));
    saveas(gcf, plotFileName);
    fprintf('Saved plot: %s\n', plotFileName);
end

%% Create overlay comparison plots by week
for w = 1:length(weeks)
    week = weeks{w};
    weekIdx = strcmp(mouseWeeks, week);
    weekMice = find(weekIdx);
    
    if isempty(weekMice)
        continue;
    end
    
    figure('Name', sprintf('%s - Original vs Flipped Overlay Comparison', week), 'Position', [100 100 1400 900]);
    
    nMice = length(weekMice);
    nCols = 4;
    nRows = ceil(nMice / nCols);
    
    for idx = 1:length(weekMice)
        i = weekMice(idx);
        subplot(nRows, nCols, idx);
        
        % Background
        imagesc(D);
        hold on;
        
        % Get both embeddings
        z_orig_img = (zEmbeddings{i} + 65) * 501 / 130;
        z_flip_img = (zEmbeddings{i + numMice} + 65) * 501 / 130;
        
        % Create density maps
        [N_orig, xedges, yedges] = histcounts2(z_orig_img(:,1), z_orig_img(:,2), ...
            linspace(1, 501, 60), linspace(1, 501, 60));
        [N_flip, ~, ~] = histcounts2(z_flip_img(:,1), z_flip_img(:,2), ...
            linspace(1, 501, 60), linspace(1, 501, 60));
        
        N_orig = imgaussfilt(N_orig', 2);
        N_flip = imgaussfilt(N_flip', 2);
        
        % Create RGB overlay (Original = Red channel, Flipped = Green channel)
        color = groupColors.(mouseGroups{i});
        overlay = zeros(size(N_orig,1), size(N_orig,2), 3);
        overlay(:,:,1) = N_orig / max(N_orig(:));  % Red for original
        overlay(:,:,2) = N_flip / max(N_flip(:));  % Green for flipped
        % Yellow areas show overlap
        
        % Display
        h = imagesc(xedges(1:end-1), yedges(1:end-1), overlay);
        alpha_map = max(N_orig/max(N_orig(:)), N_flip/max(N_flip(:))) * 0.8;
        set(h, 'AlphaData', alpha_map);
        
        % Add boundaries
        scatter(llbwb(:,2), llbwb(:,1), 0.5, '.', 'k', 'MarkerEdgeAlpha', 0.3);
        
        axis equal off;
        colormap(gca, flipud(gray));
        caxis([0 6e-4]);
        title(sprintf('%s\nRed=Orig, Green=Flip', ...
            mouseLabels{i}), 'Interpreter', 'none');
        
        % Add metrics as text
        text(10, 490, sprintf('Overlap: %.2f%%', results.densityOverlap(i)*100), ...
            'Color', 'white', 'FontSize', 8, 'BackgroundColor', 'black');
    end
    
    % Save the overlay comparison plots
    plotFileName = fullfile(plotsDir, sprintf('%s_overlay_comparison.png', week));
    saveas(gcf, plotFileName);
    fprintf('Saved plot: %s\n', plotFileName);
end

%% Create displacement heatmap by week
for w = 1:length(weeks)
    week = weeks{w};
    weekIdx = strcmp(mouseWeeks, week);
    weekGroups = unique(mouseGroups(weekIdx));
    
    if isempty(weekGroups)
        continue;
    end
    
    figure('Name', sprintf('%s - Displacement Analysis', week), 'Position', [150 150 1000 800]);
    
    % Group summary plot
    subplot(2,2,1);
    groupMeans = zeros(length(weekGroups), 4);
    groupStds = zeros(length(weekGroups), 4);
    
    for g = 1:length(weekGroups)
        idx = strcmp(mouseGroups, weekGroups{g}) & strcmp(mouseWeeks, week);
        if sum(idx) > 0
            groupMeans(g,:) = [mean(results.centroidDist(idx)), ...
                               mean(results.densityOverlap(idx)), ...
                               mean(results.densityCorr(idx)), ...
                               mean(results.avgDisplacement(idx))];
            groupStds(g,:) = [std(results.centroidDist(idx)), ...
                              std(results.densityOverlap(idx)), ...
                              std(results.densityCorr(idx)), ...
                              std(results.avgDisplacement(idx))];
        end
    end
    
    % Bar plot with error bars
    x = 1:4;
    b = bar(x, groupMeans');
    hold on;
    
    % Set colors for bars
    for g = 1:length(weekGroups)
        b(g).FaceColor = groupColors.(weekGroups{g});
    end
    
    % Add error bars
    ngroups = size(groupMeans, 2);
    nbars = size(groupMeans, 1);
    groupwidth = min(0.8, nbars/(nbars + 1.5));
    
    for i = 1:nbars
        x_err = (1:ngroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
        errorbar(x_err, groupMeans(i,:), groupStds(i,:), 'k', 'linestyle', 'none');
    end
    
    set(gca, 'XTickLabel', {'Centroid Dist', 'Overlap', 'Correlation', 'Avg Displ'});
    ylabel('Value');
    legend(weekGroups, 'Location', 'best');
    title(sprintf('%s Group Comparison Metrics', week));
    
    % Individual displacement traces
    subplot(2,2,[3,4]);
    hold on;
    
    weekMice = find(weekIdx);
    for idx = 1:length(weekMice)
        i = weekMice(idx);
        z_orig_img = (zEmbeddings{i} + 65) * 501 / 130;
        z_flip_img = (zEmbeddings{i + numMice} + 65) * 501 / 130;
        
        % Calculate displacement over time
        displacements = sqrt(sum((z_orig_img - z_flip_img).^2, 2));
        
        % Smooth and downsample for visualization
        displacements_smooth = smooth(displacements, 50);
        t = 1:100:length(displacements_smooth);
        
        color = groupColors.(mouseGroups{i});
        plot(t, displacements_smooth(t), 'Color', color, 'LineWidth', 1.5);
    end
    
    xlabel('Time (frames)');
    ylabel('Displacement (pixels)');
    title(sprintf('%s - Temporal Displacement Between Original and Flipped', week));
    
    % Create legend entries
    legendEntries = cell(length(weekMice), 1);
    for idx = 1:length(weekMice)
        i = weekMice(idx);
        legendEntries{idx} = mouseLabels{i};
    end
    legend(legendEntries, 'Interpreter', 'none', 'Location', 'eastoutside');
    grid on;
    
    % Save the displacement analysis plot
    plotFileName = fullfile(plotsDir, sprintf('%s_displacement_analysis.png', week));
    saveas(gcf, plotFileName);
    fprintf('Saved plot: %s\n', plotFileName);
end

%% Create watershed region velocity heatmap
fprintf('\nCreating watershed region velocity analysis...\n');

% Load the actual mouse data to calculate velocities
allMOUSE = cell(numMice, 1);

for i = 1:numMice
    matPath = mouseFiles{i};
    data = load(matPath);
    if isfield(data, 'pred')
        allMOUSE{i} = data.pred;
    end
end

% Initialize velocity maps for each watershed region
numRegions = max(LL(:));
regionVelocities = cell(numRegions, 1);
regionCounts = zeros(numRegions, 1);

% Get all unique positive values from LL2 - these are the valid regions
validRegionIDs = unique(LL2(LL2 > 0));

fprintf('Valid behavioral regions: %d\n', length(validRegionIDs));

% Store wrFINE for frame extraction later
wrFINE_stored = wrFINE;

% Process each mouse
for m = 1:numMice
    fprintf('Processing velocities for mouse %d/%d...\n', m, numMice);
    
    % Get embedding and mouse data
    z = zEmbeddings{m};
    z_img = (z + 65) * 501 / 130;
    mouseData = allMOUSE{m};
    
    % Calculate velocities (using center of mass or specific joint)
    % Using joint 5 (SpineM) as representative of body center
    positions = squeeze(mouseData(:, :, 5));
    velocities = [0; sqrt(sum(diff(positions).^2, 2))];
    
    % Smooth velocities
    velocities = smooth(velocities, 10);
    
    % Scale velocities to mm/s if needed (assuming 50Hz frame rate)
    frameRate = 50;  % frames per second
    velocities = velocities * frameRate;  % Convert to units per second
    
    % Map each time point to a watershed region
    for t = 1:length(velocities)
        % Find which region this point belongs to
        x = round(z_img(t, 1));
        y = round(z_img(t, 2));
        
        % Check bounds
        if x >= 1 && x <= 501 && y >= 1 && y <= 501
            % Check LL2 to ensure this is a valid behavioral region
            if LL2(y, x) > 0  % Valid behavioral region (not boundary or low-density)
                regionID = LL(y, x);  % Get the actual region ID from LL
                
                if regionID > 0 && regionID <= numRegions
                    if isempty(regionVelocities{regionID})
                        regionVelocities{regionID} = [];
                    end
                    regionVelocities{regionID} = [regionVelocities{regionID}; velocities(t)];
                    regionCounts(regionID) = regionCounts(regionID) + 1;
                end
            end
        end
    end
end

% Calculate average velocity for each region
avgVelocities = zeros(numRegions, 1);
for r = 1:numRegions
    if ~isempty(regionVelocities{r})
        avgVelocities(r) = mean(regionVelocities{r});
    else
        avgVelocities(r) = NaN;
    end
end

% Create velocity heatmap
figure('Name', 'Watershed Region Velocity Heatmap', 'Position', [200 200 800 800]);

% First display the background density map (same as original)
imagesc(D);
hold on;
colormap(gca, flipud(gray));
caxis([0 6e-4]);

% Create a map where each watershed region is colored by its average velocity
velocityMap = nan(size(LL));  % Use NaN for regions outside watersheds
for i = 1:length(validRegionIDs)
    r = validRegionIDs(i);
    if r > 0 && r <= numRegions && ~isnan(avgVelocities(r)) && regionCounts(r) > 10
        % Use LL2 to identify valid pixels
        regionMask = (LL2 == r);
        velocityMap(regionMask) = avgVelocities(r);
    end
end

% Normalize velocities for better visualization
validVels = avgVelocities(~isnan(avgVelocities));
if ~isempty(validVels)
    if exist('prctile', 'file')
        vMin = prctile(validVels, 5);
        vMax = prctile(validVels, 95);
    else
        % Simple percentile calculation
        sortedVels = sort(validVels);
        vMin = sortedVels(max(1, round(0.05 * length(sortedVels))));
        vMax = sortedVels(min(length(sortedVels), round(0.95 * length(sortedVels))));
    end
else
    vMin = 0;
    vMax = 1;
end

% Create RGB overlay for velocity
velocityRGB = zeros(size(LL,1), size(LL,2), 3);
jetMap = jet(256);

% Only color regions that are in validRegionIDs AND have actual data
finalValidRegions = [];
for i = 1:length(validRegionIDs)
    r = validRegionIDs(i);
    % Only include if we have velocity data (meaning mice actually visited this region)
    if r > 0 && r <= numRegions && ~isnan(avgVelocities(r)) && regionCounts(r) > 10
        finalValidRegions = [finalValidRegions; r];
    end
end

% Now color only the final valid regions, using LL2 to identify valid pixels
for i = 1:length(finalValidRegions)
    r = finalValidRegions(i);
    if r > 0 && r <= numRegions
        % Use LL2 to find pixels - only color where LL2 == r
        regionMask = (LL2 == r);
        
        % Normalize velocity to [0, 1]
        normVel = (avgVelocities(r) - vMin) / (vMax - vMin);
        normVel = max(0, min(1, normVel));  % Clamp to [0, 1]
        
        % Get color from jet colormap
        colorIdx = round(normVel * 255) + 1;
        regionColor = jetMap(colorIdx, :);
        
        % Apply color to this region
        for c = 1:3
            velocityRGB(:,:,c) = velocityRGB(:,:,c) + regionMask * regionColor(c);
        end
    end
end

% Create alpha mask using LL2
alphaMask = zeros(size(LL));
for i = 1:length(finalValidRegions)
    r = finalValidRegions(i);
    alphaMask = alphaMask | (LL2 == r);
end

% Overlay velocity colors on the density map
h = imagesc(velocityRGB);
set(h, 'AlphaData', alphaMask * 0.7);

% Add watershed boundaries
scatter(llbwb(:,2), llbwb(:,1), 1, '.', 'k', 'MarkerEdgeAlpha', 0.8);

% Add simple colorbar for velocity reference
figure('Name', 'Velocity Scale', 'Position', [1020 400 100 400]);
imagesc([vMin; vMax]);
colormap(jet);
c = colorbar;
ylabel(c, 'Average Velocity (units/s)');
set(gca, 'Visible', 'off');
title('Velocity Scale');

% Return to main figure
figure(findobj('Name', 'Watershed Region Velocity Heatmap'));

axis equal off;
title('Average Velocity by Behavioral Region');

% Add color scale annotation
text(10, 30, sprintf('Velocity range: %.1f - %.1f units/s\nBlue=Slow, Red=Fast', vMin, vMax), ...
    'Color', 'white', 'FontSize', 10, 'BackgroundColor', [0 0 0 0.7]);

% Save the velocity heatmap
plotFileName = fullfile(plotsDir, 'watershed_region_velocity_heatmap.png');
saveas(gcf, plotFileName);
fprintf('Saved plot: %s\n', plotFileName);

% Add text labels for regions with sufficient data AND within final valid regions
for i = 1:length(finalValidRegions)
    r = finalValidRegions(i);
    if regionCounts(r) > 100
        % Find centroid of region using LL2
        [y, x] = find(LL2 == r);
        if ~isempty(x) && ~isempty(y)
            cx = mean(x);
            cy = mean(y);
            text(cx, cy, sprintf('%.1f', avgVelocities(r)), ...
                'Color', 'white', 'FontSize', 8, 'HorizontalAlignment', 'center', ...
                'BackgroundColor', [0 0 0 0.5]);
        end
    end
end

%% Group overlap analysis by week - Identify unique regions per group within each week
for w = 1:length(weeks)
    week = weeks{w};
    weekIdx = strcmp(mouseWeeks, week);
    weekGroups = unique(mouseGroups(weekIdx));
    
    if isempty(weekGroups)
        continue;
    end
    
    fprintf('\n\n=== %s GROUP OVERLAP ANALYSIS ===\n', upper(week));
    fprintf('Analyzing which regions are unique to each group in %s...\n', week);
    
    % Track which regions each group visits
    groupRegionVisits = struct();
    groupRegionCounts = struct();
    for g = 1:length(weekGroups)
        groupRegionVisits.(weekGroups{g}) = [];
        groupRegionCounts.(weekGroups{g}) = zeros(numRegions, 1);
    end
    
    % Process each mouse to see which regions they visit
    weekMice = find(weekIdx);
    for idx = 1:length(weekMice)
        m = weekMice(idx);
        group = mouseGroups{m};
        z_img = (zEmbeddings{m} + 65) * 501 / 130;
        
        % Track which regions this mouse visits
        for t = 1:size(z_img, 1)
            x = round(z_img(t, 1));
            y = round(z_img(t, 2));
            
            if x >= 1 && x <= 501 && y >= 1 && y <= 501
                if LL2(y, x) > 0  % Valid behavioral region
                    regionID = LL(y, x);
                    groupRegionCounts.(group)(regionID) = groupRegionCounts.(group)(regionID) + 1;
                end
            end
        end
    end
    
    % Identify regions visited by each group (with minimum threshold)
    minVisits = 50;  % Minimum visits to consider a region as "visited"
    for g = 1:length(weekGroups)
        group = weekGroups{g};
        visitedRegions = find(groupRegionCounts.(group) > minVisits);
        groupRegionVisits.(group) = visitedRegions;
        fprintf('\n%s group visits %d regions\n', group, length(visitedRegions));
    end
    
    % Find unique regions for each group within week
    fprintf('\nFinding unique regions for each group in %s...\n', week);
    uniqueRegions = struct();
    
    for g = 1:length(weekGroups)
        group = weekGroups{g};
        thisGroupRegions = groupRegionVisits.(group);
        
        % Find regions that only this group visits
        otherGroupsRegions = [];
        for g2 = 1:length(weekGroups)
            if g2 ~= g
                otherGroupsRegions = [otherGroupsRegions; groupRegionVisits.(weekGroups{g2})];
            end
        end
        otherGroupsRegions = unique(otherGroupsRegions);
        
        % Unique regions are those visited by this group but not others
        uniqueRegions.(group) = setdiff(thisGroupRegions, otherGroupsRegions);
        
        fprintf('%s has %d unique regions: ', group, length(uniqueRegions.(group)));
        if length(uniqueRegions.(group)) > 0
            fprintf('%s\n', mat2str(uniqueRegions.(group)'));
        else
            fprintf('none\n');
        end
    end
    
    % Visualize unique regions for this week
    figure('Name', sprintf('%s - Group Unique Regions', week), 'Position', [300 300 1200 600]);
    for g = 1:length(weekGroups)
        subplot(1, length(weekGroups), g);
        
        % Background
        imagesc(D);
        hold on;
        colormap(gca, flipud(gray));
        caxis([0 6e-4]);
        
        % Highlight unique regions for this group
        group = weekGroups{g};
        uniqueRegionList = uniqueRegions.(group);
        
        if ~isempty(uniqueRegionList)
            % Create mask for unique regions
            uniqueMask = zeros(size(LL));
            for r = uniqueRegionList'
                uniqueMask = uniqueMask | (LL2 == r);
            end
            
            % Create colored overlay
            color = groupColors.(group);
            overlay = zeros(size(LL,1), size(LL,2), 3);
            for c = 1:3
                overlay(:,:,c) = uniqueMask * color(c);
            end
            
            % Display overlay
            h = imagesc(overlay);
            set(h, 'AlphaData', uniqueMask * 0.8);
        end
        
        % Add watershed boundaries
        scatter(llbwb(:,2), llbwb(:,1), 0.5, '.', 'k', 'MarkerEdgeAlpha', 0.6);
        
        axis equal off;
        title(sprintf('%s %s Unique Regions (%d)', week, group, length(uniqueRegionList)));
    end
    
    % Save the unique regions plot
    plotFileName = fullfile(plotsDir, sprintf('%s_group_unique_regions.png', week));
    saveas(gcf, plotFileName);
    fprintf('Saved plot: %s\n', plotFileName);
    
    % Visualize pairwise group overlaps within week
    if length(weekGroups) > 1
        fprintf('\nAnalyzing pairwise group overlaps in %s...\n', week);
        
        % Calculate number of pairs
        nPairs = nchoosek(length(weekGroups), 2);
        
        figure('Name', sprintf('%s - Pairwise Group Overlaps', week), 'Position', [350 350 1200 600]);
        
        pairIdx = 1;
        for g1 = 1:length(weekGroups)-1
            for g2 = g1+1:length(weekGroups)
                subplot(1, nPairs, pairIdx);
                
                % Background
                imagesc(D);
                hold on;
                colormap(gca, flipud(gray));
                caxis([0 6e-4]);
                
                % Get the two groups
                group1 = weekGroups{g1};
                group2 = weekGroups{g2};
                
                % Find overlapping regions (visited by both groups)
                overlapRegions = intersect(groupRegionVisits.(group1), groupRegionVisits.(group2));
                
                fprintf('%s-%s overlap: %d regions\n', group1, group2, length(overlapRegions));
                
                if ~isempty(overlapRegions)
                    % Create mask for overlap regions
                    overlapMask = zeros(size(LL));
                    for r = overlapRegions'
                        overlapMask = overlapMask | (LL2 == r);
                    end
                    
                    % Create purple overlay for overlap (mix of both colors)
                    overlay = zeros(size(LL,1), size(LL,2), 3);
                    color1 = groupColors.(group1);
                    color2 = groupColors.(group2);
                    mixedColor = (color1 + color2) / 2;
                    
                    for c = 1:3
                        overlay(:,:,c) = overlapMask * mixedColor(c);
                    end
                    
                    % Display overlay
                    h = imagesc(overlay);
                    set(h, 'AlphaData', overlapMask * 0.8);
                end
                
                % Add watershed boundaries
                scatter(llbwb(:,2), llbwb(:,1), 0.5, '.', 'k', 'MarkerEdgeAlpha', 0.6);
                
                axis equal off;
                title(sprintf('%s-%s Overlap (%d regions)', group1, group2, length(overlapRegions)));
                
                pairIdx = pairIdx + 1;
            end
        end
        
        % Save the pairwise overlaps plot
        plotFileName = fullfile(plotsDir, sprintf('%s_pairwise_group_overlaps.png', week));
        saveas(gcf, plotFileName);
        fprintf('Saved plot: %s\n', plotFileName);
    end
    
    % Save week-specific results
    save(sprintf('flip_analysis_results_%s.mat', week), 'uniqueRegions', 'groupRegionVisits');
end

%% Frame extraction for specific behavioral regions
fprintf('\n\n=== FRAME EXTRACTION FOR BEHAVIORAL REGIONS ===\n');

% Function to extract frames for a specific region
extractFramesForRegion = @(regionID, numFrames) extractRegionFrames(regionID, numFrames, ...
    wrFINE_stored, LL, LL2, zEmbeddings, mouseFiles, allMOUSE, numMice);

% Example: Extract frames for a specific region
fprintf('Frame extraction function ready. Use:\n');
fprintf('  frames = extractFramesForRegion(regionID, numFrames);\n');
fprintf('where regionID is one of the numbered regions from the velocity heatmap.\n');

% Create a separate figure showing region numbers
figure('Name', 'Behavioral Region Index Map', 'Position', [400 400 800 800]);

% Background
imagesc(D);
hold on;
colormap(gca, flipud(gray));
caxis([0 6e-4]);

% Add watershed boundaries
scatter(llbwb(:,2), llbwb(:,1), 1, '.', 'k', 'MarkerEdgeAlpha', 0.8);

% Add region numbers for all valid regions
fprintf('\nCreating region index map...\n');
for i = 1:length(finalValidRegions)
    r = finalValidRegions(i);
    % Find centroid of region
    [y, x] = find(LL2 == r);
    if ~isempty(x) && ~isempty(y)
        cx = mean(x);
        cy = mean(y);
        
        % Determine text color based on region velocity for visibility
        if ~isnan(avgVelocities(r))
            normVel = (avgVelocities(r) - vMin) / (vMax - vMin);
            if normVel > 0.5
                textColor = 'white';  % White text for fast (red) regions
            else
                textColor = 'black';  % Black text for slow (blue) regions
            end
        else
            textColor = 'yellow';
        end
        
        text(cx, cy, num2str(r), ...
            'Color', textColor, 'FontSize', 9, 'FontWeight', 'bold', ...
            'HorizontalAlignment', 'center', ...
            'BackgroundColor', [1 1 1 0.3]);
    end
end

axis equal off;
title('Behavioral Region Index Map');
text(10, 30, sprintf('Total regions: %d', length(finalValidRegions)), ...
    'Color', 'white', 'FontSize', 10, 'BackgroundColor', [0 0 0 0.7]);

% Save the region index map
plotFileName = fullfile(plotsDir, 'behavioral_region_index_map.png');
saveas(gcf, plotFileName);
fprintf('Saved plot: %s\n', plotFileName);

% Function to export all regions to folders
exportAllRegionsToFolders = @() exportAllRegions(finalValidRegions, wrFINE_stored, ...
    LL, LL2, zEmbeddings, mouseFiles, allMOUSE, numMice, avgVelocities, regionCounts);

% Function to export with visualizations
exportAllRegionsWithVideos = @() exportAllRegions(finalValidRegions, wrFINE_stored, ...
    LL, LL2, zEmbeddings, mouseFiles, allMOUSE, numMice, avgVelocities, regionCounts, true);

fprintf('\nTo export frame samples for all regions to folders, use:\n');
fprintf('  exportAllRegionsToFolders();        %% Export MAT files only\n');
fprintf('  exportAllRegionsWithVideos();      %% Export MAT files + videos\n');

%% NEW: Comprehensive region-based analysis
fprintf('\n\n=== COMPREHENSIVE REGION-BASED ANALYSIS ===\n');
fprintf('Calculating metrics for each watershed region...\n');

% Initialize region-based analysis
allRegionIDs = finalValidRegions;
numRegions = length(allRegionIDs);

% Initialize data structures
regionData = struct();
for r = 1:numRegions
    regionID = allRegionIDs(r);
    regionData(r).regionID = regionID;
    regionData(r).groupCounts = containers.Map();
    regionData(r).individualCounts = containers.Map();
    regionData(r).leftKneeAngles = [];
    regionData(r).rightKneeAngles = [];
    regionData(r).angularSpeeds = [];           % Average of left and right
    regionData(r).leftAngularSpeeds = [];       % Left side only
    regionData(r).legAsymmetries = [];
    regionData(r).comSpeeds = [];               % Center of mass speeds
    regionData(r).totalFrames = 0;
end

% Process each mouse to collect region-specific data
for i = 1:numMice
    if isempty(results.regionFrequency{i})
        continue;
    end
    
    group = mouseGroups{i};
    week = mouseWeeks{i};
    mouseLabel = mouseLabels{i};
    
    % Get region frequencies for this mouse
    regionFreq = results.regionFrequency{i};
    visitedRegions = keys(regionFreq);
    
    % Get biomechanical data for this mouse
    leftAngles = results.leftKneeAngle{i};
    rightAngles = results.rightKneeAngle{i};
    leftSpeeds = results.leftAngularSpeed{i};
    rightSpeeds = results.rightAngularSpeed{i};
    asymmetries = results.legAsymmetry{i};
    comSpeedsData = results.comSpeed{i};
    
    % Map each frame to a region and collect data
    z_img = (zEmbeddings{i} + 65) * 501 / 130;
    
    for f = 1:min(length(leftAngles), size(z_img, 1))
        x = round(z_img(f, 1));
        y = round(z_img(f, 2));
        
        if x >= 1 && x <= 501 && y >= 1 && y <= 501
            if LL2(y, x) > 0  % Valid behavioral region
                regionID = LL(y, x);
                
                % Find this region in our data structure
                regionIdx = find([regionData.regionID] == regionID);
                if ~isempty(regionIdx)
                    % Add group count
                    if isKey(regionData(regionIdx).groupCounts, group)
                        regionData(regionIdx).groupCounts(group) = regionData(regionIdx).groupCounts(group) + 1;
                    else
                        regionData(regionIdx).groupCounts(group) = 1;
                    end
                    
                    % Add individual count
                    if isKey(regionData(regionIdx).individualCounts, mouseLabel)
                        regionData(regionIdx).individualCounts(mouseLabel) = regionData(regionIdx).individualCounts(mouseLabel) + 1;
                    else
                        regionData(regionIdx).individualCounts(mouseLabel) = 1;
                    end
                    
                    % Add biomechanical data
                    regionData(regionIdx).leftKneeAngles = [regionData(regionIdx).leftKneeAngles; leftAngles(f)];
                    regionData(regionIdx).rightKneeAngles = [regionData(regionIdx).rightKneeAngles; rightAngles(f)];
                    
                    % Store both average and left-only angular speeds
                    avgSpeed = (leftSpeeds(f) + rightSpeeds(f)) / 2;
                    regionData(regionIdx).angularSpeeds = [regionData(regionIdx).angularSpeeds; avgSpeed];
                    regionData(regionIdx).leftAngularSpeeds = [regionData(regionIdx).leftAngularSpeeds; leftSpeeds(f)];
                    
                    regionData(regionIdx).legAsymmetries = [regionData(regionIdx).legAsymmetries; asymmetries(f)];
                    
                    % Add COM speed data
                    if ~isempty(comSpeedsData) && f <= length(comSpeedsData)
                        regionData(regionIdx).comSpeeds = [regionData(regionIdx).comSpeeds; comSpeedsData(f)];
                    end
                    
                    regionData(regionIdx).totalFrames = regionData(regionIdx).totalFrames + 1;
                end
            end
        end
    end
end

% Calculate comprehensive summary statistics for each region
fprintf('\nCalculating comprehensive region summary statistics...\n');
regionSummary = struct();
regionSummary.regionID = zeros(numRegions, 1);
regionSummary.totalFrames = zeros(numRegions, 1);
regionSummary.avgLeftKneeAngle = zeros(numRegions, 1);
regionSummary.avgRightKneeAngle = zeros(numRegions, 1);
regionSummary.avgAngularSpeed = zeros(numRegions, 1);        % Average of both legs
regionSummary.avgLeftAngularSpeed = zeros(numRegions, 1);    % Left side only
regionSummary.avgLegAsymmetry = zeros(numRegions, 1);
regionSummary.avgComSpeed = zeros(numRegions, 1);           % Center of mass speed

% Standard deviations
regionSummary.stdLeftKneeAngle = zeros(numRegions, 1);
regionSummary.stdRightKneeAngle = zeros(numRegions, 1);
regionSummary.stdAngularSpeed = zeros(numRegions, 1);        % Average of both legs
regionSummary.stdLeftAngularSpeed = zeros(numRegions, 1);    % Left side only
regionSummary.stdLegAsymmetry = zeros(numRegions, 1);
regionSummary.stdComSpeed = zeros(numRegions, 1);           % Center of mass speed

% Additional statistics - Medians
regionSummary.medianLeftKneeAngle = zeros(numRegions, 1);
regionSummary.medianRightKneeAngle = zeros(numRegions, 1);
regionSummary.medianAngularSpeed = zeros(numRegions, 1);
regionSummary.medianLeftAngularSpeed = zeros(numRegions, 1);
regionSummary.medianLegAsymmetry = zeros(numRegions, 1);
regionSummary.medianComSpeed = zeros(numRegions, 1);

% Min/Max values
regionSummary.minLeftKneeAngle = zeros(numRegions, 1);
regionSummary.maxLeftKneeAngle = zeros(numRegions, 1);
regionSummary.minRightKneeAngle = zeros(numRegions, 1);
regionSummary.maxRightKneeAngle = zeros(numRegions, 1);
regionSummary.minAngularSpeed = zeros(numRegions, 1);
regionSummary.maxAngularSpeed = zeros(numRegions, 1);
regionSummary.minLeftAngularSpeed = zeros(numRegions, 1);
regionSummary.maxLeftAngularSpeed = zeros(numRegions, 1);
regionSummary.minLegAsymmetry = zeros(numRegions, 1);
regionSummary.maxLegAsymmetry = zeros(numRegions, 1);
regionSummary.minComSpeed = zeros(numRegions, 1);
regionSummary.maxComSpeed = zeros(numRegions, 1);

% Percentiles (25th and 75th)
regionSummary.p25LeftKneeAngle = zeros(numRegions, 1);
regionSummary.p75LeftKneeAngle = zeros(numRegions, 1);
regionSummary.p25RightKneeAngle = zeros(numRegions, 1);
regionSummary.p75RightKneeAngle = zeros(numRegions, 1);
regionSummary.p25AngularSpeed = zeros(numRegions, 1);
regionSummary.p75AngularSpeed = zeros(numRegions, 1);
regionSummary.p25LeftAngularSpeed = zeros(numRegions, 1);
regionSummary.p75LeftAngularSpeed = zeros(numRegions, 1);
regionSummary.p25ComSpeed = zeros(numRegions, 1);
regionSummary.p75ComSpeed = zeros(numRegions, 1);

% Frequency statistics
regionSummary.uniqueMousesVisited = zeros(numRegions, 1);
regionSummary.totalGroupsVisited = zeros(numRegions, 1);
regionSummary.dominantGroup = cell(numRegions, 1);
regionSummary.dominantGroupPercentage = zeros(numRegions, 1);

for r = 1:numRegions
    regionSummary.regionID(r) = regionData(r).regionID;
    regionSummary.totalFrames(r) = regionData(r).totalFrames;
    
    if regionData(r).totalFrames > 10  % Only calculate for regions with sufficient data
        % Mean values
        regionSummary.avgLeftKneeAngle(r) = mean(regionData(r).leftKneeAngles);
        regionSummary.avgRightKneeAngle(r) = mean(regionData(r).rightKneeAngles);
        regionSummary.avgAngularSpeed(r) = mean(regionData(r).angularSpeeds);           % Average of both legs
        regionSummary.avgLeftAngularSpeed(r) = mean(regionData(r).leftAngularSpeeds);  % Left side only
        regionSummary.avgLegAsymmetry(r) = mean(regionData(r).legAsymmetries);
        if ~isempty(regionData(r).comSpeeds)
            regionSummary.avgComSpeed(r) = mean(regionData(r).comSpeeds);
        else
            regionSummary.avgComSpeed(r) = NaN;
        end
        
        % Standard deviations
        regionSummary.stdLeftKneeAngle(r) = std(regionData(r).leftKneeAngles);
        regionSummary.stdRightKneeAngle(r) = std(regionData(r).rightKneeAngles);
        regionSummary.stdAngularSpeed(r) = std(regionData(r).angularSpeeds);           % Average of both legs
        regionSummary.stdLeftAngularSpeed(r) = std(regionData(r).leftAngularSpeeds);  % Left side only
        regionSummary.stdLegAsymmetry(r) = std(regionData(r).legAsymmetries);
        if ~isempty(regionData(r).comSpeeds) && length(regionData(r).comSpeeds) > 1
            regionSummary.stdComSpeed(r) = std(regionData(r).comSpeeds);
        else
            regionSummary.stdComSpeed(r) = NaN;
        end
        
        % Median values
        regionSummary.medianLeftKneeAngle(r) = median(regionData(r).leftKneeAngles);
        regionSummary.medianRightKneeAngle(r) = median(regionData(r).rightKneeAngles);
        regionSummary.medianAngularSpeed(r) = median(regionData(r).angularSpeeds);
        regionSummary.medianLeftAngularSpeed(r) = median(regionData(r).leftAngularSpeeds);
        regionSummary.medianLegAsymmetry(r) = median(regionData(r).legAsymmetries);
        if ~isempty(regionData(r).comSpeeds)
            regionSummary.medianComSpeed(r) = median(regionData(r).comSpeeds);
        else
            regionSummary.medianComSpeed(r) = NaN;
        end
        
        % Min/Max values
        regionSummary.minLeftKneeAngle(r) = min(regionData(r).leftKneeAngles);
        regionSummary.maxLeftKneeAngle(r) = max(regionData(r).leftKneeAngles);
        regionSummary.minRightKneeAngle(r) = min(regionData(r).rightKneeAngles);
        regionSummary.maxRightKneeAngle(r) = max(regionData(r).rightKneeAngles);
        regionSummary.minAngularSpeed(r) = min(regionData(r).angularSpeeds);
        regionSummary.maxAngularSpeed(r) = max(regionData(r).angularSpeeds);
        regionSummary.minLeftAngularSpeed(r) = min(regionData(r).leftAngularSpeeds);
        regionSummary.maxLeftAngularSpeed(r) = max(regionData(r).leftAngularSpeeds);
        regionSummary.minLegAsymmetry(r) = min(regionData(r).legAsymmetries);
        regionSummary.maxLegAsymmetry(r) = max(regionData(r).legAsymmetries);
        if ~isempty(regionData(r).comSpeeds)
            regionSummary.minComSpeed(r) = min(regionData(r).comSpeeds);
            regionSummary.maxComSpeed(r) = max(regionData(r).comSpeeds);
        else
            regionSummary.minComSpeed(r) = NaN;
            regionSummary.maxComSpeed(r) = NaN;
        end
        
        % Percentiles (25th and 75th)
        if exist('prctile', 'file')
            regionSummary.p25LeftKneeAngle(r) = prctile(regionData(r).leftKneeAngles, 25);
            regionSummary.p75LeftKneeAngle(r) = prctile(regionData(r).leftKneeAngles, 75);
            regionSummary.p25RightKneeAngle(r) = prctile(regionData(r).rightKneeAngles, 25);
            regionSummary.p75RightKneeAngle(r) = prctile(regionData(r).rightKneeAngles, 75);
            regionSummary.p25AngularSpeed(r) = prctile(regionData(r).angularSpeeds, 25);
            regionSummary.p75AngularSpeed(r) = prctile(regionData(r).angularSpeeds, 75);
            regionSummary.p25LeftAngularSpeed(r) = prctile(regionData(r).leftAngularSpeeds, 25);
            regionSummary.p75LeftAngularSpeed(r) = prctile(regionData(r).leftAngularSpeeds, 75);
            if ~isempty(regionData(r).comSpeeds)
                regionSummary.p25ComSpeed(r) = prctile(regionData(r).comSpeeds, 25);
                regionSummary.p75ComSpeed(r) = prctile(regionData(r).comSpeeds, 75);
            else
                regionSummary.p25ComSpeed(r) = NaN;
                regionSummary.p75ComSpeed(r) = NaN;
            end
        else
            % Manual percentile calculation if prctile not available
                         sortedLeft = sort(regionData(r).leftKneeAngles);
             sortedRight = sort(regionData(r).rightKneeAngles);
             sortedSpeed = sort(regionData(r).angularSpeeds);
             sortedLeftSpeed = sort(regionData(r).leftAngularSpeeds);
             if ~isempty(regionData(r).comSpeeds)
                 sortedComSpeed = sort(regionData(r).comSpeeds);
             else
                 sortedComSpeed = [];
             end
            
            n = length(sortedLeft);
            if n > 4
                regionSummary.p25LeftKneeAngle(r) = sortedLeft(round(0.25*n));
                regionSummary.p75LeftKneeAngle(r) = sortedLeft(round(0.75*n));
                regionSummary.p25RightKneeAngle(r) = sortedRight(round(0.25*n));
                regionSummary.p75RightKneeAngle(r) = sortedRight(round(0.75*n));
                regionSummary.p25AngularSpeed(r) = sortedSpeed(round(0.25*n));
                                 regionSummary.p75AngularSpeed(r) = sortedSpeed(round(0.75*n));
                 regionSummary.p25LeftAngularSpeed(r) = sortedLeftSpeed(round(0.25*n));
                 regionSummary.p75LeftAngularSpeed(r) = sortedLeftSpeed(round(0.75*n));
                 if ~isempty(sortedComSpeed) && length(sortedComSpeed) >= 4
                     nCom = length(sortedComSpeed);
                     regionSummary.p25ComSpeed(r) = sortedComSpeed(round(0.25*nCom));
                     regionSummary.p75ComSpeed(r) = sortedComSpeed(round(0.75*nCom));
                 else
                     regionSummary.p25ComSpeed(r) = NaN;
                     regionSummary.p75ComSpeed(r) = NaN;
                 end
                         else
                 regionSummary.p25LeftKneeAngle(r) = NaN;
                 regionSummary.p75LeftKneeAngle(r) = NaN;
                 regionSummary.p25RightKneeAngle(r) = NaN;
                 regionSummary.p75RightKneeAngle(r) = NaN;
                 regionSummary.p25AngularSpeed(r) = NaN;
                 regionSummary.p75AngularSpeed(r) = NaN;
                 regionSummary.p25LeftAngularSpeed(r) = NaN;
                 regionSummary.p75LeftAngularSpeed(r) = NaN;
                 regionSummary.p25ComSpeed(r) = NaN;
                 regionSummary.p75ComSpeed(r) = NaN;
             end
        end
        
        % Frequency statistics
        groupKeys = keys(regionData(r).groupCounts);
        regionSummary.totalGroupsVisited(r) = length(groupKeys);
        regionSummary.uniqueMousesVisited(r) = length(keys(regionData(r).individualCounts));
        
        if ~isempty(groupKeys)
            % Find dominant group
            maxCount = 0;
            dominantGroup = '';
            totalVisits = 0;
            
            for g = 1:length(groupKeys)
                groupCount = regionData(r).groupCounts(groupKeys{g});
                totalVisits = totalVisits + groupCount;
                if groupCount > maxCount
                    maxCount = groupCount;
                    dominantGroup = groupKeys{g};
                end
            end
            
            regionSummary.dominantGroup{r} = dominantGroup;
            regionSummary.dominantGroupPercentage(r) = (maxCount / totalVisits) * 100;
        else
            regionSummary.dominantGroup{r} = 'None';
            regionSummary.dominantGroupPercentage(r) = 0;
        end
        
    else
        % Set all values to NaN for regions with insufficient data
        regionSummary.avgLeftKneeAngle(r) = NaN;
        regionSummary.avgRightKneeAngle(r) = NaN;
        regionSummary.avgAngularSpeed(r) = NaN;
        regionSummary.avgLeftAngularSpeed(r) = NaN;
        regionSummary.avgLegAsymmetry(r) = NaN;
        regionSummary.stdLeftKneeAngle(r) = NaN;
        regionSummary.stdRightKneeAngle(r) = NaN;
        regionSummary.stdAngularSpeed(r) = NaN;
        regionSummary.stdLeftAngularSpeed(r) = NaN;
        regionSummary.stdLegAsymmetry(r) = NaN;
        regionSummary.medianLeftKneeAngle(r) = NaN;
        regionSummary.medianRightKneeAngle(r) = NaN;
        regionSummary.medianAngularSpeed(r) = NaN;
        regionSummary.medianLeftAngularSpeed(r) = NaN;
        regionSummary.medianLegAsymmetry(r) = NaN;
        regionSummary.minLeftKneeAngle(r) = NaN;
        regionSummary.maxLeftKneeAngle(r) = NaN;
        regionSummary.minRightKneeAngle(r) = NaN;
        regionSummary.maxRightKneeAngle(r) = NaN;
        regionSummary.minAngularSpeed(r) = NaN;
        regionSummary.maxAngularSpeed(r) = NaN;
        regionSummary.minLeftAngularSpeed(r) = NaN;
        regionSummary.maxLeftAngularSpeed(r) = NaN;
        regionSummary.minLegAsymmetry(r) = NaN;
        regionSummary.maxLegAsymmetry(r) = NaN;
        regionSummary.p25LeftKneeAngle(r) = NaN;
        regionSummary.p75LeftKneeAngle(r) = NaN;
        regionSummary.p25RightKneeAngle(r) = NaN;
        regionSummary.p75RightKneeAngle(r) = NaN;
        regionSummary.p25AngularSpeed(r) = NaN;
        regionSummary.p75AngularSpeed(r) = NaN;
        regionSummary.p25LeftAngularSpeed(r) = NaN;
        regionSummary.p75LeftAngularSpeed(r) = NaN;
        regionSummary.avgComSpeed(r) = NaN;
        regionSummary.stdComSpeed(r) = NaN;
        regionSummary.medianComSpeed(r) = NaN;
        regionSummary.minComSpeed(r) = NaN;
        regionSummary.maxComSpeed(r) = NaN;
        regionSummary.p25ComSpeed(r) = NaN;
        regionSummary.p75ComSpeed(r) = NaN;
        regionSummary.uniqueMousesVisited(r) = 0;
        regionSummary.totalGroupsVisited(r) = 0;
        regionSummary.dominantGroup{r} = 'None';
        regionSummary.dominantGroupPercentage(r) = 0;
    end
end

fprintf('Region-based analysis complete!\n');

%% Export all results to CSV files
fprintf('\n=== EXPORTING RESULTS TO CSV ===\n');

% 1. Export main mouse results
mouseResultsTable = table();
mouseResultsTable.MouseID = results.mouseID;
mouseResultsTable.MouseLabel = results.mouseLabel;
mouseResultsTable.Group = results.group;
mouseResultsTable.Week = results.week;
mouseResultsTable.CentroidDistance = results.centroidDist;
mouseResultsTable.DensityOverlap = results.densityOverlap;
mouseResultsTable.DensityCorrelation = results.densityCorr;
mouseResultsTable.AvgDisplacement = results.avgDisplacement;
mouseResultsTable.DensityKLDiv = results.densityKLDiv;
mouseResultsTable.DensityBhattacharyya = results.densityBhattacharyya;

% Add summary biomechanical metrics
mouseResultsTable.AvgLeftKneeAngle = zeros(numMice, 1);
mouseResultsTable.AvgRightKneeAngle = zeros(numMice, 1);
mouseResultsTable.AvgAngularSpeed = zeros(numMice, 1);        % Average of both legs
mouseResultsTable.AvgLeftAngularSpeed = zeros(numMice, 1);    % Left side only
mouseResultsTable.AvgLegAsymmetry = zeros(numMice, 1);
mouseResultsTable.AvgComSpeed = zeros(numMice, 1);           % Center of mass speed

for i = 1:numMice
    if ~isempty(results.leftKneeAngle{i})
        mouseResultsTable.AvgLeftKneeAngle(i) = mean(results.leftKneeAngle{i});
        mouseResultsTable.AvgRightKneeAngle(i) = mean(results.rightKneeAngle{i});
        mouseResultsTable.AvgAngularSpeed(i) = mean((results.leftAngularSpeed{i} + results.rightAngularSpeed{i})/2);  % Average of both
        mouseResultsTable.AvgLeftAngularSpeed(i) = mean(results.leftAngularSpeed{i});  % Left side only
        mouseResultsTable.AvgLegAsymmetry(i) = mean(results.legAsymmetry{i});
        if ~isempty(results.comSpeed{i})
            mouseResultsTable.AvgComSpeed(i) = mean(results.comSpeed{i});
        else
            mouseResultsTable.AvgComSpeed(i) = NaN;
        end
    else
        mouseResultsTable.AvgLeftKneeAngle(i) = NaN;
        mouseResultsTable.AvgRightKneeAngle(i) = NaN;
        mouseResultsTable.AvgAngularSpeed(i) = NaN;
        mouseResultsTable.AvgLeftAngularSpeed(i) = NaN;
        mouseResultsTable.AvgLegAsymmetry(i) = NaN;
        mouseResultsTable.AvgComSpeed(i) = NaN;
    end
end

csvFile = fullfile(csvDir, 'mouse_results_summary.csv');
writetable(mouseResultsTable, csvFile);
fprintf('Exported mouse results: %s\n', csvFile);

% 2. Export comprehensive region summary results
regionResultsTable = table();
regionResultsTable.RegionID = regionSummary.regionID;
regionResultsTable.TotalFrames = regionSummary.totalFrames;
regionResultsTable.UniqueMousesVisited = regionSummary.uniqueMousesVisited;
regionResultsTable.TotalGroupsVisited = regionSummary.totalGroupsVisited;
regionResultsTable.DominantGroup = regionSummary.dominantGroup;
regionResultsTable.DominantGroupPercentage = regionSummary.dominantGroupPercentage;

% Mean values
regionResultsTable.AvgLeftKneeAngle = regionSummary.avgLeftKneeAngle;
regionResultsTable.AvgRightKneeAngle = regionSummary.avgRightKneeAngle;
regionResultsTable.AvgAngularSpeed = regionSummary.avgAngularSpeed;           % Average of both legs
regionResultsTable.AvgLeftAngularSpeed = regionSummary.avgLeftAngularSpeed;   % Left side only
regionResultsTable.AvgLegAsymmetry = regionSummary.avgLegAsymmetry;
regionResultsTable.AvgComSpeed = regionSummary.avgComSpeed;                 % Center of mass speed

% Standard deviations
regionResultsTable.StdLeftKneeAngle = regionSummary.stdLeftKneeAngle;
regionResultsTable.StdRightKneeAngle = regionSummary.stdRightKneeAngle;
regionResultsTable.StdAngularSpeed = regionSummary.stdAngularSpeed;           % Average of both legs
regionResultsTable.StdLeftAngularSpeed = regionSummary.stdLeftAngularSpeed;   % Left side only
regionResultsTable.StdLegAsymmetry = regionSummary.stdLegAsymmetry;
regionResultsTable.StdComSpeed = regionSummary.stdComSpeed;                 % Center of mass speed

% Median values
regionResultsTable.MedianLeftKneeAngle = regionSummary.medianLeftKneeAngle;
regionResultsTable.MedianRightKneeAngle = regionSummary.medianRightKneeAngle;
regionResultsTable.MedianAngularSpeed = regionSummary.medianAngularSpeed;
regionResultsTable.MedianLeftAngularSpeed = regionSummary.medianLeftAngularSpeed;
regionResultsTable.MedianLegAsymmetry = regionSummary.medianLegAsymmetry;
regionResultsTable.MedianComSpeed = regionSummary.medianComSpeed;

% Min/Max values
regionResultsTable.MinLeftKneeAngle = regionSummary.minLeftKneeAngle;
regionResultsTable.MaxLeftKneeAngle = regionSummary.maxLeftKneeAngle;
regionResultsTable.MinRightKneeAngle = regionSummary.minRightKneeAngle;
regionResultsTable.MaxRightKneeAngle = regionSummary.maxRightKneeAngle;
regionResultsTable.MinAngularSpeed = regionSummary.minAngularSpeed;
regionResultsTable.MaxAngularSpeed = regionSummary.maxAngularSpeed;
regionResultsTable.MinLeftAngularSpeed = regionSummary.minLeftAngularSpeed;
regionResultsTable.MaxLeftAngularSpeed = regionSummary.maxLeftAngularSpeed;
regionResultsTable.MinLegAsymmetry = regionSummary.minLegAsymmetry;
regionResultsTable.MaxLegAsymmetry = regionSummary.maxLegAsymmetry;
regionResultsTable.MinComSpeed = regionSummary.minComSpeed;
regionResultsTable.MaxComSpeed = regionSummary.maxComSpeed;

% Percentiles
regionResultsTable.P25LeftKneeAngle = regionSummary.p25LeftKneeAngle;
regionResultsTable.P75LeftKneeAngle = regionSummary.p75LeftKneeAngle;
regionResultsTable.P25RightKneeAngle = regionSummary.p25RightKneeAngle;
regionResultsTable.P75RightKneeAngle = regionSummary.p75RightKneeAngle;
regionResultsTable.P25AngularSpeed = regionSummary.p25AngularSpeed;
regionResultsTable.P75AngularSpeed = regionSummary.p75AngularSpeed;
regionResultsTable.P25LeftAngularSpeed = regionSummary.p25LeftAngularSpeed;
regionResultsTable.P75LeftAngularSpeed = regionSummary.p75LeftAngularSpeed;
regionResultsTable.P25ComSpeed = regionSummary.p25ComSpeed;
regionResultsTable.P75ComSpeed = regionSummary.p75ComSpeed;

csvFile = fullfile(csvDir, 'comprehensive_region_statistics.csv');
writetable(regionResultsTable, csvFile);
fprintf('Exported comprehensive region statistics: %s\n', csvFile);

% 3. Export detailed region frequency by group and individual
fprintf('Exporting detailed region frequency data...\n');

% Create group frequency table
allGroups = unique(mouseGroups);
allWeeks = unique(mouseWeeks);
regionGroupTable = table();

regionIDs = [];
weeks = {};
groups = {};
frequencies = [];

for w = 1:length(allWeeks)
    week = allWeeks{w};
    weekGroups = unique(mouseGroups(strcmp(mouseWeeks, week)));
    
    for g = 1:length(weekGroups)
        group = weekGroups{g};
        
        for r = 1:numRegions
            regionID = regionData(r).regionID;
            
            if isKey(regionData(r).groupCounts, group)
                freq = regionData(r).groupCounts(group);
            else
                freq = 0;
            end
            
            regionIDs = [regionIDs; regionID];
            weeks = [weeks; week];
            groups = [groups; group];
            frequencies = [frequencies; freq];
        end
    end
end

regionGroupTable.RegionID = regionIDs;
regionGroupTable.Week = weeks;
regionGroupTable.Group = groups;
regionGroupTable.Frequency = frequencies;

csvFile = fullfile(csvDir, 'region_frequency_by_group.csv');
writetable(regionGroupTable, csvFile);
fprintf('Exported region group frequencies: %s\n', csvFile);

% 4. Export individual mouse region frequencies
fprintf('Exporting individual mouse region frequencies...\n');
regionIndividualTable = table();

regionIDs = [];
mouseIDs = {};
mouseLabels = {};
groups = {};
weeks = {};
frequencies = [];

for i = 1:numMice
    if isempty(results.regionFrequency{i})
        continue;
    end
    
    regionFreq = results.regionFrequency{i};
    visitedRegions = keys(regionFreq);
    
    for j = 1:length(visitedRegions)
        regionID = visitedRegions{j};
        freq = regionFreq(regionID);
        
        regionIDs = [regionIDs; regionID];
        mouseIDs = [mouseIDs; results.mouseID{i}];
        mouseLabels = [mouseLabels; results.mouseLabel{i}];
        groups = [groups; results.group{i}];
        weeks = [weeks; results.week{i}];
        frequencies = [frequencies; freq];
    end
end

regionIndividualTable.RegionID = regionIDs;
regionIndividualTable.MouseID = mouseIDs;
regionIndividualTable.MouseLabel = mouseLabels;
regionIndividualTable.Group = groups;
regionIndividualTable.Week = weeks;
regionIndividualTable.Frequency = frequencies;

csvFile = fullfile(csvDir, 'region_frequency_by_individual.csv');
writetable(regionIndividualTable, csvFile);
fprintf('Exported individual region frequencies: %s\n', csvFile);

% 5. Export dedicated left angular speed analysis
fprintf('Exporting dedicated left angular speed analysis...\n');
leftAngularSpeedTable = table();
leftAngularSpeedTable.MouseID = results.mouseID;
leftAngularSpeedTable.MouseLabel = results.mouseLabel;
leftAngularSpeedTable.Group = results.group;
leftAngularSpeedTable.Week = results.week;
leftAngularSpeedTable.AvgLeftAngularSpeed = mouseResultsTable.AvgLeftAngularSpeed;

csvFile = fullfile(csvDir, 'left_angular_speed_analysis.csv');
writetable(leftAngularSpeedTable, csvFile);
fprintf('Exported left angular speed analysis: %s\n', csvFile);

% 6. Export region-based left angular speed summary
leftRegionTable = table();
leftRegionTable.RegionID = regionSummary.regionID;
leftRegionTable.TotalFrames = regionSummary.totalFrames;
leftRegionTable.AvgLeftAngularSpeed = regionSummary.avgLeftAngularSpeed;
leftRegionTable.StdLeftAngularSpeed = regionSummary.stdLeftAngularSpeed;

csvFile = fullfile(csvDir, 'left_angular_speed_by_region.csv');
writetable(leftRegionTable, csvFile);
fprintf('Exported left angular speed by region: %s\n', csvFile);

% 7. Export dedicated angular speed comprehensive analysis
fprintf('Exporting comprehensive angular speed analysis...\n');
angularSpeedTable = table();
angularSpeedTable.RegionID = regionSummary.regionID;
angularSpeedTable.TotalFrames = regionSummary.totalFrames;
angularSpeedTable.AvgBilateralAngularSpeed = regionSummary.avgAngularSpeed;
angularSpeedTable.StdBilateralAngularSpeed = regionSummary.stdAngularSpeed;
angularSpeedTable.MedianBilateralAngularSpeed = regionSummary.medianAngularSpeed;
angularSpeedTable.MinBilateralAngularSpeed = regionSummary.minAngularSpeed;
angularSpeedTable.MaxBilateralAngularSpeed = regionSummary.maxAngularSpeed;
angularSpeedTable.P25BilateralAngularSpeed = regionSummary.p25AngularSpeed;
angularSpeedTable.P75BilateralAngularSpeed = regionSummary.p75AngularSpeed;
angularSpeedTable.AvgLeftAngularSpeed = regionSummary.avgLeftAngularSpeed;
angularSpeedTable.StdLeftAngularSpeed = regionSummary.stdLeftAngularSpeed;
angularSpeedTable.MedianLeftAngularSpeed = regionSummary.medianLeftAngularSpeed;
angularSpeedTable.MinLeftAngularSpeed = regionSummary.minLeftAngularSpeed;
angularSpeedTable.MaxLeftAngularSpeed = regionSummary.maxLeftAngularSpeed;
angularSpeedTable.P25LeftAngularSpeed = regionSummary.p25LeftAngularSpeed;
angularSpeedTable.P75LeftAngularSpeed = regionSummary.p75LeftAngularSpeed;
angularSpeedTable.AvgComSpeed = regionSummary.avgComSpeed;
angularSpeedTable.StdComSpeed = regionSummary.stdComSpeed;
angularSpeedTable.MedianComSpeed = regionSummary.medianComSpeed;
angularSpeedTable.MinComSpeed = regionSummary.minComSpeed;
angularSpeedTable.MaxComSpeed = regionSummary.maxComSpeed;
angularSpeedTable.P25ComSpeed = regionSummary.p25ComSpeed;
angularSpeedTable.P75ComSpeed = regionSummary.p75ComSpeed;

csvFile = fullfile(csvDir, 'comprehensive_angular_speed_analysis.csv');
writetable(angularSpeedTable, csvFile);
fprintf('Exported comprehensive angular speed analysis: %s\n', csvFile);

% 8. Export joint angle comprehensive analysis
fprintf('Exporting comprehensive joint angle analysis...\n');
jointAngleTable = table();
jointAngleTable.RegionID = regionSummary.regionID;
jointAngleTable.TotalFrames = regionSummary.totalFrames;
jointAngleTable.AvgLeftKneeAngle = regionSummary.avgLeftKneeAngle;
jointAngleTable.StdLeftKneeAngle = regionSummary.stdLeftKneeAngle;
jointAngleTable.MedianLeftKneeAngle = regionSummary.medianLeftKneeAngle;
jointAngleTable.MinLeftKneeAngle = regionSummary.minLeftKneeAngle;
jointAngleTable.MaxLeftKneeAngle = regionSummary.maxLeftKneeAngle;
jointAngleTable.P25LeftKneeAngle = regionSummary.p25LeftKneeAngle;
jointAngleTable.P75LeftKneeAngle = regionSummary.p75LeftKneeAngle;
jointAngleTable.AvgRightKneeAngle = regionSummary.avgRightKneeAngle;
jointAngleTable.StdRightKneeAngle = regionSummary.stdRightKneeAngle;
jointAngleTable.MedianRightKneeAngle = regionSummary.medianRightKneeAngle;
jointAngleTable.MinRightKneeAngle = regionSummary.minRightKneeAngle;
jointAngleTable.MaxRightKneeAngle = regionSummary.maxRightKneeAngle;
jointAngleTable.P25RightKneeAngle = regionSummary.p25RightKneeAngle;
jointAngleTable.P75RightKneeAngle = regionSummary.p75RightKneeAngle;

csvFile = fullfile(csvDir, 'comprehensive_joint_angle_analysis.csv');
writetable(jointAngleTable, csvFile);
fprintf('Exported comprehensive joint angle analysis: %s\n', csvFile);

% 9. Export watershed region frequency comprehensive analysis
fprintf('Exporting comprehensive watershed region frequency analysis...\n');
regionFreqTable = table();
regionFreqTable.RegionID = regionSummary.regionID;
regionFreqTable.TotalFrames = regionSummary.totalFrames;
regionFreqTable.UniqueMousesVisited = regionSummary.uniqueMousesVisited;
regionFreqTable.TotalGroupsVisited = regionSummary.totalGroupsVisited;
regionFreqTable.DominantGroup = regionSummary.dominantGroup;
regionFreqTable.DominantGroupPercentage = regionSummary.dominantGroupPercentage;

% Add usage density (frames per unique mouse)
regionFreqTable.AvgFramesPerMouse = zeros(length(regionSummary.regionID), 1);
for r = 1:length(regionSummary.regionID)
    if regionSummary.uniqueMousesVisited(r) > 0
        regionFreqTable.AvgFramesPerMouse(r) = regionSummary.totalFrames(r) / regionSummary.uniqueMousesVisited(r);
    else
        regionFreqTable.AvgFramesPerMouse(r) = 0;
    end
end

csvFile = fullfile(csvDir, 'comprehensive_watershed_frequency_analysis.csv');
writetable(regionFreqTable, csvFile);
fprintf('Exported comprehensive watershed frequency analysis: %s\n', csvFile);

% 10. Export detailed group breakdown per region
fprintf('Exporting detailed group breakdown per region...\n');
detailedGroupTable = table();
regionIDs = [];
groupNames = {};
framesByGroup = [];
percentageByGroup = [];
uniqueMousesByGroup = [];

for r = 1:numRegions
    regionID = regionData(r).regionID;
    groupKeys = keys(regionData(r).groupCounts);
    totalFramesInRegion = regionData(r).totalFrames;
    
    for g = 1:length(groupKeys)
        group = groupKeys{g};
        frames = regionData(r).groupCounts(group);
        percentage = (frames / totalFramesInRegion) * 100;
        
        % Count unique mice from this group in this region
        individualKeys = keys(regionData(r).individualCounts);
        uniqueMiceFromGroup = 0;
        for m = 1:length(individualKeys)
            mouseLabel = individualKeys{m};
            % Find mouse index to get group
            mouseIdx = find(strcmp(results.mouseLabel, mouseLabel));
            if ~isempty(mouseIdx) && strcmp(results.group{mouseIdx}, group)
                uniqueMiceFromGroup = uniqueMiceFromGroup + 1;
            end
        end
        
        regionIDs = [regionIDs; regionID];
        groupNames = [groupNames; group];
        framesByGroup = [framesByGroup; frames];
        percentageByGroup = [percentageByGroup; percentage];
        uniqueMousesByGroup = [uniqueMousesByGroup; uniqueMiceFromGroup];
    end
end

detailedGroupTable.RegionID = regionIDs;
detailedGroupTable.Group = groupNames;
detailedGroupTable.FrameCount = framesByGroup;
detailedGroupTable.PercentageOfRegion = percentageByGroup;
detailedGroupTable.UniqueMiceFromGroup = uniqueMousesByGroup;

csvFile = fullfile(csvDir, 'detailed_group_breakdown_per_region.csv');
writetable(detailedGroupTable, csvFile);
fprintf('Exported detailed group breakdown per region: %s\n', csvFile);

% 11. Export dedicated COM speed analysis
fprintf('Exporting dedicated COM speed analysis...\n');
comSpeedTable = table();
comSpeedTable.RegionID = regionSummary.regionID;
comSpeedTable.TotalFrames = regionSummary.totalFrames;
comSpeedTable.AvgComSpeed = regionSummary.avgComSpeed;
comSpeedTable.StdComSpeed = regionSummary.stdComSpeed;
comSpeedTable.MedianComSpeed = regionSummary.medianComSpeed;
comSpeedTable.MinComSpeed = regionSummary.minComSpeed;
comSpeedTable.MaxComSpeed = regionSummary.maxComSpeed;
comSpeedTable.P25ComSpeed = regionSummary.p25ComSpeed;
comSpeedTable.P75ComSpeed = regionSummary.p75ComSpeed;
comSpeedTable.UniqueMousesVisited = regionSummary.uniqueMousesVisited;
comSpeedTable.DominantGroup = regionSummary.dominantGroup;

csvFile = fullfile(csvDir, 'comprehensive_com_speed_analysis.csv');
writetable(comSpeedTable, csvFile);
fprintf('Exported comprehensive COM speed analysis: %s\n', csvFile);

% 12. Export per-mouse COM speed summary
mouseComSpeedTable = table();
mouseComSpeedTable.MouseID = results.mouseID;
mouseComSpeedTable.MouseLabel = results.mouseLabel;
mouseComSpeedTable.Group = results.group;
mouseComSpeedTable.Week = results.week;
mouseComSpeedTable.AvgComSpeed = mouseResultsTable.AvgComSpeed;

csvFile = fullfile(csvDir, 'mouse_com_speed_analysis.csv');
writetable(mouseComSpeedTable, csvFile);
fprintf('Exported mouse COM speed analysis: %s\n', csvFile);

fprintf('CSV export complete! Total files exported: 12\n');

%% Save enhanced results
save('flip_analysis_results_weekdata_enhanced.mat', 'results', 'mouseFiles', 'mouseGroups', 'mouseWeeks', ...
    'avgVelocities', 'regionCounts', 'LL', 'LL2', 'groupColors', ...
    'extractFramesForRegion', 'exportAllRegionsToFolders', 'exportAllRegionsWithVideos', 'finalValidRegions', ...
    'regionData', 'regionSummary', 'mouseResultsTable', 'regionResultsTable', ...
    'regionGroupTable', 'regionIndividualTable', 'comSpeedTable', 'mouseComSpeedTable');
fprintf('\nEnhanced results saved to flip_analysis_results_weekdata_enhanced.mat\n');
fprintf('All plots saved to: %s\n', plotsDir);
fprintf('All CSV files saved to: %s\n', csvDir);

%% Helper function for frame extraction
function frames = extractRegionFrames(regionID, numFrames, wrFINE, LL, LL2, zEmbeddings, mouseFiles, allMOUSE, numMice)
    fprintf('\nExtracting %d frames from region %d...\n', numFrames, regionID);
    
    % Find all time points that fall within this region
    regionTimePoints = [];
    mouseSources = [];
    
    for m = 1:numMice
        z_img = (zEmbeddings{m} + 65) * 501 / 130;
        
        for t = 1:size(z_img, 1)
            x = round(z_img(t, 1));
            y = round(z_img(t, 2));
            
            if x >= 1 && x <= 501 && y >= 1 && y <= 501
                if LL2(y, x) == regionID  % This point is in the target region
                    regionTimePoints = [regionTimePoints; t];
                    mouseSources = [mouseSources; m];
                end
            end
        end
    end
    
    fprintf('Found %d time points in region %d\n', length(regionTimePoints), regionID);
    
    if isempty(regionTimePoints)
        fprintf('No data found for region %d\n', regionID);
        frames = [];
        return;
    end
    
    % Extract frames around random time points
    frames = [];
    halfWindow = floor(numFrames / 2);
    
    % Try to get diverse samples
    maxAttempts = min(10, length(regionTimePoints));
    
    for attempt = 1:maxAttempts
        % Random selection
        idx = randi(length(regionTimePoints));
        t = regionTimePoints(idx);
        m = mouseSources(idx);
        
        % Get frame indices
        startFrame = max(1, t - halfWindow);
        endFrame = min(size(allMOUSE{m}, 1), startFrame + numFrames - 1);
        
        if endFrame - startFrame + 1 == numFrames
            % Extract frames
            frameData = allMOUSE{m}(startFrame:endFrame, :, :);
            
            frames = [frames; {struct('mouse', mouseFiles{m}, ...
                                     'startFrame', startFrame, ...
                                     'endFrame', endFrame, ...
                                     'regionID', regionID, ...
                                     'data', frameData)}];
            
            fprintf('Extracted frames %d-%d from %s\n', startFrame, endFrame, mouseFiles{m});
            
            if length(frames) >= 3  % Get up to 3 examples
                break;
            end
        end
    end
    
    fprintf('Successfully extracted %d frame sequences from region %d\n', length(frames), regionID);
end

%% Function to export all regions to folders
function exportAllRegions(finalValidRegions, wrFINE, LL, LL2, zEmbeddings, mouseFiles, allMOUSE, numMice, avgVelocities, regionCounts, saveVideos)
    if nargin < 11
        saveVideos = false;
    end
    
    fprintf('\n=== EXPORTING ALL REGIONS TO FOLDERS ===\n');
    if saveVideos
        fprintf('Video export enabled - will save AVI files\n');
    end
    
    % Create main output directory
    outputDir = 'behavioral_regions_frames';
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end
    
    % Parameters
    numFramesToExtract = 60;
    minFramesRequired = 50;  % Minimum frames in region to export
    
    % Create summary file
    summaryFile = fullfile(outputDir, 'region_summary.txt');
    fid = fopen(summaryFile, 'w');
    fprintf(fid, 'Behavioral Region Frame Export Summary\n');
    fprintf(fid, '=====================================\n\n');
    fprintf(fid, 'Region ID\tAvg Velocity\tTotal Frames\tExported Samples\n');
    
    exportedCount = 0;
    
    for i = 1:length(finalValidRegions)
        regionID = finalValidRegions(i);
        
        if regionCounts(regionID) < minFramesRequired
            continue;  % Skip regions with too few frames
        end
        
        fprintf('Processing region %d (%d/%d)...\n', regionID, i, length(finalValidRegions));
        
        % Create folder for this region
        regionFolder = fullfile(outputDir, sprintf('region_%03d', regionID));
        if ~exist(regionFolder, 'dir')
            mkdir(regionFolder);
        end
        
        % Find all time points in this region
        regionTimePoints = [];
        mouseSources = [];
        
        for m = 1:numMice
            z_img = (zEmbeddings{m} + 65) * 501 / 130;
            
            for t = 1:size(z_img, 1)
                x = round(z_img(t, 1));
                y = round(z_img(t, 2));
                
                if x >= 1 && x <= 501 && y >= 1 && y <= 501
                    if LL2(y, x) == regionID
                        regionTimePoints = [regionTimePoints; t];
                        mouseSources = [mouseSources; m];
                    end
                end
            end
        end
        
        if isempty(regionTimePoints)
            continue;
        end
        
        % Extract up to 5 sample sequences
        halfWindow = floor(numFramesToExtract / 2);
        samplesExtracted = 0;
        maxSamples = 5;
        
        % Try to get diverse samples by spacing them out
        indices = round(linspace(1, length(regionTimePoints), min(maxSamples*3, length(regionTimePoints))));
        
        for idx = indices
            if samplesExtracted >= maxSamples
                break;
            end
            
            t = regionTimePoints(idx);
            m = mouseSources(idx);
            
            % Get frame indices
            startFrame = max(1, t - halfWindow);
            endFrame = min(size(allMOUSE{m}, 1), startFrame + numFramesToExtract - 1);
            
            if endFrame - startFrame + 1 == numFramesToExtract
                % Extract frames - ensure correct format [frames, 3, joints]
                frameData = allMOUSE{m}(startFrame:endFrame, :, :);
                
                % Verify data dimensions
                if size(frameData, 2) ~= 3 || size(frameData, 3) ~= 23
                    fprintf('Warning: Unexpected data dimensions for region %d: [%d, %d, %d]\n', ...
                        regionID, size(frameData, 1), size(frameData, 2), size(frameData, 3));
                end
                
                % Save as MAT file
                [~, baseFileName, ~] = fileparts(mouseFiles{m});
                sampleFile = fullfile(regionFolder, sprintf('sample_%02d_%s_frames_%d-%d.mat', ...
                    samplesExtracted+1, baseFileName, startFrame, endFrame));
                
                save(sampleFile, 'frameData', 'regionID', 'startFrame', 'endFrame', 'mouseFiles');
                
                % Save video if requested
                if saveVideos
                    try
                        videoFile = fullfile(regionFolder, sprintf('sample_%02d_%s_frames_%d-%d.avi', ...
                            samplesExtracted+1, baseFileName, startFrame, endFrame));
                        
                        % Create visualization
                        createVideoFromFrames(frameData, videoFile, regionID);
                        
                    catch ME
                        fprintf('Warning: Could not create video for region %d sample %d: %s\n', ...
                            regionID, samplesExtracted+1, ME.message);
                    end
                end
                
                samplesExtracted = samplesExtracted + 1;
            end
        end
        
        % Write region info file
        infoFile = fullfile(regionFolder, 'region_info.txt');
        fid2 = fopen(infoFile, 'w');
        fprintf(fid2, 'Region %d Information\n', regionID);
        fprintf(fid2, '====================\n');
        fprintf(fid2, 'Average Velocity: %.2f units/s\n', avgVelocities(regionID));
        fprintf(fid2, 'Total Frames in Region: %d\n', length(regionTimePoints));
        fprintf(fid2, 'Samples Extracted: %d\n', samplesExtracted);
        fprintf(fid2, 'Frame Length per Sample: %d\n', numFramesToExtract);
        fclose(fid2);
        
        % Update summary
        fprintf(fid, '%d\t\t%.2f\t\t%d\t\t%d\n', regionID, avgVelocities(regionID), ...
            length(regionTimePoints), samplesExtracted);
        
        if samplesExtracted > 0
            exportedCount = exportedCount + 1;
        end
    end
    
    fclose(fid);
    
    fprintf('\n=== EXPORT COMPLETE ===\n');
    fprintf('Exported data for %d regions to folder: %s\n', exportedCount, outputDir);
    fprintf('See region_summary.txt for details.\n');
end

%% Function to create video visualization from frame data
function createVideoFromFrames(frameData, videoFile, regionID)
    % Create proper skeleton structure (like original SocialMapper)
    joints_idx = [
        1 2;   % Snout to EarL
        1 3;   % Snout to EarR
        2 3;   % EarL to EarR
        1 4;   % Snout to SpineF (for visualization)
        4 5;   % SpineF to SpineM
        5 6;   % SpineM to SpineL
        6 7;   % SpineL to TailBase
        4 8;   % SpineF to ShoulderL
        8 9;   % ShoulderL to ElbowL
        9 10;  % ElbowL to WristL
        10 11; % WristL to HandL
        4 12;  % SpineF to ShoulderR
        12 13; % ShoulderR to ElbowR
        13 14; % ElbowR to WristR
        14 15; % WristR to HandR
        6 16;  % SpineL to HipL
        16 17; % HipL to KneeL
        17 18; % KneeL to AnkleL
        18 19; % AnkleL to FootL
        6 20;  % SpineL to HipR
        20 21; % HipR to KneeR
        21 22; % KneeR to AnkleR
        22 23; % AnkleR to FootR
    ];
    
    % Define colors for different body parts
    chead = [1 .6 .2];         % orange for head
    cspine = [.2 .635 .172];   % green for spine
    cLF = [0 0 1];             % blue for left front limb
    cRF = [1 0 0];             % red for right front limb
    cLH = [0 1 1];             % cyan for left hind limb
    cRH = [1 0 1];             % magenta for right hind limb
    
    % Color mapping for each joint connection
    scM = [chead; chead; chead; cspine; cspine; cspine; cspine; ...
           cLF; cLF; cLF; cLF; ...
           cRF; cRF; cRF; cRF; ...
           cLH; cLH; cLH; cLH; ...
           cRH; cRH; cRH; cRH];
    
    % Create skeleton struct with required fields
    skeleton.color = scM;
    skeleton.joints_idx = joints_idx;
    skeleton.mcolor = [0 0 0];  % marker color (black)
    
    % Check if Keypoint3DAnimator is available
    if ~exist('Keypoint3DAnimator', 'class')
        % Try simple 2D visualization instead with skeleton connections
        createSimple2DVideo(frameData, videoFile, regionID, joints_idx);
        return;
    end
    
    % Create figure
    close all;
    fig = figure('Name', sprintf('Region %d Behavior', regionID), ...
        'Position', [100 100 800 600], 'Visible', 'off');
    
    % Create animator with proper skeleton struct
    h = Keypoint3DAnimator(frameData, skeleton, 'Position', [0 0 1 1], ...
        'MarkerSize', 30, 'LineWidth', 2);
    
    % Set view and properties
    view(h.getAxes, 20, 30);
    axis equal;
    set(gca, 'Color', 'w');
    set(gcf, 'Color', 'white');
    
    % Add title
    title(sprintf('Region %d Behavioral Sequence', regionID));
    
    % Write video
    numFrames = size(frameData, 1);
    frames = 1:numFrames;
    
    try
        h.writeVideo(frames, videoFile, 'FPS', 25, 'Quality', 70);
        fprintf('  Saved video: %s\n', videoFile);
    catch ME
        fprintf('  Failed to write video: %s\n', ME.message);
        % Try fallback 2D visualization
        createSimple2DVideo(frameData, videoFile, regionID, joints_idx);
    end
    
    close(fig);
end

%% Simple 2D video creation fallback
function createSimple2DVideo(frameData, videoFile, regionID, joints_idx)
    % Create video writer
    writerObj = VideoWriter(videoFile);
    writerObj.FrameRate = 25;
    writerObj.Quality = 70;
    open(writerObj);
    
    fig = figure('Visible', 'off', 'Position', [100 100 800 600]);
    
    numFrames = size(frameData, 1);
    
    for f = 1:numFrames
        clf;
        
        % Extract frame
        pose = squeeze(frameData(f, :, :));
        
        % Plot in 2D (X-Y view)
        subplot(1,2,1);
        hold on;
        
        % Plot skeleton
        for i = 1:size(joints_idx, 1)
            j1 = joints_idx(i, 1);
            j2 = joints_idx(i, 2);
            plot([pose(1, j1), pose(1, j2)], ...
                 [pose(2, j1), pose(2, j2)], 'b-', 'LineWidth', 2);
        end
        
        % Plot joints
        scatter(pose(1, :), pose(2, :), 50, 'r', 'filled');
        
        axis equal;
        grid on;
        xlabel('X');
        ylabel('Y');
        title(sprintf('Region %d - XY View (Frame %d/%d)', regionID, f, numFrames));
        
        % Plot in 2D (X-Z view)
        subplot(1,2,2);
        hold on;
        
        % Plot skeleton
        for i = 1:size(joints_idx, 1)
            j1 = joints_idx(i, 1);
            j2 = joints_idx(i, 2);
            plot([pose(1, j1), pose(1, j2)], ...
                 [pose(3, j1), pose(3, j2)], 'b-', 'LineWidth', 2);
        end
        
        % Plot joints
        scatter(pose(1, :), pose(3, :), 50, 'r', 'filled');
        
        axis equal;
        grid on;
        xlabel('X');
        ylabel('Z');
        title(sprintf('Region %d - XZ View (Frame %d/%d)', regionID, f, numFrames));
        
        % Capture frame
        frame = getframe(fig);
        writeVideo(writerObj, frame);
    end
    
    close(writerObj);
    close(fig);
    
    fprintf('  Saved 2D video: %s\n', videoFile);
end

%% Helper function for angle calculation
function angle = calculateAngle(v1, v2)
    % Calculate angle between two 3D vectors in degrees
    % Handle zero vectors
    if norm(v1) == 0 || norm(v2) == 0
        angle = 0;
        return;
    end
    
    % Normalize vectors
    v1_norm = v1 / norm(v1);
    v2_norm = v2 / norm(v2);
    
    % Calculate angle using dot product
    cosAngle = dot(v1_norm, v2_norm);
    
    % Clamp to valid range to avoid numerical errors
    cosAngle = max(-1, min(1, cosAngle));
    
    % Convert to degrees
    angle = acos(cosAngle) * 180 / pi;
end

%% Helper functions for statistical testing

function pValue = permutationTest(group1, group2, nPerms)
    % Simple permutation test for two groups
    if nargin < 3
        nPerms = 1000;
    end
    
    % Observed difference
    obsDiff = abs(mean(group1) - mean(group2));
    
    % Combine groups
    combined = [group1(:); group2(:)];
    n1 = length(group1);
    n2 = length(group2);
    
    % Permutation test
    permDiffs = zeros(nPerms, 1);
    for i = 1:nPerms
        % Random permutation
        permIdx = randperm(length(combined));
        perm1 = combined(permIdx(1:n1));
        perm2 = combined(permIdx(n1+1:end));
        
        permDiffs(i) = abs(mean(perm1) - mean(perm2));
    end
    
    % Calculate p-value
    pValue = sum(permDiffs >= obsDiff) / nPerms;
end

 