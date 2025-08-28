%% Generate Group Difference Heatmap - MATLAB Version
% Uses IDENTICAL code from analyze_saved_maps_and_counts.m
% Reads frequency differences from Python analysis and plots on MATLAB map

clear; close all; clc;

%% Load frequency difference data from Python analysis
csvDir = fullfile('analysis_outputs', 'group_difference_heatmaps');
diffData = readtable(fullfile(csvDir, 'region_frequency_differences.csv'));

% Extract differences for each region (1-24)
regionDifferences = zeros(24, 1);
for i = 1:height(diffData)
    regionIdx = diffData.Region(i);
    regionDifferences(regionIdx) = diffData.Difference_Other_minus_TBI(i);
end

%% COPY-PASTE FROM analyze_saved_maps_and_counts.m - IDENTICAL CODE
outDir = 'analysis_outputs';
figDir = fullfile(outDir, 'figures');
if ~exist(figDir,'dir'), mkdir(figDir); end

% --- User-adjustable options (affect only analysis display/segmentation) ---
opts = struct();
% Expand or shrink the displayed coordinate range used to map z->image.
% 1.00 = original range from the saved grid (xx). >1 adds margins; <1 tightens.
opts.displayRangeExpand = 1;
% Cropped figure padding (pixels) around LL2>0 support
opts.cropPadPx = 20;
% Filter watershed regions by density (exclude low-density regions entirely)
opts.filterRegionsByDensity = false;   % DISABLED - was filtering out all regions
opts.regionDensityPercentile = 5;      % percentile threshold for region average density (LOWERED)
opts.regionMinDensity = 1e-8;          % absolute minimum average density for regions (LOWERED)
% Cropping options removed - not needed
% Optional re-segmentation of watershed boundaries from the saved density map D
opts.resegmentEnable = true;      % set true to re-run watershed on D here
opts.resegmentSigma = 0;        % Gaussian blur sigma on D before watershed (0 = none)
opts.resegmentGamma = 1.7;        % Nonlinear boosting: use D.^gamma (>=1 to sharpen peaks, try 1.5-3.0)
opts.resegmentMinDensity = 5e-6;  % Mask very low densities as outside (slightly higher threshold)
opts.resegmentConnectivity = 4;   % Watershed connectivity (8=4-connected, 18=8-connected; lower = more regions)
opts.resegmentFillHoles = true;   % Fil holes in regions after watershed
opts.resegmentMinRegionSize = 10; % Remove regions smaller than this many pixels (LOWERED)
opts.forceAllBoundaries = true;   % Force boundaries around all regions, including edge-touching ones

% Check if processed watershed file exists
processedFile = 'processed_watershed_for_analysis.mat';
if exist(processedFile, 'file')
    fprintf('Loading pre-processed watershed data for consistent region numbering...\n');
    load(processedFile);
    fprintf('Loaded %d regions with consistent numbering from analysis\n', numRegions);
else
    % Load saved results and process
    mapFile = 'watershed_SNI_TBI.mat';   % contains D, LL, LL2, llbwb, xx
    load(mapFile, 'D','LL','LL2','llbwb','xx');
    
    % Re-create watershed segmentation from D with different settings
    if opts.resegmentEnable
    fprintf('\n=== Re-segmentation (IDENTICAL to analyze script) ===\n');
    fprintf('Original D range: [%.6f, %.6f]\n', min(D(:)), max(D(:)));
    
    Dw = D;
    if opts.resegmentSigma > 0
        fprintf('Applying Gaussian blur with sigma=%.2f\n', opts.resegmentSigma);
        if exist('imgaussfilt','file')
            Dw = imgaussfilt(Dw, opts.resegmentSigma);
        else
            ksz = max(1, ceil(6*opts.resegmentSigma));
            K = ones(ksz, ksz) / max(1, (ksz*ksz));
            Dw = conv2(Dw, K, 'same');
        end
    end
    if opts.resegmentGamma ~= 1
        fprintf('Applying gamma correction: %.2f\n', opts.resegmentGamma);
        Dw = Dw .^ opts.resegmentGamma;
        fprintf('After gamma D range: [%.6f, %.6f]\n', min(Dw(:)), max(Dw(:)));
    end
    
    fprintf('Running watershed with connectivity=%d\n', opts.resegmentConnectivity);
    LL = watershed(-Dw, opts.resegmentConnectivity);
    LL2 = LL;
    
    % Mask low densities as background
    backgroundMask = Dw < opts.resegmentMinDensity;
    LL2(backgroundMask) = -1;
    fprintf('Background pixels (D<%.2e): %d (%.1f%%)\n', opts.resegmentMinDensity, nnz(backgroundMask), 100*nnz(backgroundMask)/numel(Dw));
    
    % Fill holes in regions if enabled
    if opts.resegmentFillHoles
        fprintf('Filling holes in regions\n');
        uniqueRegions = unique(LL2(LL2>0));
        for i = 1:numel(uniqueRegions)
            regionId = uniqueRegions(i);
            regionMask = (LL2 == regionId);
            if exist('imfill','file')
                filledMask = imfill(regionMask, 'holes');
            else
                % Simple morphological closing as fallback
                se = ones(3,3);
                filledMask = imdilate(imerode(regionMask, se), se);
            end
            % Only fill pixels that were background (LL2==-1 or LL2==0), not other regions
            fillPixels = filledMask & ~regionMask & (LL2 <= 0);
            LL2(fillPixels) = regionId;
        end
    end
    
    % Remove small regions if enabled
    if opts.resegmentMinRegionSize > 0
        fprintf('Removing regions smaller than %d pixels\n', opts.resegmentMinRegionSize);
        uniqueRegions = unique(LL2(LL2>0));
        for i = 1:numel(uniqueRegions)
            regionId = uniqueRegions(i);
            regionMask = (LL2 == regionId);
            if nnz(regionMask) < opts.resegmentMinRegionSize
                LL2(regionMask) = -1;  % Set to background
            end
        end
    end
    
    % Recompute boundaries
    if opts.forceAllBoundaries
        % Create boundaries around ALL regions, including edge-touching ones
        LLBW = false(size(LL2));
        uniqueRegions = unique(LL2(LL2>0));
        for i = 1:numel(uniqueRegions)
            regionId = uniqueRegions(i);
            regionMask = (LL2 == regionId);
            % Get boundary of this region using edge detection
            regionBoundary = regionMask & ~imerode(regionMask, ones(3,3));
            LLBW = LLBW | regionBoundary;
        end
        B = bwboundaries(LLBW);
        C = B; % Use all boundaries including outer ones
    else
        % Original method - only internal boundaries
        LLBW = (LL2 == 0);
        B = bwboundaries(LLBW);
        if numel(B) >= 2
            C = B(2:end);
        else
            C = B;
        end
    end
    llbwb = combineCellsLocal(C);
    
    % Debug boundary creation
    finalRegions = numel(unique(LL2(LL2>0)));
    fprintf('Final regions after processing: %d\n', finalRegions);
        fprintf('=== END Re-segmentation ===\n\n');
    end
    
    % Build valid region set from filtered LL2 (>0 only) and create consecutive numbering
    validRegionIds = unique(LL2(LL2>0));
    validRegionIds = validRegionIds(:)';
    numRegions = numel(validRegionIds);
    fprintf('Number of regions for plotting: %d\n', numRegions);
end

%% Create difference map
differenceMap = zeros(size(D));

% Map consecutive region IDs to difference values
for k = 1:numRegions
    oldId = validRegionIds(k);
    regionMask = (LL2 == oldId);
    if k <= length(regionDifferences)
        differenceMap(regionMask) = regionDifferences(k);
    end
end

%% Plot the difference heatmap - IDENTICAL style to analyze script
figure('Name','Group Difference Heatmap (Other - TBI)','Position',[100 100 1800 900]);

% Subplot 1: Base behavioral map (IDENTICAL to analyze script)
subplot(1,3,1);
imagesc(D); axis equal off; colormap(flipud(gray)); caxis([0 max(D(:))*0.8]); hold on;
scatter(llbwb(:,2), llbwb(:,1), 1, 'k', '.');
for k = 1:numRegions
    oldId = validRegionIds(k);
    [yy,xxi] = find(LL2==oldId);
    if isempty(yy), continue; end
    cx = round(mean(xxi)); cy = round(mean(yy));
    text(cx, cy, sprintf('%d', k), 'Color','y','FontSize',8,'FontWeight','bold','HorizontalAlignment','center');
end
title('Behavioral Map with Region Indices');

% Subplot 2: Difference heatmap
subplot(1,3,2);
% Mask background
maskedDiff = differenceMap;
maskedDiff(LL2 <= 0) = NaN;

% Create custom colormap (blue for Other>TBI, red for TBI>Other)
n = 256;
blueToRed = [linspace(0,1,n/2)', linspace(0,1,n/2)', ones(n/2,1); ...
             ones(n/2,1), linspace(1,0,n/2)', linspace(1,0,n/2)'];
             
imagesc(maskedDiff); axis equal off;
colormap(gca, blueToRed);
maxDiff = max(abs(differenceMap(:)));
caxis([-maxDiff maxDiff]);
colorbar;
hold on;
scatter(llbwb(:,2), llbwb(:,1), 1, 'k', '.');
title('Frequency Difference (Other - TBI)');

% Subplot 3: Overlay
subplot(1,3,3);
imagesc(D); axis equal off; colormap(flipud(gray)); caxis([0 max(D(:))*0.8]); hold on;

% Create transparent overlay
h = imagesc(maskedDiff);
set(h, 'AlphaData', ~isnan(maskedDiff) * 0.7);
colormap(gca, blueToRed);
caxis([-maxDiff maxDiff]);
colorbar;
scatter(llbwb(:,2), llbwb(:,1), 1, 'k', '.');
title('Difference Overlay on Behavioral Map');

% Save figure
saveas(gcf, fullfile(figDir, 'group_difference_heatmap_matlab.png'));
fprintf('\nSaved: %s\n', fullfile(figDir, 'group_difference_heatmap_matlab.png'));

% IMPORTANT: Save the processed watershed data for consistent region numbering
processedDataFile = 'processed_watershed_for_heatmaps.mat';
save(processedDataFile, 'D', 'Dw', 'LL2', 'llbwb', 'validRegionIds', 'numRegions');
fprintf('\nSaved processed watershed data to: %s\n', processedDataFile);
fprintf('This ensures consistent region numbering across all analyses!\n');

%% Also create a simple difference plot with just the overlay
figure('Name','Group Difference Overlay','Position',[100 100 900 900]);
imagesc(D); axis equal off; colormap(flipud(gray)); caxis([0 max(D(:))*0.8]); hold on;

% Create transparent overlay
h = imagesc(maskedDiff);
set(h, 'AlphaData', ~isnan(maskedDiff) * 0.8);

% Better colormap for differences
redblue = [linspace(1,0,128)' linspace(0,0,128)' linspace(0,1,128)'; ...
           linspace(0,1,128)' linspace(0,0,128)' linspace(1,0,128)'];
colormap(gca, redblue);
caxis([-maxDiff maxDiff]);
c = colorbar;
c.Label.String = 'Frequency Difference (Other - TBI)';

scatter(llbwb(:,2), llbwb(:,1), 1, 'k', '.');

% Add region numbers
for k = 1:numRegions
    oldId = validRegionIds(k);
    [yy,xxi] = find(LL2==oldId);
    if isempty(yy), continue; end
    cx = round(mean(xxi)); cy = round(mean(yy));
    if k <= length(regionDifferences) && abs(regionDifferences(k)) > 0.05
        % Only label regions with large differences
        text(cx, cy, sprintf('%d', k), 'Color','y','FontSize',10,'FontWeight','bold','HorizontalAlignment','center');
    end
end

title('Group Frequency Differences: Other Groups vs TBI', 'FontSize', 14);
saveas(gcf, fullfile(figDir, 'group_difference_overlay_matlab.png'));
fprintf('Saved: %s\n', fullfile(figDir, 'group_difference_overlay_matlab.png'));

%% Helper function (COPY from analyze script)
function out = combineCellsLocal(c)
% Combine a cell array of Mx2 boundary arrays into one Nx2 array
    if isempty(c)
        out = [];
        return;
    end
    n = 0;
    for i = 1:numel(c)
        n = n + size(c{i},1);
    end
    out = zeros(n, 2);
    k = 0;
    for i = 1:numel(c)
        ci = c{i};
        out(k+(1:size(ci,1)),:) = ci;
        k = k + size(ci,1);
    end
end
