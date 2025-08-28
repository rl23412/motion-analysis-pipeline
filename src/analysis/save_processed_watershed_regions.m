%% Save the processed watershed regions from analyze_saved_maps_and_counts.m
% Run this ONCE after running analyze_saved_maps_and_counts.m to save the exact regions

clear; close all; clc;

fprintf('This script saves the processed watershed regions for consistent numbering.\n');
fprintf('Make sure you have already run analyze_saved_maps_and_counts.m!\n\n');

% Check if the analyze script has been run by looking for its outputs
if ~exist(fullfile('analysis_outputs', 'csv', 'per_file_region_counts.csv'), 'file')
    error('Please run analyze_saved_maps_and_counts.m first!');
end

% Run the EXACT same processing as analyze_saved_maps_and_counts.m
fprintf('Running identical processing to analyze_saved_maps_and_counts.m...\n');

% --- User-adjustable options (MUST BE IDENTICAL to analyze script) ---
opts = struct();
opts.displayRangeExpand = 1;
opts.cropPadPx = 20;
opts.filterRegionsByDensity = false;
opts.regionDensityPercentile = 5;
opts.regionMinDensity = 1e-8;
opts.resegmentEnable = true;
opts.resegmentSigma = 0;
opts.resegmentGamma = 1.7;
opts.resegmentMinDensity = 5e-6;
opts.resegmentConnectivity = 4;
opts.resegmentFillHoles = true;
opts.resegmentMinRegionSize = 10;
opts.forceAllBoundaries = true;

% Load saved results
mapFile = 'watershed_SNI_TBI.mat';
load(mapFile, 'D','LL','LL2','llbwb','xx');

% Re-create watershed segmentation EXACTLY as in analyze script
if opts.resegmentEnable
    Dw = D;
    if opts.resegmentGamma ~= 1
        Dw = Dw .^ opts.resegmentGamma;
    end
    
    LL = watershed(-Dw, opts.resegmentConnectivity);
    LL2 = LL;
    
    backgroundMask = Dw < opts.resegmentMinDensity;
    LL2(backgroundMask) = -1;
    
    if opts.resegmentFillHoles
        uniqueRegions = unique(LL2(LL2>0));
        for i = 1:numel(uniqueRegions)
            regionId = uniqueRegions(i);
            regionMask = (LL2 == regionId);
            if exist('imfill','file')
                filledMask = imfill(regionMask, 'holes');
            else
                se = ones(3,3);
                filledMask = imdilate(imerode(regionMask, se), se);
            end
            fillPixels = filledMask & ~regionMask & (LL2 <= 0);
            LL2(fillPixels) = regionId;
        end
    end
    
    if opts.resegmentMinRegionSize > 0
        uniqueRegions = unique(LL2(LL2>0));
        for i = 1:numel(uniqueRegions)
            regionId = uniqueRegions(i);
            regionMask = (LL2 == regionId);
            if nnz(regionMask) < opts.resegmentMinRegionSize
                LL2(regionMask) = -1;
            end
        end
    end
    
    % Recompute boundaries
    if opts.forceAllBoundaries
        LLBW = false(size(LL2));
        uniqueRegions = unique(LL2(LL2>0));
        for i = 1:numel(uniqueRegions)
            regionId = uniqueRegions(i);
            regionMask = (LL2 == regionId);
            regionBoundary = regionMask & ~imerode(regionMask, ones(3,3));
            LLBW = LLBW | regionBoundary;
        end
        B = bwboundaries(LLBW);
        C = B;
    else
        LLBW = (LL2 == 0);
        B = bwboundaries(LLBW);
        if numel(B) >= 2
            C = B(2:end);
        else
            C = B;
        end
    end
    llbwb = combineCellsLocal(C);
end

% Build valid region set (CRITICAL - this determines the numbering!)
validRegionIds = unique(LL2(LL2>0));
validRegionIds = validRegionIds(:)';
numRegions = numel(validRegionIds);

fprintf('\nProcessed watershed has %d regions\n', numRegions);
fprintf('Valid region IDs: %s\n', mat2str(validRegionIds));

% Save the processed data
processedDataFile = 'processed_watershed_for_analysis.mat';
save(processedDataFile, 'D', 'Dw', 'LL2', 'llbwb', 'validRegionIds', 'numRegions', 'opts');
fprintf('\nSaved processed watershed data to: %s\n', processedDataFile);

% Create visualization of region numbering
figure('Name','Region Numbering Reference','Position',[100 100 900 900]);
imagesc(D); axis equal off; colormap(flipud(gray)); caxis([0 max(D(:))*0.8]); hold on;
scatter(llbwb(:,2), llbwb(:,1), 1, 'k', '.');

% Add region numbers (consecutive 1-24)
for k = 1:numRegions
    oldId = validRegionIds(k);
    [yy,xxi] = find(LL2==oldId);
    if isempty(yy), continue; end
    cx = round(mean(xxi)); cy = round(mean(yy));
    text(cx, cy, sprintf('%d', k), 'Color','y','FontSize',10,'FontWeight','bold','HorizontalAlignment','center');
end
title(sprintf('Region Numbering (1-%d) Used in Frequency Analysis', numRegions));

saveas(gcf, fullfile('analysis_outputs', 'figures', 'region_numbering_reference.png'));
fprintf('Saved region numbering reference figure\n');

fprintf('\n=== IMPORTANT ===\n');
fprintf('Use the saved processed_watershed_for_analysis.mat file for all future visualizations\n');
fprintf('to ensure consistent region numbering!\n');

function out = combineCellsLocal(c)
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
