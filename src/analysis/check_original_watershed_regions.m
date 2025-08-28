%% Check what watershed regions were used in the original analysis
% This script loads the complete embedding results to see which LL2 was used

clear; close all; clc;

% Load the complete embedding results that were used for frequency counting
resFile = 'complete_embedding_results_SNI_TBI.mat';
if exist(resFile, 'file')
    fprintf('Loading original embedding results...\n');
    S = load(resFile);
    if isfield(S, 'results') && isfield(S.results, 'watershed_regions_used')
        fprintf('Found watershed regions info in results\n');
    else
        fprintf('No explicit watershed region info in results file\n');
    end
end

% Check if there's a saved processed watershed file
processedFile = 'processed_watershed_for_heatmaps.mat';
if exist(processedFile, 'file')
    fprintf('\nFound processed watershed file!\n');
    load(processedFile);
    fprintf('Number of regions: %d\n', numRegions);
    fprintf('Valid region IDs: %s\n', mat2str(validRegionIds));
else
    fprintf('\nNo processed watershed file found.\n');
    fprintf('You need to run analyze_saved_maps_and_counts.m first to generate the regions.\n');
end

% Load the original watershed file
mapFile = 'watershed_SNI_TBI.mat';
if exist(mapFile, 'file')
    fprintf('\nLoading original watershed file...\n');
    original = load(mapFile);
    uniqueOriginal = unique(original.LL2(original.LL2 > 0));
    fprintf('Original LL2 has %d regions\n', length(uniqueOriginal));
    fprintf('Original region IDs: %s\n', mat2str(uniqueOriginal));
end

% Check the region counts CSV to see what regions were actually used
csvFile = fullfile('analysis_outputs', 'csv', 'per_file_region_counts.csv');
if exist(csvFile, 'file')
    fprintf('\nChecking region counts CSV...\n');
    T = readtable(csvFile);
    regionCols = T.Properties.VariableNames(contains(T.Properties.VariableNames, 'Region_'));
    fprintf('CSV has %d region columns: %s ... %s\n', length(regionCols), regionCols{1}, regionCols{end});
end

fprintf('\n=== IMPORTANT ===\n');
fprintf('The region numbering must match between:\n');
fprintf('1. The watershed segmentation (LL2)\n');
fprintf('2. The frequency counts in the CSV files\n');
fprintf('3. The visualization\n');
fprintf('\nRun analyze_saved_maps_and_counts.m ONCE to generate consistent regions,\n');
fprintf('then use the saved processed_watershed_for_heatmaps.mat for all visualizations.\n');
