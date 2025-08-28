%% Analyze saved embedding maps and region counts (independent script)
% Loads outputs from custom_embedding_pipeline_SNI_TBI.m
% - Plots behavioral density map with watershed boundaries and region indices
% - Plots each video's re-embedded points over the map and saves separate PNGs
% - Computes per-file frame counts per watershed region and saves a CSV
%
clear; close all; clc;

outDir = 'analysis_outputs';
figDir = fullfile(outDir, 'figures');
perVideoDir = fullfile(figDir, 'per_video');
csvDir = fullfile(outDir, 'csv');
if ~exist(outDir,'dir'), mkdir(outDir); end
if ~exist(figDir,'dir'), mkdir(figDir); end
if ~exist(perVideoDir,'dir'), mkdir(perVideoDir); end
if ~exist(csvDir,'dir'), mkdir(csvDir); end

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

% Load saved results
mapFile = 'watershed_SNI_TBI.mat';   % contains D, LL, LL2, llbwb, xx
resFile = 'complete_embedding_results_SNI_TBI.mat'; % contains results struct

load(mapFile, 'D','LL','LL2','llbwb','xx');
S = load(resFile);
results = S.results;

% Optionally re-create watershed segmentation from D with different settings
if opts.resegmentEnable
    fprintf('\n=== DEBUG: Re-segmentation ===\n');
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
    fprintf('Boundary components found: %d\n', numel(B));
    fprintf('Boundary points total: %d\n', size(llbwb,1));
    
    % Check which regions have boundaries
    uniqueRegions = unique(LL2(LL2>0));
    fprintf('Checking boundary coverage for each region:\n');
    for i = 1:numel(uniqueRegions)
        regionId = uniqueRegions(i);
        regionMask = (LL2 == regionId);
        
        % Check if region touches image edges
        touchesEdge = any(regionMask(1,:)) || any(regionMask(end,:)) || any(regionMask(:,1)) || any(regionMask(:,end));
        
        % Check if region has internal boundaries (adjacent to other regions or background)
        se = ones(3,3); se(2,2) = 0; % 8-connected neighborhood excluding center
        dilatedMask = imdilate(regionMask, se);
        boundaryMask = dilatedMask & ~regionMask & (LL2 == 0);
        hasBoundary = any(boundaryMask(:));
        
        fprintf('  Region %d: touches_edge=%s, has_boundary=%s, size=%d pixels\n', ...
            i, mat2str(touchesEdge), mat2str(hasBoundary), nnz(regionMask));
    end
    fprintf('=== END DEBUG ===\n\n');
end

% Filter watershed regions by average density if enabled
if opts.filterRegionsByDensity
    fprintf('\n=== DEBUG: Region Density Filtering ===\n');
    allRegionIds = unique(LL2(LL2>0));
    allRegionIds = allRegionIds(:)';
    fprintf('Original regions: %d\n', numel(allRegionIds));
    
    % Calculate average density for each region
    regionAvgDensity = zeros(size(allRegionIds));
    for i = 1:numel(allRegionIds)
        regionId = allRegionIds(i);
        regionMask = (LL2 == regionId);
        regionDensities = D(regionMask);
        regionAvgDensity(i) = mean(regionDensities);
    end
    
    fprintf('Region density range: [%.6f, %.6f]\n', min(regionAvgDensity), max(regionAvgDensity));
    
    % Calculate threshold
    densityThreshold = prctile(regionAvgDensity, opts.regionDensityPercentile);
    densityThreshold = max(opts.regionMinDensity, densityThreshold);
    fprintf('Density threshold (%.1f percentile): %.6f\n', opts.regionDensityPercentile, densityThreshold);
    
    % Filter regions
    validRegionMask = regionAvgDensity >= densityThreshold;
    validRegionIds = allRegionIds(validRegionMask);
    
    fprintf('Regions kept: %d/%d (%.1f%%)\n', numel(validRegionIds), numel(allRegionIds), 100*numel(validRegionIds)/numel(allRegionIds));
    
    % Create filtered LL2 by setting excluded regions to -1 (background)
    LL2_filtered = LL2;
    excludedRegions = allRegionIds(~validRegionMask);
    for i = 1:numel(excludedRegions)
        LL2_filtered(LL2 == excludedRegions(i)) = -1;
    end
    
    % Update LL2 to use filtered version
    LL2 = LL2_filtered;
    
    % Recompute boundaries from filtered regions
    LLBW = (LL2 == 0);
    B = bwboundaries(LLBW);
    if numel(B) >= 2
        C = B(2:end);
    else
        C = B;
    end
    llbwb = combineCellsLocal(C);
    fprintf('New boundary points: %d\n', size(llbwb,1));
    fprintf('=== END DEBUG ===\n\n');
else
    validRegionIds = unique(LL2(LL2>0));
end

% Build valid region set from filtered LL2 (>0 only) and create consecutive numbering
validRegionIds = validRegionIds(:)';
numRegions = numel(validRegionIds);

% Cropping code removed - not needed

% Plot main behavioral density map with region indices (masked by LL2>0)
fig = figure('Name','Behavioral Map with Region Indices','Position',[100 100 900 900]);
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
saveas(fig, fullfile(figDir, 'behavioral_map_with_indices.png'));
close(fig);

% Cropped figure removed - not needed

% Prepare per-file region counts
fileNames = results.reembedding_labels_all;
zAll = results.zEmbeddings_all;   % cell array, each N_t x 2
numFiles = numel(fileNames);

% Helper to convert z (in t-SNE coords) to image indices
mapBounds = [min(xx) max(xx)];
% Expand or shrink display range symmetrically around center if requested
if opts.displayRangeExpand ~= 1
    ctr = mean(mapBounds);
    halfR = (mapBounds(2)-mapBounds(1))/2;
    halfR = halfR * opts.displayRangeExpand;
    mapBounds = [ctr - halfR, ctr + halfR];
end
gridSize = size(D,1);

counts = zeros(numFiles, numRegions);

% Initialize frame indices storage for each video and region
frameIndices = cell(numFiles, numRegions);
for i = 1:numFiles
    for j = 1:numRegions
        frameIndices{i,j} = [];
    end
end

% Per-video plotting and counting
for i = 1:numFiles
    z = zAll{i};
    if isempty(z) || size(z,2)~=2
        continue;
    end
    % Map z to image coordinates (1..gridSize)
    xImg = round( (z(:,1) - mapBounds(1)) / (mapBounds(2)-mapBounds(1)) * (gridSize-1) + 1 );
    yImg = round( (z(:,2) - mapBounds(1)) / (mapBounds(2)-mapBounds(1)) * (gridSize-1) + 1 );
    valid = xImg>=1 & xImg<=gridSize & yImg>=1 & yImg<=gridSize;
    xImg = xImg(valid); yImg = yImg(valid);

    % Count per region via LL2 lookup (exclude outside/background) and map to consecutive ids
    % Also track frame indices for each region
    linIdx = sub2ind(size(LL2), yImg, xImg);
    regIdsOld = double(LL2(linIdx));
    
    % Keep track of original frame indices (before filtering invalid ones)
    validFrameIndices = find(valid);
    
    % Filter out background/invalid regions
    validRegionMask = regIdsOld >= 1;
    regIdsOld = regIdsOld(validRegionMask);
    frameIndicesForRegions = validFrameIndices(validRegionMask);
    
    % Map old region IDs to consecutive IDs
    regIds = zeros(size(regIdsOld));
    if ~isempty(regIdsOld)
        [tf,loc] = ismember(regIdsOld, validRegionIds);
        regIds(tf) = loc(tf);
        frameIndicesForRegions = frameIndicesForRegions(tf);
    end
    
    % Count frames per region and store frame indices
    if ~isempty(regIds)
        edges = 0.5:1:(double(numRegions)+0.5);
        c = histcounts(regIds, edges);
        counts(i,:) = c;
        
        % Store frame indices for each region
        for regionIdx = 1:numRegions
            regionFrames = frameIndicesForRegions(regIds == regionIdx);
            frameIndices{i, regionIdx} = regionFrames;
        end
    end

    % Plot per-video overlay
    figv = figure('Name', sprintf('Overlay %s', fileNames{i}), 'Position',[50 50 900 900]);
    imagesc(D); axis equal off; colormap(flipud(gray)); caxis([0 max(D(:))*0.8]); hold on;
    scatter(llbwb(:,2), llbwb(:,1), 1, 'k', '.');
    xImgPlot = xImg; yImgPlot = yImg;
    % subsample for plotting speed if huge
    showN = min(50000, numel(xImgPlot));
    if showN > 0
        idxShow = round(linspace(1, numel(xImgPlot), showN));
        scatter(xImgPlot(idxShow), yImgPlot(idxShow), 2, 'c', '.');
    end
    title(strrep(fileNames{i},'_','\_'));
    saveas(figv, fullfile(perVideoDir, sprintf('%03d_%s.png', i, sanitize_filename(fileNames{i}))));
    close(figv);
end

% Save CSV with counts per region
T = array2table(counts, 'VariableNames', compose('Region_%d', 1:numRegions));
T.File = fileNames(:);
T = movevars(T, 'File', 'Before', 1);
writetable(T, fullfile(csvDir, 'per_file_region_counts.csv'));

fprintf('Saved main map, per-video overlays, counts, and frame indices to %s\n', outDir);

% Save region label mapping (old watershed id -> new consecutive id) for reference
mapTable = table(validRegionIds', (1:numRegions)', 'VariableNames', {'OriginalLabel','ConsecutiveIndex'});
writetable(mapTable, fullfile(csvDir, 'region_label_mapping.csv'));

% Save frame indices and sequence metadata for each video and region
frameIndicesDir = fullfile(csvDir, 'frame_indices_per_video');
sequenceMetaDir = fullfile(csvDir, 'sequence_metadata');
if ~exist(frameIndicesDir,'dir'), mkdir(frameIndicesDir); end
if ~exist(sequenceMetaDir,'dir'), mkdir(sequenceMetaDir); end

fprintf('Saving frame indices and sequence metadata per video and region...\n');

% Initialize global sequence database for top-35 selection
globalSequenceDatabase = [];

for i = 1:numFiles
    videoName = sanitize_filename(fileNames{i});
    
    % Create a table with frame indices for each region
    maxFrames = 0;
    for j = 1:numRegions
        maxFrames = max(maxFrames, length(frameIndices{i,j}));
    end
    
    % Create table with padded frame indices (NaN for missing values)
    frameData = NaN(maxFrames, numRegions);
    for j = 1:numRegions
        regionFrames = frameIndices{i,j};
        if ~isempty(regionFrames)
            frameData(1:length(regionFrames), j) = regionFrames;
        end
    end
    
    % Convert to table and save frame indices
    regionNames = compose('Region_%d', 1:numRegions);
    frameTable = array2table(frameData, 'VariableNames', regionNames);
    csvFileName = fullfile(frameIndicesDir, sprintf('%03d_%s_frame_indices.csv', i, videoName));
    writetable(frameTable, csvFileName, 'WriteRowNames', false);
    
    % Analyze sequences for this video and add to global database
    sequenceMetadata = [];
    
    for j = 1:numRegions
        regionFrames = frameIndices{i,j};
        if length(regionFrames) < 5, continue; end  % Skip very short sequences
        
        % Find continuous sequences
        sequences = findContinuousSequences(regionFrames);
        
        for seqIdx = 1:length(sequences)
            seq = sequences{seqIdx};
            if length(seq) < 10, continue; end  % Minimum sequence length
            
            % Calculate sequence quality metrics
            [seqQuality, avgDensity, motionVariance] = calculateSequenceQuality(i, j, seq, zAll, D, LL2, validRegionIds, mapBounds, gridSize);
            
            % Store sequence metadata
            seqMeta = struct();
            seqMeta.VideoIndex = i;
            seqMeta.VideoName = fileNames{i};
            seqMeta.RegionIndex = j;
            seqMeta.SequenceIndex = seqIdx;
            seqMeta.StartFrame = seq(1);
            seqMeta.EndFrame = seq(end);
            seqMeta.Length = length(seq);
            seqMeta.Quality = seqQuality;
            seqMeta.AvgDensity = avgDensity;
            seqMeta.MotionVariance = motionVariance;
            seqMeta.Frames = seq;  % Store actual frame indices
            
            sequenceMetadata = [sequenceMetadata; seqMeta];
            globalSequenceDatabase = [globalSequenceDatabase; seqMeta];
        end
    end
    
    % Save sequence metadata for this video
    if ~isempty(sequenceMetadata)
        metaTable = struct2table(sequenceMetadata);
        metaFileName = fullfile(sequenceMetaDir, sprintf('%03d_%s_sequences.csv', i, videoName));
        writetable(metaTable, metaFileName);
    end
end

% Save a summary file showing frame count per region per video
summaryTable = array2table(counts, 'VariableNames', compose('Region_%d', 1:numRegions));
summaryTable.File = fileNames(:);
summaryTable = movevars(summaryTable, 'File', 'Before', 1);
writetable(summaryTable, fullfile(csvDir, 'frame_indices_summary.csv'));

% Save global sequence database for top-35 selection
if ~isempty(globalSequenceDatabase)
    fprintf('Saving global sequence database (%d sequences)...\n', length(globalSequenceDatabase));
    globalTable = struct2table(globalSequenceDatabase);
    
    % Sort by quality score (descending) for easy top-35 selection
    [~, sortIdx] = sort([globalSequenceDatabase.Quality], 'descend');
    globalTable = globalTable(sortIdx, :);
    
    % Add rank column
    globalTable.QualityRank = (1:height(globalTable))';
    globalTable = movevars(globalTable, 'QualityRank', 'Before', 1);
    
    writetable(globalTable, fullfile(csvDir, 'global_sequence_database.csv'));
    
    % Create top-35 sequences table for easy video extraction
    top35Table = globalTable(1:min(35, height(globalTable)), :);
    writetable(top35Table, fullfile(csvDir, 'top_35_sequences_all_regions.csv'));
    
    % Create per-region top-35 tables
    fprintf('Creating per-region top-35 sequence tables...\n');
    for j = 1:numRegions
        regionSeqs = globalTable(globalTable.RegionIndex == j, :);
        if height(regionSeqs) > 0
            top35Region = regionSeqs(1:min(35, height(regionSeqs)), :);
            regionFileName = fullfile(csvDir, sprintf('top_35_sequences_region_%d.csv', j));
            writetable(top35Region, regionFileName);
            fprintf('  Region %d: %d sequences (top 35 saved)\n', j, min(35, height(regionSeqs)));
        end
    end
end

fprintf('Frame indices and sequence metadata saved to %s\n', outDir);

function fn = sanitize_filename(s)
    fn = regexprep(s, '[^\w\-\.]+', '_');
end

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

function sequences = findContinuousSequences(frameIndices)
% Find continuous sequences in frame indices
    if isempty(frameIndices)
        sequences = {};
        return;
    end
    
    % Sort frame indices
    sortedFrames = sort(frameIndices);
    sequences = {};
    currentSeq = [sortedFrames(1)];
    
    for i = 2:length(sortedFrames)
        if sortedFrames(i) == sortedFrames(i-1) + 1
            % Consecutive frame
            currentSeq = [currentSeq, sortedFrames(i)];
        else
            % Gap found, save current sequence and start new one
            if length(currentSeq) > 0
                sequences{end+1} = currentSeq;
            end
            currentSeq = [sortedFrames(i)];
        end
    end
    
    % Add the last sequence
    if length(currentSeq) > 0
        sequences{end+1} = currentSeq;
    end
end

function [quality, avgDensity, motionVariance] = calculateSequenceQuality(videoIdx, regionIdx, sequence, zAll, D, LL2, validRegionIds, mapBounds, gridSize)
% Calculate quality metrics for a behavioral sequence
    try
        % Get embedding coordinates for this video
        z = zAll{videoIdx};
        if isempty(z) || size(z,1) < max(sequence)
            quality = 0; avgDensity = 0; motionVariance = 0;
            return;
        end
        
        % Get coordinates for frames in this sequence
        seqCoords = z(sequence, :);
        
        % Map to image coordinates
        xImg = round((seqCoords(:,1) - mapBounds(1)) / (mapBounds(2) - mapBounds(1)) * (gridSize-1) + 1);
        yImg = round((seqCoords(:,2) - mapBounds(1)) / (mapBounds(2) - mapBounds(1)) * (gridSize-1) + 1);
        
        % Keep valid coordinates
        valid = xImg>=1 & xImg<=gridSize & yImg>=1 & yImg<=gridSize;
        xImg = xImg(valid); yImg = yImg(valid);
        seqCoords = seqCoords(valid, :);
        
        if isempty(seqCoords)
            quality = 0; avgDensity = 0; motionVariance = 0;
            return;
        end
        
        % Calculate average density in this region
        linIdx = sub2ind(size(D), yImg, xImg);
        densityValues = D(linIdx);
        avgDensity = mean(densityValues);
        
        % Calculate motion variance (behavioral diversity)
        if size(seqCoords, 1) > 1
            motionVectors = diff(seqCoords, 1, 1);  % Frame-to-frame displacement
            motionMagnitudes = sqrt(sum(motionVectors.^2, 2));
            motionVariance = var(motionMagnitudes);
        else
            motionVariance = 0;
        end
        
        % Calculate sequence compactness (lower = more compact/stable behavior)
        centroid = mean(seqCoords, 1);
        distances = sqrt(sum((seqCoords - centroid).^2, 2));
        compactness = mean(distances);
        
        % Composite quality score (higher is better)
        % Factors: length, density, motion diversity, but penalize excessive spread
        lengthScore = log(length(sequence)) / log(1000);  % Normalized to 1000 frames max
        densityScore = avgDensity / max(D(:));  % Normalized to max density
        motionScore = min(1, motionVariance / 100);  % Capped motion diversity
        compactnessScore = 1 / (1 + compactness/50);  % Penalty for excessive spread
        
        quality = lengthScore * 0.4 + densityScore * 0.3 + motionScore * 0.2 + compactnessScore * 0.1;
        
    catch ME
        % Handle errors gracefully
        quality = 0; avgDensity = 0; motionVariance = 0;
    end
end
