function analysis_results = analyze_maps_and_counts(varargin)
%% ANALYZE_MAPS_AND_COUNTS Downstream analysis of behavioral embedding results
% This function performs comprehensive analysis of saved embedding maps and
% region counts, generating visualizations and statistical summaries.
%
% Usage:
%   results = analyze_maps_and_counts()
%   results = analyze_maps_and_counts('config', config)
%   results = analyze_maps_and_counts('input_dir', path, 'output_dir', path)
%
% Inputs (optional):
%   config - Configuration structure
%   input_dir - Directory containing input files (default: current)
%   output_dir - Directory for outputs (default: 'outputs/analysis')
%   map_file - Watershed map file (default: 'watershed_SNI_TBI.mat')
%   results_file - Embedding results file (default: 'complete_embedding_results_SNI_TBI.mat')
%
% Outputs:
%   analysis_results - Structure containing analysis results and metadata

%% Parse inputs
p = inputParser;
addParameter(p, 'config', [], @isstruct);
addParameter(p, 'input_dir', '.', @ischar);
addParameter(p, 'output_dir', 'outputs/analysis', @ischar);
addParameter(p, 'map_file', 'watershed_SNI_TBI.mat', @ischar);
addParameter(p, 'results_file', 'complete_embedding_results_SNI_TBI.mat', @ischar);
addParameter(p, 'verbose', true, @islogical);
parse(p, varargin{:});

config = p.Results.config;
input_dir = p.Results.input_dir;
output_dir = p.Results.output_dir;
map_file = p.Results.map_file;
results_file = p.Results.results_file;
verbose = p.Results.verbose;

% Load default config if not provided
if isempty(config)
    if exist('config/pipeline_config.m', 'file')
        run('config/pipeline_config.m');
        config = pipeline_config();
    else
        % Use default options
        config = create_default_config();
    end
end

if verbose
    fprintf('\n=== Behavioral Map Analysis ===\n');
    fprintf('Started: %s\n\n', datestr(now));
end

%% Setup output directories
outDir = output_dir;
figDir = fullfile(outDir, 'figures');
perVideoDir = fullfile(figDir, 'per_video');
csvDir = fullfile(outDir, 'csv');
statDir = fullfile(csvDir, 'statistical_analysis');

dirs = {outDir, figDir, perVideoDir, csvDir, statDir, ...
        fullfile(csvDir, 'frame_indices_per_video'), ...
        fullfile(csvDir, 'sequence_metadata')};

for i = 1:length(dirs)
    if ~exist(dirs{i},'dir'), mkdir(dirs{i}); end
end

%% Analysis options from config
if isfield(config, 'analysis')
    opts = config.analysis;
else
    opts = struct();
    opts.displayRangeExpand = 1;
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
end

%% Load saved results
if verbose, fprintf('Loading analysis input files...\n'); end

mapPath = fullfile(input_dir, map_file);
resPath = fullfile(input_dir, results_file);

if ~exist(mapPath, 'file')
    error('Map file not found: %s', mapPath);
end
if ~exist(resPath, 'file')
    error('Results file not found: %s', resPath);
end

load(mapPath, 'D','LL','LL2','llbwb','xx');
S = load(resPath);
results = S.results;

%% Optionally re-create watershed segmentation
if opts.resegmentEnable
    if verbose, fprintf('Re-segmenting watershed regions...\n'); end
    
    Dw = D;
    if opts.resegmentSigma > 0
        if exist('imgaussfilt','file')
            Dw = imgaussfilt(Dw, opts.resegmentSigma);
        else
            ksz = max(1, ceil(6*opts.resegmentSigma));
            K = ones(ksz, ksz) / max(1, (ksz*ksz));
            Dw = conv2(Dw, K, 'same');
        end
    end
    if opts.resegmentGamma ~= 1
        Dw = Dw .^ opts.resegmentGamma;
    end
    
    LL = watershed(-Dw, opts.resegmentConnectivity);
    LL2 = LL;
    
    % Mask low densities as background
    backgroundMask = Dw < opts.resegmentMinDensity;
    LL2(backgroundMask) = -1;
    
    % Fill holes in regions if enabled
    if opts.resegmentFillHoles
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
            fillPixels = filledMask & ~regionMask & (LL2 <= 0);
            LL2(fillPixels) = regionId;
        end
    end
    
    % Remove small regions if enabled
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
    llbwb = combineCells_local(C);
end

%% Filter watershed regions by density if enabled
if opts.filterRegionsByDensity
    if verbose, fprintf('Filtering regions by density...\n'); end
    
    allRegionIds = unique(LL2(LL2>0));
    regionAvgDensity = zeros(size(allRegionIds));
    for i = 1:numel(allRegionIds)
        regionId = allRegionIds(i);
        regionMask = (LL2 == regionId);
        regionDensities = D(regionMask);
        regionAvgDensity(i) = mean(regionDensities);
    end
    
    densityThreshold = max(opts.regionMinDensity, ...
        prctile(regionAvgDensity, opts.regionDensityPercentile));
    
    validRegionMask = regionAvgDensity >= densityThreshold;
    validRegionIds = allRegionIds(validRegionMask);
    
    LL2_filtered = LL2;
    excludedRegions = allRegionIds(~validRegionMask);
    for i = 1:numel(excludedRegions)
        LL2_filtered(LL2 == excludedRegions(i)) = -1;
    end
    LL2 = LL2_filtered;
    
    % Recompute boundaries from filtered regions
    LLBW = (LL2 == 0);
    B = bwboundaries(LLBW);
    if numel(B) >= 2, C = B(2:end); else, C = B; end
    llbwb = combineCells_local(C);
else
    validRegionIds = unique(LL2(LL2>0));
end

validRegionIds = validRegionIds(:)';
numRegions = numel(validRegionIds);

%% Plot main behavioral density map with region indices
if verbose, fprintf('Creating main behavioral map...\n'); end

fig = figure('Name','Behavioral Map with Region Indices','Position',[100 100 900 900]);
imagesc(D); axis equal off; colormap(flipud(gray)); caxis([0 max(D(:))*0.8]); hold on;
scatter(llbwb(:,2), llbwb(:,1), 1, 'k', '.');

% Add region indices
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

%% Process individual videos
if verbose, fprintf('Processing individual videos...\n'); end

fileNames = results.reembedding_labels_all;
zAll = results.zEmbeddings_all;
numFiles = numel(fileNames);

% Helper to convert z coordinates to image indices
mapBounds = [min(xx) max(xx)];
if opts.displayRangeExpand ~= 1
    ctr = mean(mapBounds);
    halfR = (mapBounds(2)-mapBounds(1))/2 * opts.displayRangeExpand;
    mapBounds = [ctr - halfR, ctr + halfR];
end
gridSize = size(D,1);

% Initialize data structures
counts = zeros(numFiles, numRegions);
frameIndices = cell(numFiles, numRegions);
for i = 1:numFiles
    for j = 1:numRegions
        frameIndices{i,j} = [];
    end
end

% Process each video
for i = 1:numFiles
    z = zAll{i};
    if isempty(z) || size(z,2)~=2
        continue;
    end
    
    % Map z to image coordinates
    xImg = round((z(:,1) - mapBounds(1)) / (mapBounds(2)-mapBounds(1)) * (gridSize-1) + 1);
    yImg = round((z(:,2) - mapBounds(1)) / (mapBounds(2)-mapBounds(1)) * (gridSize-1) + 1);
    valid = xImg>=1 & xImg<=gridSize & yImg>=1 & yImg<=gridSize;
    xImg = xImg(valid); yImg = yImg(valid);

    % Count per region and track frame indices
    linIdx = sub2ind(size(LL2), yImg, xImg);
    regIdsOld = double(LL2(linIdx));
    
    validFrameIndices = find(valid);
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
        
        for regionIdx = 1:numRegions
            regionFrames = frameIndicesForRegions(regIds == regionIdx);
            frameIndices{i, regionIdx} = regionFrames;
        end
    end

    % Create per-video overlay plot
    if config.output.generate_figures
        figv = figure('Name', sprintf('Overlay %s', fileNames{i}), 'Position',[50 50 900 900]);
        imagesc(D); axis equal off; colormap(flipud(gray)); caxis([0 max(D(:))*0.8]); hold on;
        scatter(llbwb(:,2), llbwb(:,1), 1, 'k', '.');
        
        % Plot video points (subsample if too many)
        showN = min(50000, numel(xImg));
        if showN > 0
            idxShow = round(linspace(1, numel(xImg), showN));
            scatter(xImg(idxShow), yImg(idxShow), 2, 'c', '.');
        end
        title(strrep(fileNames{i},'_','\_'));
        saveas(figv, fullfile(perVideoDir, sprintf('%03d_%s.png', i, sanitize_filename(fileNames{i}))));
        close(figv);
    end
end

%% Save CSV outputs
if verbose, fprintf('Saving CSV outputs...\n'); end

% Main counts table
T = array2table(counts, 'VariableNames', compose('Region_%d', 1:numRegions));
T.File = fileNames(:);
T = movevars(T, 'File', 'Before', 1);
writetable(T, fullfile(csvDir, 'per_file_region_counts.csv'));

% Region label mapping
mapTable = table(validRegionIds', (1:numRegions)', 'VariableNames', {'OriginalLabel','ConsecutiveIndex'});
writetable(mapTable, fullfile(csvDir, 'region_label_mapping.csv'));

% Frame indices per video
frameIndicesDir = fullfile(csvDir, 'frame_indices_per_video');
for i = 1:numFiles
    videoName = sanitize_filename(fileNames{i});
    
    maxFrames = 0;
    for j = 1:numRegions
        maxFrames = max(maxFrames, length(frameIndices{i,j}));
    end
    
    frameData = NaN(maxFrames, numRegions);
    for j = 1:numRegions
        regionFrames = frameIndices{i,j};
        if ~isempty(regionFrames)
            frameData(1:length(regionFrames), j) = regionFrames;
        end
    end
    
    regionNames = compose('Region_%d', 1:numRegions);
    frameTable = array2table(frameData, 'VariableNames', regionNames);
    csvFileName = fullfile(frameIndicesDir, sprintf('%03d_%s_frame_indices.csv', i, videoName));
    writetable(frameTable, csvFileName, 'WriteRowNames', false);
end

%% Sequence analysis and metadata
if verbose, fprintf('Analyzing behavioral sequences...\n'); end

globalSequenceDatabase = [];

for i = 1:numFiles
    sequenceMetadata = [];
    
    for j = 1:numRegions
        regionFrames = frameIndices{i,j};
        if length(regionFrames) < 5, continue; end
        
        sequences = findContinuousSequences(regionFrames);
        
        for seqIdx = 1:length(sequences)
            seq = sequences{seqIdx};
            if length(seq) < 10, continue; end
            
            [seqQuality, avgDensity, motionVariance] = calculateSequenceQuality(...
                i, j, seq, zAll, D, LL2, validRegionIds, mapBounds, gridSize);
            
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
            
            sequenceMetadata = [sequenceMetadata; seqMeta];
            globalSequenceDatabase = [globalSequenceDatabase; seqMeta];
        end
    end
    
    % Save per-video sequence metadata
    if ~isempty(sequenceMetadata)
        metaTable = struct2table(sequenceMetadata);
        metaFileName = fullfile(csvDir, 'sequence_metadata', ...
            sprintf('%03d_%s_sequences.csv', i, sanitize_filename(fileNames{i})));
        writetable(metaTable, metaFileName);
    end
end

% Save global sequence database
if ~isempty(globalSequenceDatabase)
    globalTable = struct2table(globalSequenceDatabase);
    [~, sortIdx] = sort([globalSequenceDatabase.Quality], 'descend');
    globalTable = globalTable(sortIdx, :);
    globalTable.QualityRank = (1:height(globalTable))';
    globalTable = movevars(globalTable, 'QualityRank', 'Before', 1);
    writetable(globalTable, fullfile(csvDir, 'global_sequence_database.csv'));
    
    % Create top-35 sequences tables
    top35Table = globalTable(1:min(35, height(globalTable)), :);
    writetable(top35Table, fullfile(csvDir, 'top_35_sequences_all_regions.csv'));
    
    for j = 1:numRegions
        regionSeqs = globalTable(globalTable.RegionIndex == j, :);
        if height(regionSeqs) > 0
            top35Region = regionSeqs(1:min(35, height(regionSeqs)), :);
            regionFileName = fullfile(csvDir, sprintf('top_35_sequences_region_%d.csv', j));
            writetable(top35Region, regionFileName);
        end
    end
end

%% Statistical analysis
if verbose, fprintf('Performing statistical analysis...\n'); end

% Group-based analysis if group information is available
if isfield(results, 'group_info') && ~isempty(results.group_info)
    stat_results = perform_group_statistical_analysis(counts, fileNames, results.group_info, numRegions);
    
    % Save statistical results
    writetable(struct2table(stat_results.summary), fullfile(statDir, 'group_summary_statistics.csv'));
    if ~isempty(stat_results.pairwise)
        writetable(struct2table(stat_results.pairwise), fullfile(statDir, 'pairwise_statistical_tests.csv'));
    end
end

%% Prepare results structure
analysis_results = struct();
analysis_results.timestamp = datestr(now);
analysis_results.config = config;
analysis_results.options = opts;
analysis_results.data = struct();
analysis_results.data.num_files = numFiles;
analysis_results.data.num_regions = numRegions;
analysis_results.data.region_mapping = validRegionIds;
analysis_results.data.counts = counts;
analysis_results.data.frame_indices = frameIndices;

analysis_results.outputs = struct();
analysis_results.outputs.output_dir = outDir;
analysis_results.outputs.figures_dir = figDir;
analysis_results.outputs.csv_dir = csvDir;

if exist('globalSequenceDatabase', 'var') && ~isempty(globalSequenceDatabase)
    analysis_results.sequences = struct();
    analysis_results.sequences.total_sequences = length(globalSequenceDatabase);
    analysis_results.sequences.database = globalSequenceDatabase;
end

if exist('stat_results', 'var')
    analysis_results.statistics = stat_results;
end

% Save analysis results
save(fullfile(outDir, 'analysis_results.mat'), 'analysis_results');

if verbose
    fprintf('\n=== Analysis Complete ===\n');
    fprintf('Processed %d files with %d behavioral regions\n', numFiles, numRegions);
    fprintf('Outputs saved to: %s\n', outDir);
    fprintf('Completed: %s\n\n', datestr(now));
end

end

%% Helper Functions

function config = create_default_config()
    config = struct();
    config.output = struct();
    config.output.generate_figures = true;
    config.analysis = struct();
    config.analysis.displayRangeExpand = 1;
    config.analysis.filterRegionsByDensity = false;
    config.analysis.resegmentEnable = true;
    config.analysis.resegmentGamma = 1.7;
    config.analysis.resegmentMinDensity = 5e-6;
    config.analysis.resegmentConnectivity = 4;
    config.analysis.resegmentFillHoles = true;
    config.analysis.resegmentMinRegionSize = 10;
    config.analysis.forceAllBoundaries = true;
end

function fn = sanitize_filename(s)
    fn = regexprep(s, '[^\w\-\.]+', '_');
end

function out = combineCells_local(c)
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
    if isempty(frameIndices)
        sequences = {};
        return;
    end
    
    sortedFrames = sort(frameIndices);
    sequences = {};
    currentSeq = [sortedFrames(1)];
    
    for i = 2:length(sortedFrames)
        if sortedFrames(i) == sortedFrames(i-1) + 1
            currentSeq = [currentSeq, sortedFrames(i)];
        else
            if length(currentSeq) > 0
                sequences{end+1} = currentSeq;
            end
            currentSeq = [sortedFrames(i)];
        end
    end
    
    if length(currentSeq) > 0
        sequences{end+1} = currentSeq;
    end
end

function [quality, avgDensity, motionVariance] = calculateSequenceQuality(videoIdx, regionIdx, sequence, zAll, D, LL2, validRegionIds, mapBounds, gridSize)
    try
        z = zAll{videoIdx};
        if isempty(z) || size(z,1) < max(sequence)
            quality = 0; avgDensity = 0; motionVariance = 0;
            return;
        end
        
        seqCoords = z(sequence, :);
        
        xImg = round((seqCoords(:,1) - mapBounds(1)) / (mapBounds(2) - mapBounds(1)) * (gridSize-1) + 1);
        yImg = round((seqCoords(:,2) - mapBounds(1)) / (mapBounds(2) - mapBounds(1)) * (gridSize-1) + 1);
        
        valid = xImg>=1 & xImg<=gridSize & yImg>=1 & yImg<=gridSize;
        xImg = xImg(valid); yImg = yImg(valid);
        seqCoords = seqCoords(valid, :);
        
        if isempty(seqCoords)
            quality = 0; avgDensity = 0; motionVariance = 0;
            return;
        end
        
        linIdx = sub2ind(size(D), yImg, xImg);
        densityValues = D(linIdx);
        avgDensity = mean(densityValues);
        
        if size(seqCoords, 1) > 1
            motionVectors = diff(seqCoords, 1, 1);
            motionMagnitudes = sqrt(sum(motionVectors.^2, 2));
            motionVariance = var(motionMagnitudes);
        else
            motionVariance = 0;
        end
        
        centroid = mean(seqCoords, 1);
        distances = sqrt(sum((seqCoords - centroid).^2, 2));
        compactness = mean(distances);
        
        lengthScore = log(length(sequence)) / log(1000);
        densityScore = avgDensity / max(D(:));
        motionScore = min(1, motionVariance / 100);
        compactnessScore = 1 / (1 + compactness/50);
        
        quality = lengthScore * 0.4 + densityScore * 0.3 + motionScore * 0.2 + compactnessScore * 0.1;
        
    catch
        quality = 0; avgDensity = 0; motionVariance = 0;
    end
end

function stat_results = perform_group_statistical_analysis(counts, fileNames, group_info, numRegions)
    % Placeholder for statistical analysis
    % This would implement group comparisons, ANOVA, etc.
    stat_results = struct();
    stat_results.summary = struct();
    stat_results.pairwise = [];
    
    % Add actual statistical analysis here based on group_info
    warning('Statistical analysis not fully implemented - placeholder results returned');
end