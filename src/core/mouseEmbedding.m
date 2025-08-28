%% Mouse Embedding for week1 and week2 data with left-right flipping
% Revised version for individual mice from different experimental groups
% Includes left-right flipping to create mirror populations

clear; close all; clc;

%% Load mouseFileOrder and MAT files
load('mouseFileOrder.mat', 'mouseOrderShort', 'metadata');

% Convert to cell array if needed
if ~iscell(mouseOrderShort)
    if ischar(mouseOrderShort)
        mouseOrderShort = cellstr(mouseOrderShort);
    elseif isstring(mouseOrderShort)
        mouseOrderShort = cellstr(mouseOrderShort);
    end
end

% Check the loaded data
fprintf('Type of mouseOrderShort: %s\n', class(mouseOrderShort));
fprintf('Size of mouseOrderShort: %s\n', mat2str(size(mouseOrderShort)));

% Find actual MAT files 
validIdx = ~cellfun(@isempty, mouseOrderShort);
validFiles = mouseOrderShort(validIdx);
numMice = length(validFiles);
mouseFiles = validFiles;

fprintf('Found %d valid files\n', numMice);

%% Load all mouse data
allMOUSE = cell(numMice, 1);
mouseGroups = cell(numMice, 1);  % Store group labels 
mouseWeeks = cell(numMice, 1);   % Store week labels

fprintf('Loading %d mouse files...\n', numMice);
for i = 1:numMice
    matPath = mouseFiles{i};
    fprintf('Loading %s...\n', matPath);
    
    % Determine week and group from metadata
    if ismember(i, metadata.week1_indices.DRG)
        mouseWeeks{i} = 'week1';
        mouseGroups{i} = 'DRG';
    elseif ismember(i, metadata.week1_indices.SC)
        mouseWeeks{i} = 'week1';
        mouseGroups{i} = 'SC';
    elseif ismember(i, metadata.week2_indices.DRG)
        mouseWeeks{i} = 'week2';
        mouseGroups{i} = 'DRG';
    elseif ismember(i, metadata.week2_indices.IT)
        mouseWeeks{i} = 'week2';
        mouseGroups{i} = 'IT';
    elseif ismember(i, metadata.week2_indices.SC)
        mouseWeeks{i} = 'week2';
        mouseGroups{i} = 'SC';
    elseif ismember(i, metadata.week2_indices.SNI)
        mouseWeeks{i} = 'week2';
        mouseGroups{i} = 'SNI';
    else
        error('Could not determine week/group for file index %d', i);
    end
    
    data = load(matPath);
    if isfield(data, 'pred')
        % Data format: frames x 3 x 23 (coordinates x joints)
        allMOUSE{i} = data.pred;
    else
        error('File %s does not have expected pred structure', matPath);
    end
end

fprintf('Successfully loaded %d files\n', numMice);
fprintf('Week 1: %d files, Week 2: %d files\n', ...
    sum(strcmp(mouseWeeks, 'week1')), sum(strcmp(mouseWeeks, 'week2')));

%% Create flipped versions of the data
% Left-right flip by negating x-coordinates AND swapping left/right joints
allMOUSE_flipped = cell(numMice, 1);

% Define left-right joint mapping for rat23 format
% Center joints stay the same: 1(Snout), 4-7(Spine/Tail)
% Left-right pairs to swap:
jointMapping = 1:23;  % Initialize with identity mapping
jointMapping([2, 3]) = [3, 2];      % EarL <-> EarR
jointMapping([8:11, 12:15]) = [12:15, 8:11];   % Left arm <-> Right arm
jointMapping([16:19, 20:23]) = [20:23, 16:19]; % Left leg <-> Right leg

for i = 1:numMice
    flipped_data = allMOUSE{i};
    
    % Create a copy to work with
    temp_data = zeros(size(flipped_data));
    
    % Swap left and right joints
    for j = 1:23
        temp_data(:, :, j) = flipped_data(:, :, jointMapping(j));
    end
    
    % Negate x-coordinates to complete the mirror
    temp_data(:, 1, :) = -temp_data(:, 1, :);
    
    allMOUSE_flipped{i} = temp_data;
end

% Combine original and flipped data for training
allMOUSE_combined = [allMOUSE; allMOUSE_flipped];

%% Set up skeleton
% Skeleton for rat23 format
joints_idx = [1 2; 1 3; 2 3; ...  % head connections
    1 4; 4 5; 5 6; 6 7; ...       % spine
    4 8; 8 9; 9 10; 10 11; ...    % left arm
    4 12; 12 13; 13 14; 14 15; ... % right arm
    6 16; 16 17; 17 18; 18 19; ... % left leg
    6 20; 20 21; 21 22; 22 23];    % right leg

% Define colors for visualization
chead = [1 .6 .2]; % orange
cspine = [.2 .635 .172]; % green
cLF = [0 0 1]; % blue
cRF = [1 0 0]; % red
cLH = [0 1 1]; % cyan
cRH = [1 0 1]; % magenta

scM = [chead; chead; chead; cspine; cspine; cspine; cspine; ...
    cLF; cLF; cLF; cLF; cRF; cRF; cRF; cRF; ...
    cLH; cLH; cLH; cLH; cRH; cRH; cRH; cRH];

skeleton.color = scM;
skeleton.joints_idx = joints_idx;

%% Visualize single animal
close all
figure('Name','Mouse Test - Original');
h = Keypoint3DAnimator(allMOUSE{1}, skeleton, 'MarkerSize', 15);
set(gca,'Color','w')
axis equal;
axis([-200 250 -150 350 -10 150]);
view(h.getAxes,-30,40);
title('Original Mouse');

figure('Name','Mouse Test - Flipped');
h2 = Keypoint3DAnimator(allMOUSE_flipped{1}, skeleton, 'MarkerSize', 15);
set(gca,'Color','w')
axis equal;
axis([-200 250 -150 350 -10 150]);
view(h2.getAxes,-30,40);
title('Flipped Mouse');

%% BEHAVIORAL EMBEDDING

% Define joint pairs for distance calculations
xIdx = 1:23; yIdx = 1:23;
[Xi, Yi] = meshgrid(xIdx,yIdx);
Xi = Xi(:); Yi = Yi(:);
IDX = find(Xi~=Yi);
nx = length(xIdx);

% PCA parameters
firstBatch = true;
currentImage = 0;
batchSize = 30000;
mu = zeros(1, 506);

% Calculate characteristic length for each mouse (original and flipped)
lengtht = zeros(numMice*2, 1);
for i = 1:numMice*2
    ma1 = allMOUSE_combined{i};
    % Calculate distance between keypoints 1 and 7 (nose to tail base)  
    sj = returnDist3d(squeeze(ma1(:,:,1)), squeeze(ma1(:,:,7)));  
    % Use 95th percentile as the characteristic length  
    lengtht(i) = prctile(sj, 95);
end

%% PCA on all data (original + flipped)
fprintf('\nPerforming PCA on combined data...\n');
for j = 1:length(allMOUSE_combined)
    fprintf('Processing mouse %d/%d\n', j, length(allMOUSE_combined));
    ma1 = allMOUSE_combined{j};
        nn1 = size(ma1,1);
        p1Dist = zeros(nx^2,size(ma1,1));
        for i = 1:size(p1Dist,1)
            p1Dist(i,:) = returnDist3d(squeeze(ma1(:,:,Xi(i))),squeeze(ma1(:,:,Yi(i))));
        end
        p1Dsmooth = zeros(size(p1Dist));
        for i = 1:size(p1Dist,1)
            p1Dsmooth(i,:) = smooth(medfilt1(p1Dist(i,:),3),3);
        end
    
        p1Dist = p1Dsmooth(IDX,:)';
        
    % Scale by characteristic length
        scaleVal = lengtht(j)./90;
        p1Dist = p1Dist.*scaleVal;
        
        if firstBatch
            firstBatch = false;
            if size(p1Dist,1) < batchSize
                cBatchSize = size(p1Dist,1);
                X = p1Dist;
            else
                cBatchSize = batchSize;
                X = p1Dist;
            end
            currentImage = cBatchSize;
            mu = sum(X);
            C = cov(X).*cBatchSize + (mu'*mu)./ cBatchSize;
        else
            if size(p1Dist,1) < batchSize
                cBatchSize = size(p1Dist,1);
                X = p1Dist;
            else
                cBatchSize = batchSize;
                X = p1Dist(randperm(size(p1Dist,1),cBatchSize),:);
            end
            tempMu = sum(X);
            mu = mu + tempMu;
            C = C + cov(X).*cBatchSize + (tempMu'*tempMu)./cBatchSize;
            currentImage = currentImage + cBatchSize;
    end
end

L = currentImage; mu = mu ./ L; C = C ./ L - mu'*mu;
fprintf('Finding Principal Components\n');
[vecs,vals] = eig(C); vals = flipud(diag(vals)); vecs = fliplr(vecs);
mus = mu;

% Save PCA results
save('vecsValsMouse_weekdata.mat','C','L','mus','vals','vecs');

%% Create behavioral embedding
vecs15 = vecs(:,1:15);
minF = .5; maxF = 20; pcaModes = 20; numModes = pcaModes;
parameters = setRunParameters([]);
parameters.samplingFreq = 50;
parameters.minF = minF;
parameters.maxF = maxF;
% Ensure required fields for downstream MotionMapper functions
if ~isfield(parameters, 'kdNeighbors'); parameters.kdNeighbors = 5; end
if ~isfield(parameters, 'minTemplateLength'); parameters.minTemplateLength = 10; end
if ~isfield(parameters, 'sigmaTolerance'); parameters.sigmaTolerance = 1e-5; end
if ~isfield(parameters, 'maxNeighbors'); parameters.maxNeighbors = 200; end
numPerDataSet = 320;

mD = cell(size(allMOUSE_combined)); 
mA = cell(size(allMOUSE_combined));

fprintf('\nCreating behavioral embeddings...\n');
for j = 1:length(allMOUSE_combined)
    fprintf('Processing embedding %d/%d\n', j, length(allMOUSE_combined));
    ma1 = allMOUSE_combined{j};

    nn1 = size(ma1,1);
    p1Dist = zeros(nx^2,size(ma1,1));
    for i = 1:size(p1Dist,1)
        p1Dist(i,:) = returnDist3d(squeeze(ma1(:,:,Xi(i))),squeeze(ma1(:,:,Yi(i))));
    end

    p1Dsmooth = zeros(size(p1Dist));
    for i = 1:size(p1Dist,1)
        p1Dsmooth(i,:) = smooth(medfilt1(p1Dist(i,:),3),3);
    end
    p1Dist = p1Dsmooth(IDX,:)';
    
    % Get floor value and scale
    allz = squeeze(ma1(:,3,[19 23])); 
    zz = allz(:);
    fz = prctile(zz,10);
    sj = returnDist3d(squeeze(ma1(:,:,1)),squeeze(ma1(:,:,7)));
    lz = prctile(sj,95);
    scaleVal = 90./lz;
    p1Dist = p1Dist.*scaleVal;
    
    p1 = bsxfun(@minus,p1Dist,mus);
    proj = p1*vecs15;
   
    [data,~] = findWavelets(proj,numModes,parameters);
    
    n = size(p1Dist,1);
    amps = sum(data,2);
    data2 = log(data);
    data2(data2<-5) = -5;
    
    % Joint velocities
    jv = zeros(n,length(xIdx));
    for i = 1:length(xIdx)
        jv(:,i) = [0; medfilt1(sqrt(sum(diff(squeeze(ma1(:,:,xIdx(i)))).^2,2)),10)];
    end
    jv = jv.*scaleVal;
    jv(jv>=5) = 5;
    
    % Z-coordinates
    p1z = zeros(nx,nn1);
    for i = 1:nx
        p1z(i,:) = smooth(medfilt1(squeeze(ma1(:,3,xIdx(i))),3),3);
    end
    allz1 = squeeze(ma1(:,3,[19 23])); 
    zz1 = allz1(:); 
    fz1 = prctile(zz1,10);
    floorval = fz1;
    p1z = (p1z-floorval).*scaleVal; 
    
    nnData = [data2 .25*p1z' .5*jv];
    
    % Subsample for t-SNE
    yData = tsne(nnData(1:20:end,:));
    [signalData,signalAmps] = findTemplatesFromData(...
        nnData(1:20:end,:),yData,amps(1:20:end,:),numPerDataSet,parameters);
    mD{j} = signalData; 
    mA{j} = signalAmps;
end

save('loneSignalDataAmps_weekdata.mat','mA','mD');

%% Create t-SNE embedding on all data
allD = combineCells(mD,1); 
allA = combineCells(mA);

fprintf('\nRunning t-SNE on combined data...\n');
Y = tsne(allD);
save('train_weekdata.mat','Y','allD');

%% Create watershed regions
fprintf('\nCreating watershed regions...\n');
[xx, d] = findPointDensity(Y, 1, 501, [-65 65]);
D = d;

% Watershed
LL = watershed(-d,18);
LL2 = LL; 
LL2(d < 1e-6) = -1;

% Find boundaries
LLBW = LL2==0;
LLBWB = bwboundaries(LLBW);
llbwb = LLBWB(2:end);
llbwb = combineCells(llbwb');

% Plot density map
figure('Name', 'Behavioral Density Map');
imagesc(D); 
axis equal off; 
colormap(flipud(gray)); 
caxis([0 6e-4]);
hold on; 
scatter(llbwb(:,2),llbwb(:,1),'.','k');
title('Behavioral Density Map - All Mice');

%% Re-embed individual mice onto the trained map
fprintf('\nRe-embedding individual mice...\n');
load('train_weekdata.mat','Y','allD');
trainingSetData = allD; 
trainingEmbeddingZ = Y;

wrFINE = cell(numMice*2, 1);
zEmbeddings = cell(numMice*2, 1);  % Store the actual embedding coordinates
parameters.batchSize = 10000;

for j = 1:length(allMOUSE_combined)
    fprintf('Re-embedding mouse %d/%d\n', j, length(allMOUSE_combined));
    ma1 = allMOUSE_combined{j};
    
    % Repeat all the feature extraction
    nn1 = size(ma1,1);
    p1Dist = zeros(nx^2,size(ma1,1));
    for i = 1:size(p1Dist,1)
        p1Dist(i,:) = returnDist3d(squeeze(ma1(:,:,Xi(i))),squeeze(ma1(:,:,Yi(i))));
    end

    p1Dsmooth = zeros(size(p1Dist));
    for i = 1:size(p1Dist,1)
        p1Dsmooth(i,:) = smooth(medfilt1(p1Dist(i,:),3),3);
    end
    p1Dist = p1Dsmooth(IDX,:)';
    
    allz = squeeze(ma1(:,3,[19 23])); 
    zz = allz(:);
    fz = prctile(zz,10);
    sj = returnDist3d(squeeze(ma1(:,:,1)),squeeze(ma1(:,:,7)));
    lz = prctile(sj,95);
    scaleVal = 90./lz;
    p1Dist = p1Dist.*scaleVal;

    p1 = bsxfun(@minus,p1Dist,mus);
    proj = p1*vecs15;

    [data,~] = findWavelets(proj,numModes,parameters);

    n = size(p1Dist,1);
    amps = sum(data,2);
    data2 = log(data);
    data2(data2<-5) = -5;
    
    jv = zeros(n,length(xIdx));
    for i = 1:length(xIdx)
        jv(:,i) = [0; medfilt1(sqrt(sum(diff(squeeze(ma1(:,:,xIdx(i)))).^2,2)),10)];
    end
    jv = jv.*scaleVal;
    jv(jv>=5) = 5;

    p1z = zeros(nx,nn1);
    for i = 1:nx
        p1z(i,:) = smooth(medfilt1(squeeze(ma1(:,3,xIdx(i))),3),3);
    end
    allz1 = squeeze(ma1(:,3,[19 23])); 
    zz1 = allz1(:); 
    fz1 = prctile(zz1,10);
    floorval = fz1;
    p1z = (p1z-floorval).*scaleVal;

    nnData = [data2 .25*p1z' .5*jv];

    % Find embeddings
    [zValues,zCosts,zGuesses,inConvHull,meanMax,exitFlags] = ...
        findTDistributedProjections_fmin(nnData,trainingSetData,...
        trainingEmbeddingZ,[],parameters);

    z = zValues; 
    z(~inConvHull,:) = zGuesses(~inConvHull,:);
    
    % Save the embedding coordinates
    zEmbeddings{j} = z;
    
    % Find watershed regions
vSmooth = .5;
medianLength = 1;
pThreshold = [];
minRest = 5;
obj = [];
    fitOnly = false;
numGMM = 2;

    [wr,~,~,~,~,~,~,~] = findWatershedRegions_v2(z,xx,LL,vSmooth,...
        medianLength,pThreshold,minRest,obj,fitOnly,numGMM);
    
    wrFINE{j} = wr;
end

save('mouseEmbeddingResults_weekdata.mat','wrFINE','zEmbeddings','LL','LL2','D','xx',...
    'mouseGroups','mouseWeeks','mouseFiles','metadata');

%% Create comparison plots for each week with density effect
% Define colors for each group
groupColors = struct();
groupColors.DRG = [1 0 0];     % Red
groupColors.SC = [0 0 1];      % Blue
groupColors.IT = [0 1 0];      % Green
groupColors.SNI = [1 0.5 0];   % Orange

weeks = {'week1', 'week2'};

for w = 1:length(weeks)
    week = weeks{w};
    
    figure('Name', sprintf('%s Group Comparisons - Original vs Flipped', week));
    
    % Get groups for this week
    if strcmp(week, 'week1')
        groups = {'DRG', 'SC'};
    else
        groups = {'DRG', 'IT', 'SC', 'SNI'};
    end
    
    plotIdx = 1;
    for g = 1:length(groups)
        % Original
        subplot(length(groups), 2, plotIdx);
        
        % Find mice in this week/group
        groupIdx = find(strcmp(mouseWeeks, week) & strcmp(mouseGroups, groups{g}));
        
        % Create axes and display background
        ax = gca;
        imagesc(D);
        hold on;
        
        % Collect original mice data
        allPoints = [];
        for idx = groupIdx'
            z = zEmbeddings{idx};
            z_img = (z + 65) * 501 / 130;
            allPoints = [allPoints; z_img];
        end
        
        if ~isempty(allPoints)
            % Create 2D histogram for density
            [N, xedges, yedges] = histcounts2(allPoints(:,1), allPoints(:,2), ...
                linspace(1,501,60), linspace(1,501,60));
            
            % Smooth the density
            N = imgaussfilt(N', 3);
            
            % Create overlay with group color
            overlay = zeros(size(N,1), size(N,2), 3);
            for c = 1:3
                overlay(:,:,c) = groupColors.(groups{g})(c);
            end
            
            % Display density overlay
            h = imagesc(xedges(1:end-1), yedges(1:end-1), overlay);
            set(h, 'AlphaData', N/max(N(:))*0.8);
        end
        
        % Add watershed boundaries on top
        scatter(llbwb(:,2),llbwb(:,1),0.5,'.','k','MarkerEdgeAlpha',0.4);
        
        axis equal off;
        colormap(ax, flipud(gray));
        caxis([0 6e-4]);
        title(sprintf('%s %s - Original', week, groups{g}));
        
        % Flipped
        subplot(length(groups), 2, plotIdx+1);
        
        % Create axes and display background
        ax = gca;
        imagesc(D);
        hold on;
        
        % Collect flipped mice data
        allPoints = [];
        for idx = groupIdx'
            z = zEmbeddings{idx + numMice};
            z_img = (z + 65) * 501 / 130;
            allPoints = [allPoints; z_img];
        end
        
        if ~isempty(allPoints)
            % Create 2D histogram for density
            [N, xedges, yedges] = histcounts2(allPoints(:,1), allPoints(:,2), ...
                linspace(1,501,60), linspace(1,501,60));
            
            % Smooth the density
            N = imgaussfilt(N', 3);
            
            % Create overlay with darker group color
            overlay = zeros(size(N,1), size(N,2), 3);
            for c = 1:3
                overlay(:,:,c) = groupColors.(groups{g})(c) * 0.6;
            end
            
            % Display density overlay
            h = imagesc(xedges(1:end-1), yedges(1:end-1), overlay);
            set(h, 'AlphaData', N/max(N(:))*0.8);
        end
        
        % Add watershed boundaries on top
        scatter(llbwb(:,2),llbwb(:,1),0.5,'.','k','MarkerEdgeAlpha',0.4);
        
        axis equal off;
        colormap(ax, flipud(gray));
        caxis([0 6e-4]);
        title(sprintf('%s %s - Flipped', week, groups{g}));
        
        plotIdx = plotIdx + 2;
    end
end

%% Create combined plot showing all groups per week with density
for w = 1:length(weeks)
    week = weeks{w};
    
    figure('Name', sprintf('All Groups %s - Original vs Flipped', week));
    
    % Get groups for this week
    if strcmp(week, 'week1')
        groups = {'DRG', 'SC'};
    else
        groups = {'DRG', 'IT', 'SC', 'SNI'};
    end
    
    % Original mice
    subplot(1,2,1);
    ax1 = gca;
    imagesc(D); 
    hold on;
    
    % Process each group to create density overlays
    legendHandles = [];
    legendLabels = {};
    
    for g = 1:length(groups)
        groupIdx = find(strcmp(mouseWeeks, week) & strcmp(mouseGroups, groups{g}));
        allPoints = [];
        for idx = groupIdx'
            z = zEmbeddings{idx};
            z_img = (z + 65) * 501 / 130;
            allPoints = [allPoints; z_img];
        end
        
        if ~isempty(allPoints)
            % Create 2D histogram for density
            [N, xedges, yedges] = histcounts2(allPoints(:,1), allPoints(:,2), ...
                linspace(1,501,80), linspace(1,501,80));
            
            % Smooth the density
            N = imgaussfilt(N', 2);
            
            % Create colored overlay for this group
            overlay = zeros(size(N,1), size(N,2), 3);
            for c = 1:3
                overlay(:,:,c) = groupColors.(groups{g})(c);
            end
            
            % Display density overlay
            h = imagesc(xedges(1:end-1), yedges(1:end-1), overlay);
            set(h, 'AlphaData', N/max(N(:))*0.6);
            
            % Create invisible scatter plot for legend
            h_legend = scatter(nan, nan, 100, groupColors.(groups{g}), 'filled', 's');
            legendHandles = [legendHandles, h_legend];
            legendLabels = [legendLabels, groups{g}];
        end
    end
    
    % Add watershed boundaries on top
    scatter(llbwb(:,2),llbwb(:,1),0.5,'.','k','MarkerEdgeAlpha',0.4);
    
    axis equal off;
    colormap(ax1, flipud(gray));
    caxis([0 6e-4]);
    title(sprintf('All Groups %s - Original', week));
    legend(legendHandles, legendLabels, 'Location', 'best');
    
    % Flipped mice
    subplot(1,2,2);
    ax2 = gca;
    imagesc(D); 
    hold on;
    
    % Process each group to create density overlays
    legendHandles = [];
    legendLabels = {};
    
    for g = 1:length(groups)
        groupIdx = find(strcmp(mouseWeeks, week) & strcmp(mouseGroups, groups{g}));
        allPoints = [];
        for idx = groupIdx'
            z = zEmbeddings{idx + numMice};
            z_img = (z + 65) * 501 / 130;
            allPoints = [allPoints; z_img];
        end
        
        if ~isempty(allPoints)
            % Create 2D histogram for density
            [N, xedges, yedges] = histcounts2(allPoints(:,1), allPoints(:,2), ...
                linspace(1,501,80), linspace(1,501,80));
            
            % Smooth the density
            N = imgaussfilt(N', 2);
            
            % Create colored overlay for this group
            overlay = zeros(size(N,1), size(N,2), 3);
            for c = 1:3
                overlay(:,:,c) = groupColors.(groups{g})(c) * 0.6;
            end
            
            % Display density overlay
            h = imagesc(xedges(1:end-1), yedges(1:end-1), overlay);
            set(h, 'AlphaData', N/max(N(:))*0.6);
            
            % Create invisible scatter plot for legend
            h_legend = scatter(nan, nan, 100, groupColors.(groups{g})*0.6, 'filled', 's');
            legendHandles = [legendHandles, h_legend];
            legendLabels = [legendLabels, [groups{g} ' - Flipped']];
        end
    end
    
    % Add watershed boundaries on top
    scatter(llbwb(:,2),llbwb(:,1),0.5,'.','k','MarkerEdgeAlpha',0.4);
    
    axis equal off;
    colormap(ax2, flipud(gray));
    caxis([0 6e-4]);
    title(sprintf('All Groups %s - Flipped', week));
    legend(legendHandles, legendLabels, 'Location', 'best');
end

fprintf('\nAnalysis complete!\n');
fprintf('Total mice analyzed: %d original + %d flipped = %d total\n', numMice, numMice, numMice*2);
fprintf('Week 1: %d mice\n', sum(strcmp(mouseWeeks, 'week1')));
fprintf('Week 2: %d mice\n', sum(strcmp(mouseWeeks, 'week2')));






