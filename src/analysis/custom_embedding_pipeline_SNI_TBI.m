%% Custom Embedding Pipeline for SNI_2 and week4-TBI_3 Training
% This script creates behavioral embeddings using SNI_2.mat and week4-TBI_3.mat
% as training data, then re-embeds all other files in the dataset
%
% Key Features:
% 1. Uses only SNI_2.mat and week4-TBI_3.mat for training PCA and t-SNE embedding
% 2. Re-embeds all other 28 files using the trained embedding space
% 3. Creates watershed regions and behavioral analysis
% 4. Includes batch correction and parameter optimization

clear; close all; clc;

fprintf('=== CUSTOM EMBEDDING PIPELINE FOR SNI_2 AND WEEK4-TBI_3 TRAINING ===\n');
fprintf('Training files: SNI_2.mat and week4-TBI_3.mat\n');
fprintf('Re-embedding: All other 28 files\n\n');

%% SECTION 1: SETUP AND CONFIGURATION
fprintf('[SECTION 1] Setup and Configuration\n');

% Define data directory
data_dir = '/work/rl349/dannce/mouse14/allData';

% Define all available files
all_files = {
    'DRG_1.mat', 'DRG_2.mat', 'DRG_3.mat', 'DRG_4.mat', 'DRG_5.mat', ...
    'IT_1.mat', 'IT_2.mat','IT_3.mat', ...
    'SC_1.mat', 'SC_2.mat', 'SC_3.mat', 'SC_4.mat', 'SC_5.mat', 'SC_6.mat', ...
    'SNI_1.mat', 'SNI_2.mat', 'SNI_3.mat', ...
    'week4-DRG_1.mat', 'week4-DRG_2.mat', 'week4-DRG_3.mat', ...
    'week4-SC_1.mat', 'week4-SC_2.mat', 'week4-SC_3.mat', ...
    'week4-SNI_1.mat', 'week4-SNI_2.mat', 'week4-SNI_3.mat', ...
    'week4-TBI_1.mat', 'week4-TBI_2.mat', 'week4-TBI_3.mat', 'week4-TBI_4.mat'
};

% Define training files (use all files for robust embedding)
training_files = all_files;

% Define re-embedding files (re-embed all files to produce outputs for every dataset)
reembedding_files = all_files;

fprintf('Total files: %d\n', length(all_files));
fprintf('Training files: %d (%s)\n', length(training_files), strjoin(training_files, ', '));
fprintf('Re-embedding files: %d\n', length(reembedding_files));

% Create metadata for file organization
metadata = create_file_metadata(all_files);

%% SECTION 2: LOAD AND PREPARE TRAINING DATA
fprintf('\n[SECTION 2] Loading Training Data\n');

% Load training data
training_data = {};
training_labels = {};
total_training_frames = 0;

for i = 1:length(training_files)
    file_path = fullfile(data_dir, training_files{i});
    fprintf('Loading training file: %s\n', training_files{i});
    
    if exist(file_path, 'file')
        data = load(file_path);
        if isfield(data, 'pred')
            training_data{i} = data.pred;
            training_labels{i} = training_files{i};
            total_training_frames = total_training_frames + size(data.pred, 1);
            fprintf('  Loaded: %s with %d frames\n', training_files{i}, size(data.pred, 1));
        else
            error('File %s does not contain pred field', training_files{i});
        end
    else
        error('Training file not found: %s', file_path);
    end
end

fprintf('Total training frames: %d\n', total_training_frames);

%% SECTION 3: PREPARE TRAINING SET (no flipping)
fprintf('\n[SECTION 3] Preparing Training Set (no flipping)\n');
training_data_combined = training_data;
training_labels_combined = training_labels;
fprintf('Training datasets: %d\n', length(training_data_combined));

%% SECTION 4: SET UP SKELETON AND PARAMETERS
fprintf('\n[SECTION 4] Setting Up Skeleton and Parameters\n');

% Skeleton for mouse14 format (14 joints)
joints_idx = [1 2; 1 3; 2 3; ...  % head connections: Snout-EarL, Snout-EarR, EarL-EarR
    1 4; 4 5; 5 6; ...            % spine: Snout-SpineF, SpineF-SpineM, SpineM-Tail(base)
    4 7; 7 8; ...                 % left front limb: SpineF-ForShdL, ForShdL-ForepawL
    4 9; 9 10; ...                % right front limb: SpineF-ForeShdR, ForeShdR-ForepawR
    5 11; 11 12; ...              % left hind limb: SpineM-HindShdL, HindShdL-HindpawL
    5 13; 13 14];                 % right hind limb: SpineM-HindShdR, HindShdR-HindpawR

% Define colors for visualization
chead = [1 .6 .2]; % orange
cspine = [.2 .635 .172]; % green
cLF = [0 0 1]; % blue
cRF = [1 0 0]; % red
cLH = [0 1 1]; % cyan
cRH = [1 0 1]; % magenta

scM = [chead; chead; chead; ...      % head connections (3)
    cspine; cspine; cspine; ...       % spine connections (3)
    cLF; cLF; ...                     % left front limb (2)
    cRF; cRF; ...                     % right front limb (2)  
    cLH; cLH; ...                     % left hind limb (2)
    cRH; cRH];                        % right hind limb (2)

skeleton.color = scM;
skeleton.joints_idx = joints_idx;

% Set up parameters
parameters = setRunParameters([]);
parameters.samplingFreq = 50;
parameters.minF = 0.5;
parameters.maxF = 20;
parameters.numModes = 20;

% PCA parameters
nPCA = 15;
pcaModes = 20;
numModes = pcaModes;

fprintf('Skeleton configured with %d joints and %d connections\n', 14, size(joints_idx, 1));
fprintf('Parameters: samplingFreq=%d, minF=%.1f, maxF=%.1f, PCA modes=%d\n', ...
    parameters.samplingFreq, parameters.minF, parameters.maxF, nPCA);

%% SECTION 5: EXTRACT FEATURES AND PERFORM PCA ON TRAINING DATA
fprintf('\n[SECTION 5] Feature Extraction and PCA on Training Data\n');

% Helper functions
returnDist3d = @(x,y) sqrt(sum((x-y).^2,2));

% Define joint pairs for distance calculations
xIdx = 1:14; yIdx = 1:14;
[Xi, Yi] = meshgrid(xIdx, yIdx);
Xi = Xi(:); Yi = Yi(:);
IDX = find(Xi ~= Yi);
nx = length(xIdx);

% Calculate characteristic length for each training dataset
lengtht = zeros(length(training_data_combined), 1);
for i = 1:length(training_data_combined)
    ma1 = training_data_combined{i};
    % Distance between keypoints 1 and 6 (snout to tail base)  
    sj = returnDist3d(squeeze(ma1(:,:,1)), squeeze(ma1(:,:,6)));  
    % Use 95th percentile as the characteristic length  
    lengtht(i) = prctile(sj, 95);
    fprintf('Characteristic length for %s: %.2f\n', training_labels_combined{i}, lengtht(i));
end

% PCA on training data only
fprintf('Performing PCA on training data...\n');
firstBatch = true;
currentImage = 0;
batchSize = 30000;
mu = zeros(1, 506);

for j = 1:length(training_data_combined)
    fprintf('Processing training dataset %d/%d (%s)\n', j, length(training_data_combined), training_labels_combined{j});
    ma1 = training_data_combined{j};
    nn1 = size(ma1,1);
    
    % Calculate pairwise distances
    p1Dist = zeros(nx^2,size(ma1,1));
    for i = 1:size(p1Dist,1)
        p1Dist(i,:) = returnDist3d(squeeze(ma1(:,:,Xi(i))),squeeze(ma1(:,:,Yi(i))));
    end
    
    % Smooth distances
    p1Dsmooth = zeros(size(p1Dist));
    for i = 1:size(p1Dist,1)
        if exist('medfilt1', 'file')
            p1Dsmooth(i,:) = smooth(medfilt1(p1Dist(i,:),3),3);
        else
            p1Dsmooth(i,:) = smooth(p1Dist(i,:),3);
        end
    end
    
    p1Dist = p1Dsmooth(IDX,:)';
    
    % Scale by characteristic length
    scaleVal = lengtht(j)./90;
    p1Dist = p1Dist.*scaleVal;
    
    % PCA computation
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
fprintf('Computing Principal Components...\n');
[vecs,vals] = eig(C); vals = flipud(diag(vals)); vecs = fliplr(vecs);
mus = mu;

% Save PCA results
save('vecsMus_SNI_TBI_training.mat','C','L','mus','vals','vecs');
fprintf('PCA complete. Saved to vecsMus_SNI_TBI_training.mat\n');

%% SECTION 6: CREATE TRAINING EMBEDDING
fprintf('\n[SECTION 6] Creating Training Embedding\n');

vecs15 = vecs(:,1:nPCA);
numPerDataSet = 320;  % Standard subsampling

% Collect raw high-D samples per file (no per-file t-SNE/templates)
mD_training_samples = cell(size(training_data_combined)); 
mA_training_samples = cell(size(training_data_combined));
train_groups = cell(size(training_data_combined));
train_weeks = cell(size(training_data_combined));

fprintf('Collecting behavioral feature samples for training (pre-ComBat)...\n');
for j = 1:length(training_data_combined)
    fprintf('Processing features %d/%d (%s)\n', j, length(training_data_combined), training_labels_combined{j});
    ma1 = training_data_combined{j};

    nn1 = size(ma1,1);
    p1Dist = zeros(nx^2,size(ma1,1));
    for i = 1:size(p1Dist,1)
        p1Dist(i,:) = returnDist3d(squeeze(ma1(:,:,Xi(i))),squeeze(ma1(:,:,Yi(i))));
    end

    p1Dsmooth = zeros(size(p1Dist));
    for i = 1:size(p1Dist,1)
        if exist('medfilt1', 'file')
            p1Dsmooth(i,:) = smooth(medfilt1(p1Dist(i,:),3),3);
        else
            p1Dsmooth(i,:) = smooth(p1Dist(i,:),3);
        end
    end
    p1Dist = p1Dsmooth(IDX,:)';
    
    % Get floor value and scale
    allz = squeeze(ma1(:,3,[12 14])); 
    zz = allz(:);
    fz = prctile(zz,10);
    sj = returnDist3d(squeeze(ma1(:,:,1)),squeeze(ma1(:,:,6)));
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
        if exist('medfilt1', 'file')
            jv(:,i) = [0; medfilt1(sqrt(sum(diff(squeeze(ma1(:,:,xIdx(i)))).^2,2)),10)];
        else
            jv(:,i) = [0; sqrt(sum(diff(squeeze(ma1(:,:,xIdx(i)))).^2,2))];
        end
    end
    jv = jv.*scaleVal;
    jv(jv>=5) = 5;
    
    % Z-coordinates
    p1z = zeros(nx,nn1);
    for i = 1:nx
        if exist('medfilt1', 'file')
            p1z(i,:) = smooth(medfilt1(squeeze(ma1(:,3,xIdx(i))),3),3);
        else
            p1z(i,:) = smooth(squeeze(ma1(:,3,xIdx(i))),3);
        end
    end
    allz1 = squeeze(ma1(:,3,[12 14])); 
    zz1 = allz1(:); 
    fz1 = prctile(zz1,10);
    floorval = fz1;
    p1z = (p1z-floorval).*scaleVal; 
    
    nnData = [data2 .25*p1z' .5*jv];
    
    % Subsample for training templates (store for global ComBat)
    sampleIdx = 1:20:size(nnData,1);
    mD_training_samples{j} = nnData(sampleIdx,:);
    mA_training_samples{j} = amps(sampleIdx,:);
    [g,w] = extract_metadata_from_filename(training_labels_combined{j});
    train_groups{j} = g; train_weeks{j} = w;
end

% Global ComBat correction on concatenated samples
fprintf('Applying ComBat batch correction on high-dimensional features (training samples)...\n');
allD_train_raw = combineCells(mD_training_samples);
allA_train = combineCells(mA_training_samples);

% Ensure ComBat is on path
if exist('combat','file')==0
    addpath('Matlab/scripts');
end

% Build batch vector (use file identity as batch proxy) and covariates (Group, Week)
allBatchLabels_points = {};
allCovarLabels_points = {};
startRow = 1;
for j = 1:length(mD_training_samples)
    nRows = size(mD_training_samples{j},1);
    % Use explicit batch rules based on filename and week/group
    batchLabelJ = infer_batch_label(training_labels_combined{j});
    blabels = repmat({batchLabelJ}, nRows, 1);
    allBatchLabels_points = [allBatchLabels_points; blabels];
    covLabel = repmat({[train_groups{j} '_' train_weeks{j}]}, nRows, 1);
    allCovarLabels_points = [allCovarLabels_points; covLabel];
    startRow = startRow + nRows;
end
[uniqueBatches,~,batchVector] = unique(allBatchLabels_points);
[uniqueCov,~,covVector] = unique(allCovarLabels_points);
% One-hot covariate matrix (Group_Week)
mod = zeros(length(covVector), length(uniqueCov));
for i = 1:length(uniqueCov)
    mod(covVector==i,i) = 1;
end

% Remove any all-zero columns (safety)
keepCols = find(any(mod~=0,1));
mod = mod(:, keepCols);

% Build batch design to check confounding
numB = length(uniqueBatches);
batchDesign = zeros(length(batchVector), numB);
for i = 1:numB
    batchDesign(batchVector==i,i) = 1;
end

% Detect confounding and reduce covariates to be independent of batch space
fprintf('Checking confounding between batch and covariates...\n');
designMatrix = [batchDesign mod];
if rank(designMatrix) < size(designMatrix,2)
    fprintf('  Confounding detected. Orthogonalizing covariates w.r.t. batch...\n');
    % Residualize mod against batchDesign (least squares)
    B = batchDesign\mod;              % solve batchDesign * B ≈ mod
    mod_res = mod - batchDesign*B;     % residual covariate space
    % QR with column pivoting to select independent columns
    [Q,R,E] = qr(mod_res,0);
    diagR = abs(diag(R));
    tol = max(size(mod_res)) * eps(norm(R, 'fro'));
    r = sum(diagR > tol);
    if r > 0
        mod = mod_res(:, E(1:r));
        fprintf('  Kept %d independent covariate column(s).\n', r);
    else
        mod = [];
        fprintf('  All covariate columns are confounded. Proceeding without covariates.\n');
    end
else
    fprintf('  No confounding detected. Using covariates as provided.\n');
end

% Call real ComBat (parametric), features x samples
X = allD_train_raw';
% Remove NaNs/Infs (replace with feature medians per feature)
if any(~isfinite(X(:)))
    featMed = nanmedian(X,2);
    bad = ~isfinite(X);
    repl = repmat(featMed,1,size(X,2));
    X(bad) = repl(bad);
end

try
    Xcorr = combat(X, batchVector, mod, 1);
    allD_train_corrected = Xcorr';
    fprintf('ComBat correction successful.\n');
catch ME
    fprintf('ComBat failed (%s). Retrying without covariates...\n', ME.message);
    Xcorr = combat(X, batchVector, [], 1);
    allD_train_corrected = Xcorr';
end

% Global t-SNE on corrected samples
fprintf('Running t-SNE on ComBat-corrected training samples (%d points) ...\n', size(allD_train_corrected,1));
yData = tsne(allD_train_corrected);

% Global template selection on corrected samples
[signalData,signalAmps] = findTemplatesFromData( allD_train_corrected, yData, allA_train, numPerDataSet, parameters);

% Use selected templates as training set
mD_training = {signalData}; 
mA_training = {signalAmps};

save('trainingSignalData_SNI_TBI.mat','mA_training','mD_training');
fprintf('Training embeddings saved to trainingSignalData_SNI_TBI.mat\n');

%% SECTION 7: CREATE t-SNE EMBEDDING ON TRAINING DATA
fprintf('\n[SECTION 7] Creating t-SNE Embedding on Training Data\n');

% Combine training embeddings (already global)
allD_training = signalData; 
allA_training = signalAmps;

fprintf('Running t-SNE on training data (%d points)...\n', size(allD_training, 1));
Y_training = tsne(allD_training);
save('train_SNI_TBI.mat','Y_training','allD_training');

fprintf('Training t-SNE complete. Saved to train_SNI_TBI.mat\n');

%% SECTION 8: CREATE WATERSHED REGIONS FROM TRAINING DATA
fprintf('\n[SECTION 8] Creating Watershed Regions from Training Data\n');

% Dynamic symmetric bounds around zero with small padding
yMin = min(Y_training, [], 1); yMax = max(Y_training, [], 1);
maxAbs = max([abs(yMin(1)), abs(yMax(1)), abs(yMin(2)), abs(yMax(2))]);
maxAbs = maxAbs * 1.05;  % 5%% padding
sigma_density = 0.8;     % slightly smaller sigma to increase region granularity

[xx, d] = findPointDensity(Y_training, sigma_density, 501, [-maxAbs maxAbs]);
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
figure('Name', 'Training Data Behavioral Density Map');
imagesc(D); 
axis equal off; 
colormap(flipud(gray)); 
caxis([0 max(D(:))*0.8]);
hold on; 
scatter(llbwb(:,2),llbwb(:,1),'.','k');
title('Training Data Behavioral Density Map (ComBat-corrected, dynamic bounds)');

save('watershed_SNI_TBI.mat', 'D', 'LL', 'LL2', 'llbwb', 'xx');
fprintf('Watershed regions saved to watershed_SNI_TBI.mat\n');

%% SECTION 9: LOAD AND RE-EMBED ALL OTHER FILES
fprintf('\n[SECTION 9] Re-embedding All Other Files\n');

% Load the trained embedding space
load('train_SNI_TBI.mat','Y_training','allD_training');
trainingSetData = allD_training; 
trainingEmbeddingZ = Y_training;

% Load all re-embedding files
reembedding_data = {};
reembedding_labels = {};
reembedding_metadata = {};

fprintf('Loading %d files for re-embedding...\n', length(reembedding_files));
for i = 1:length(reembedding_files)
    file_path = fullfile(data_dir, reembedding_files{i});
    fprintf('Loading: %s\n', reembedding_files{i});
    
    if exist(file_path, 'file')
        data = load(file_path);
        if isfield(data, 'pred')
            reembedding_data{i} = data.pred;
            reembedding_labels{i} = reembedding_files{i};
            
            % Extract metadata
            [group, week] = extract_metadata_from_filename(reembedding_files{i});
            reembedding_metadata{i} = struct('group', group, 'week', week);
            
            fprintf('  Loaded: %s (%s, %s) with %d frames\n', ...
                reembedding_files{i}, group, week, size(data.pred, 1));
        else
            warning('File %s does not contain pred field', reembedding_files{i});
        end
    else
        warning('File not found: %s', file_path);
    end
end

% Remove empty entries
valid_idx = ~cellfun(@isempty, reembedding_data);
reembedding_data = reembedding_data(valid_idx);
reembedding_labels = reembedding_labels(valid_idx);
reembedding_metadata = reembedding_metadata(valid_idx);

fprintf('Successfully loaded %d files for re-embedding\n', length(reembedding_data));

%% SECTION 10: CREATE FLIPPED VERSIONS FOR RE-EMBEDDING DATA
% Disabled (no flipping requested). Proceed with original data only.
% (Intentionally left as a no-op to avoid jointMapping errors.)

% Prepare re-embedding dataset list (already set to all_files above)

%% SECTION 11: RE-EMBED ALL FILES ONTO TRAINED SPACE
fprintf('\n[SECTION 11] Re-embedding All Files onto Trained Space\n');

zEmbeddings_all = cell(length(reembedding_data), 1);
wrFINE_all = cell(length(reembedding_data), 1);
parameters.batchSize = 10000;

fprintf('Re-embedding %d datasets...\n', length(reembedding_data));

for j = 1:length(reembedding_data)
    if mod(j, 5) == 0
        fprintf('Re-embedding dataset %d/%d (%s)\n', j, length(reembedding_data), reembedding_labels{j});
    end
    
    ma1 = reembedding_data{j};
    
    % Extract features using same pipeline as training
    nn1 = size(ma1,1);
    p1Dist = zeros(nx^2,size(ma1,1));
    for i = 1:size(p1Dist,1)
        p1Dist(i,:) = returnDist3d(squeeze(ma1(:,:,Xi(i))),squeeze(ma1(:,:,Yi(i))));
    end

    p1Dsmooth = zeros(size(p1Dist));
    for i = 1:size(p1Dist,1)
        if exist('medfilt1', 'file')
            p1Dsmooth(i,:) = smooth(medfilt1(p1Dist(i,:),3),3);
        else
            p1Dsmooth(i,:) = smooth(p1Dist(i,:),3);
        end
    end
    p1Dist = p1Dsmooth(IDX,:)';
    
    allz = squeeze(ma1(:,3,[12 14])); 
    zz = allz(:);
    fz = prctile(zz,10);
    sj = returnDist3d(squeeze(ma1(:,:,1)),squeeze(ma1(:,:,6)));
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
        if exist('medfilt1', 'file')
            jv(:,i) = [0; medfilt1(sqrt(sum(diff(squeeze(ma1(:,:,xIdx(i)))).^2,2)),10)];
        else
            jv(:,i) = [0; sqrt(sum(diff(squeeze(ma1(:,:,xIdx(i)))).^2,2))];
        end
    end
    jv = jv.*scaleVal;
    jv(jv>=5) = 5;

    p1z = zeros(nx,nn1);
    for i = 1:nx
        if exist('medfilt1', 'file')
            p1z(i,:) = smooth(medfilt1(squeeze(ma1(:,3,xIdx(i))),3),3);
        else
            p1z(i,:) = smooth(squeeze(ma1(:,3,xIdx(i))),3);
        end
    end
    allz1 = squeeze(ma1(:,3,[12 14])); 
    zz1 = allz1(:); 
    fz1 = prctile(zz1,10);
    floorval = fz1;
    p1z = (p1z-floorval).*scaleVal;

    nnData = [data2 .25*p1z' .5*jv];

    % Find embeddings using trained space
    [zValues,zCosts,zGuesses,inConvHull,meanMax,exitFlags] = ...
        findTDistributedProjections_fmin(nnData,trainingSetData,...
        trainingEmbeddingZ,[],parameters);

    z = zValues; 
    z(~inConvHull,:) = zGuesses(~inConvHull,:);
    
    % Save the embedding coordinates
    zEmbeddings_all{j} = z;
    
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
    
    wrFINE_all{j} = wr;
end

fprintf('Re-embedding complete for all %d datasets\n', length(reembedding_data));

%% SECTION 12: SAVE RESULTS AND CREATE ANALYSIS
fprintf('\n[SECTION 12] Saving Results and Creating Analysis\n');

% Save comprehensive results
results = struct();
results.training_files = training_files;
results.reembedding_files = reembedding_files;
results.reembedding_labels_all = reembedding_labels;
results.reembedding_metadata_all = reembedding_metadata;
results.zEmbeddings_all = zEmbeddings_all;
results.wrFINE_all = wrFINE_all;
results.Y_training = Y_training;
results.D = D;
results.LL = LL;
results.LL2 = LL2;
results.llbwb = llbwb;
results.parameters = parameters;
results.nPCA = nPCA;
results.skeleton = skeleton;

save('complete_embedding_results_SNI_TBI.mat', 'results', '-v7.3');
fprintf('Complete results saved to complete_embedding_results_SNI_TBI.mat\n');

%% SECTION 13: CREATE VISUALIZATION AND ANALYSIS
fprintf('\n[SECTION 13] Creating Visualizations and Analysis\n');

% Define colors for each group
groupColors = struct();
groupColors.DRG = [1 0 0];         % Red
groupColors.SC = [0 0 1];          % Blue
groupColors.IT = [0 1 0];          % Green
groupColors.SNI = [1 0.5 0];       % Orange
groupColors.TBI = [0.5 0 1];       % Purple
groupColors.week4_DRG = [0.7 0 0]; % Dark Red
groupColors.week4_SC = [0 0 0.7];  % Dark Blue
groupColors.week4_SNI = [0.7 0.3 0]; % Dark Orange
groupColors.week4_TBI = [0.3 0 0.7]; % Dark Purple

% Create group comparison plots
create_group_comparison_plots(results, groupColors);

% Create individual mouse plots
create_individual_mouse_plots(results, groupColors);

% Create temporal analysis plots
create_temporal_analysis_plots(results, groupColors);

fprintf('\nAnalysis complete! All visualizations created.\n');

%% DISPLAY SUMMARY
fprintf('\n=== PIPELINE SUMMARY ===\n');
fprintf('Training files used: %s\n', strjoin(training_files, ', '));
fprintf('Total files re-embedded: %d\n', length(reembedding_files));
fprintf('Training frames: %d\n', total_training_frames);
fprintf('PCA components: %d\n', nPCA);
fprintf('Wavelet modes: %d\n', numModes);
fprintf('Watershed regions: %d\n', max(LL(:)));

% Show group distribution
unique_groups = {};
group_counts = struct();
for i = 1:length(reembedding_metadata)
    if ~isempty(reembedding_metadata{i})
        group = reembedding_metadata{i}.group;
        if ~any(strcmp(unique_groups, group))
            unique_groups{end+1} = group;
            group_counts.(matlab.lang.makeValidName(group)) = 1;
        else
            field_name = matlab.lang.makeValidName(group);
            if isfield(group_counts, field_name)
                group_counts.(field_name) = group_counts.(field_name) + 1;
            else
                group_counts.(field_name) = 1;
            end
        end
    end
end

fprintf('\nGroup distribution in re-embedded data:\n');
for i = 1:length(unique_groups)
    field_name = matlab.lang.makeValidName(unique_groups{i});
    if isfield(group_counts, field_name)
        fprintf('  %s: %d files\n', unique_groups{i}, group_counts.(field_name));
    end
end

fprintf('\nResults saved to: complete_embedding_results_SNI_TBI.mat\n');
fprintf('All visualizations have been created and saved.\n');

%% HELPER FUNCTIONS

function metadata = create_file_metadata(file_list)
    % Create metadata structure for organizing files
    metadata = struct();
    metadata.all_files = file_list;
    metadata.groups = {};
    metadata.weeks = {};
    
    for i = 1:length(file_list)
        [group, week] = extract_metadata_from_filename(file_list{i});
        metadata.groups{i} = group;
        metadata.weeks{i} = week;
    end
end

function [group, week] = extract_metadata_from_filename(filename)
    % Extract group and week information from filename
    
    % Remove .mat extension
    name = filename(1:end-4);
    
    % Check if it's a week4 file
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

function label = infer_batch_label(filename)
    % User-specified batch rules
    % week1: DRG_1, DRG_2, IT_1..3, SC_1..3 => week1_batchA; rest => week1_batchB
    % week4: DRG/SC/SNI => week4_nonTBI; TBI_1..3 => week4_TBI_1to3; TBI_4 => week4_TBI_4
    [group, week] = extract_metadata_from_filename(filename);
    [grp, id] = extract_group_id_from_filename(filename);
    if strcmpi(week,'week4')
        if any(strcmpi(grp, {'DRG','SC','SCI','SNI'}))
            label = 'week4_nonTBI';
        elseif strcmpi(grp,'TBI')
            if ~isnan(id) && id>=1 && id<=3
                label = 'week4_TBI_1to3';
            elseif ~isnan(id) && id==4
                label = 'week4_TBI_4';
            else
                label = 'week4_TBI_other';
            end
        else
            label = 'week4_other';
        end
    else
        % default to week1 rules
        if (strcmpi(grp,'DRG') && ~isnan(id) && any(id==[1 2])) || ...
           (strcmpi(grp,'IT')  && ~isnan(id) && any(id==[1 2 3])) || ...
           (strcmpi(grp,'SC')  && ~isnan(id) && any(id==[1 2 3])) || ...
           (strcmpi(grp,'SCI') && ~isnan(id) && any(id==[1 2 3]))
            label = 'week1_batchA';
        else
            label = 'week1_batchB';
        end
    end
end

function [grp, id] = extract_group_id_from_filename(filename)
    % Returns group string and numeric id if present
    name = filename(1:end-4);
    if startsWith(name,'week4-')
        name = name(7:end);
    end
    % split by underscore
    us = find(name=='_',1);
    if isempty(us)
        grp = upper(name);
        id = NaN;
        return;
    end
    grp = upper(name(1:us-1));
    idstr = name(us+1:end);
    id = str2double(idstr);
    if isnan(id); id = NaN; end
end

function [xx, density] = findPointDensity(points, sigma, gridSize, xRange)
    % Return xx as a 1D grid vector (compatible with downstream code)
    % and a 2D density map over xx-by-xx.
    
    if size(points, 1) < 10
        xx = linspace(xRange(1), xRange(2), gridSize);
        density = zeros(gridSize, gridSize);
        return;
    end
    
    % Build 1D grids
    if numel(xRange) == 2
        xx = linspace(xRange(1), xRange(2), gridSize);
        yy = xx;
    else
        xx = linspace(xRange(1), xRange(2), gridSize);
        yy = linspace(xRange(3), xRange(4), gridSize);
    end
    
    % Mesh for density computation (not returned)
    [gridX, gridY] = meshgrid(xx, yy);
    
    % Subsample for efficiency
    if size(points, 1) > 10000
        idx = randperm(size(points, 1), 10000);
        points = points(idx, :);
    end
    
    % KDE with Gaussian kernels
    density = zeros(gridSize, gridSize);
    invTwoSigma2 = 1 / (2 * sigma^2);
    normConst = 2 * pi * sigma^2;
    for i = 1:size(points, 1)
        px = points(i, 1);
        py = points(i, 2);
        dist2 = (gridX - px).^2 + (gridY - py).^2;
        density = density + exp(-dist2 * invTwoSigma2);
    end
    density = density / (size(points, 1) * normConst);
end

function combined = combineCells(cellArray, dim)
    % Combine cell array contents
    if nargin < 2
        dim = 1;
    end
    
    if isempty(cellArray)
        combined = [];
        return;
    end
    
    % Remove empty cells
    cellArray = cellArray(~cellfun(@isempty, cellArray));
    
    if isempty(cellArray)
        combined = [];
        return;
    end
    
    if dim == 1
        combined = vertcat(cellArray{:});
    else
        combined = horzcat(cellArray{:});
    end
end

function params = setRunParameters(params)
    % Set default run parameters if not provided
    
    if isempty(params)
        params = struct();
    end
    
    if ~isfield(params, 'samplingFreq')
        params.samplingFreq = 50;  % Hz
    end
    
    if ~isfield(params, 'minF')
        params.minF = 0.5;  % Hz
    end
    
    if ~isfield(params, 'maxF')
        params.maxF = 20;  % Hz
    end
    
    if ~isfield(params, 'omega0')
        params.omega0 = 5;  % Wavelet center frequency
    end
    
    if ~isfield(params, 'numPeriods')
        params.numPeriods = 5;  % Number of wavelet periods
    end
    
    if ~isfield(params, 'batchSize')
        params.batchSize = 10000;  % For processing large datasets
    end
    
    % MotionMapper specific parameters
    if ~isfield(params, 'kdNeighbors')
        params.kdNeighbors = 5;  % Number of nearest neighbors for template matching
    end
    
    if ~isfield(params, 'templateLength')
        params.templateLength = 25;  % Template length in frames
    end
    
    if ~isfield(params, 'minTemplateLength')
        params.minTemplateLength = 10;  % Minimum template length in frames
    end
    
    % Defaults needed for t-SNE projection search
    if ~isfield(params, 'sigmaTolerance')
        params.sigmaTolerance = 1e-5;  % Tolerance for sigma search in perplexity match
    end
    
    if ~isfield(params, 'maxNeighbors')
        params.maxNeighbors = 200;  % Max neighbors for sparse probability computation
    end
    
    if ~isfield(params, 'numProcessors')
        params.numProcessors = 1;  % Number of processors for parallel processing
    end
end

function create_group_comparison_plots(results, groupColors)
    % Create comprehensive group comparison plots
    
    fprintf('Creating group comparison plots...\n');
    
    % Extract unique groups and weeks
    all_groups = {};
    all_weeks = {};
    for i = 1:length(results.reembedding_metadata_all)
        if ~isempty(results.reembedding_metadata_all{i})
            group = results.reembedding_metadata_all{i}.group;
            week = results.reembedding_metadata_all{i}.week;
            
            if ~any(strcmp(all_groups, group))
                all_groups{end+1} = group;
            end
            if ~any(strcmp(all_weeks, week))
                all_weeks{end+1} = week;
            end
        end
    end
    
    % Create overview plot
    figure('Name', 'All Groups Overview', 'Position', [100 100 1500 1000]);
    
    % Plot background density
    imagesc(results.D);
    axis equal off;
    colormap(flipud(gray));
    caxis([0 max(results.D(:))*0.8]);
    hold on;
    
    % Add watershed boundaries
    scatter(results.llbwb(:,2), results.llbwb(:,1), 0.5, '.', 'k', 'MarkerEdgeAlpha', 0.3);
    
    % Plot each group with different colors
    legend_handles = [];
    legend_labels = {};
    
    for g = 1:length(all_groups)
        group = all_groups{g};
        
        % Find indices for this group
        group_indices = [];
        for i = 1:length(results.reembedding_metadata_all)
            if ~isempty(results.reembedding_metadata_all{i}) && ...
               strcmp(results.reembedding_metadata_all{i}.group, group)
                group_indices = [group_indices, i];
            end
        end
        
        if ~isempty(group_indices)
            % Collect all points for this group
            all_points = [];
            for idx = group_indices
                z = results.zEmbeddings_all{idx};
                z_img = (z + 65) * 501 / 130;  % Transform to image coordinates
                all_points = [all_points; z_img];
            end
            
            % Create density overlay
            if size(all_points, 1) > 10
                [N, xedges, yedges] = histcounts2(all_points(:,1), all_points(:,2), ...
                    linspace(1, 501, 60), linspace(1, 501, 60));
                
                % Smooth the density
                if exist('imgaussfilt', 'file')
                    N = imgaussfilt(N', 3);
                else
                    N = conv2(N', ones(5)/25, 'same');
                end
                
                % Get color for this group
                clean_group = strrep(group, '_flip', '');
                if isfield(groupColors, clean_group)
                    color = groupColors.(clean_group);
                else
                    color = rand(1, 3);  % Random color if not defined
                end
                
                % Create colored overlay
                overlay = zeros(size(N,1), size(N,2), 3);
                for c = 1:3
                    overlay(:,:,c) = color(c);
                end
                
                % Display density overlay
                h = imagesc(xedges(1:end-1), yedges(1:end-1), overlay);
                set(h, 'AlphaData', N/max(N(:))*0.6);
                
                % Add to legend
                h_legend = scatter(nan, nan, 100, color, 'filled', 's');
                legend_handles = [legend_handles, h_legend];
                legend_labels = [legend_labels, {strrep(group, '_', ' ')}];
            end
        end
    end
    
    title('All Groups - Re-embedded using SNI\_2 + week4-TBI\_3 Training');
    legend(legend_handles, legend_labels, 'Location', 'eastoutside');
end

function create_individual_mouse_plots(results, groupColors)
    % Create individual mouse density plots
    
    fprintf('Creating individual mouse plots...\n');
    
    % Group by experimental group
    group_indices = struct();
    for i = 1:length(results.reembedding_metadata_all)
        if ~isempty(results.reembedding_metadata_all{i})
            group = results.reembedding_metadata_all{i}.group;
            clean_group = strrep(group, '_flip', '');
            
            if ~isfield(group_indices, clean_group)
                group_indices.(clean_group) = [];
            end
            group_indices.(clean_group) = [group_indices.(clean_group), i];
        end
    end
    
    % Create plots for each group
    field_names = fieldnames(group_indices);
    for g = 1:length(field_names)
        group = field_names{g};
        indices = group_indices.(group);
        
        if length(indices) >= 4  % Only create plots for groups with enough samples
            figure('Name', sprintf('%s Individual Mice', group), 'Position', [100 100 1800 1000]);
            
            n_mice = length(indices);
            n_cols = min(6, n_mice);
            n_rows = ceil(n_mice / n_cols);
            
            for m = 1:length(indices)
                idx = indices(m);
                
                subplot(n_rows, n_cols, m);
                
                % Background
                imagesc(results.D);
                hold on;
                
                % Get embedding data
                z = results.zEmbeddings_all{idx};
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
                
                % Add watershed boundaries
                scatter(results.llbwb(:,2), results.llbwb(:,1), 0.5, '.', 'k', 'MarkerEdgeAlpha', 0.3);
                
                axis equal off;
                colormap(gca, flipud(gray));
                caxis([0 max(results.D(:))*0.8]);
                
                % Create title from filename
                if ~isempty(results.reembedding_labels_all{idx})
                    mouse_name = strrep(results.reembedding_labels_all{idx}, '_', '\_');
                    title(mouse_name, 'Interpreter', 'tex', 'FontSize', 8);
                end
            end
            
            sgtitle(sprintf('%s - Individual Mouse Density Maps', group), 'FontSize', 14);
        end
    end
end

function create_temporal_analysis_plots(results, groupColors)
    % Create temporal analysis plots comparing week1 vs week4
    
    fprintf('Creating temporal analysis plots...\n');
    
    % Separate by week
    week1_indices = [];
    week4_indices = [];
    
    for i = 1:length(results.reembedding_metadata_all)
        if ~isempty(results.reembedding_metadata_all{i})
            week = results.reembedding_metadata_all{i}.week;
            if strcmp(week, 'week1')
                week1_indices = [week1_indices, i];
            elseif strcmp(week, 'week4')
                week4_indices = [week4_indices, i];
            end
        end
    end
    
    if ~isempty(week1_indices) && ~isempty(week4_indices)
        figure('Name', 'Temporal Analysis: Week1 vs Week4', 'Position', [100 100 1400 700]);
        
        % Week 1
        subplot(1, 2, 1);
        imagesc(results.D);
        hold on;
        
        % Group week1 data by experimental group
        week1_groups = struct();
        for idx = week1_indices
            if ~isempty(results.reembedding_metadata_all{idx})
                group = results.reembedding_metadata_all{idx}.group;
                clean_group = strrep(group, '_flip', '');
                
                if ~isfield(week1_groups, clean_group)
                    week1_groups.(clean_group) = [];
                end
                week1_groups.(clean_group) = [week1_groups.(clean_group), idx];
            end
        end
        
        % Plot each group
        legend_handles1 = [];
        legend_labels1 = {};
        group_names = fieldnames(week1_groups);
        
        for g = 1:length(group_names)
            group = group_names{g};
            indices = week1_groups.(group);
            
            % Collect all points
            all_points = [];
            for idx = indices
                z = results.zEmbeddings_all{idx};
                z_img = (z + 65) * 501 / 130;
                all_points = [all_points; z_img];
            end
            
            if size(all_points, 1) > 10
                [N, xedges, yedges] = histcounts2(all_points(:,1), all_points(:,2), ...
                    linspace(1, 501, 50), linspace(1, 501, 50));
                
                if exist('imgaussfilt', 'file')
                    N = imgaussfilt(N', 2);
                else
                    N = conv2(N', ones(3)/9, 'same');
                end
                
                % Get color
                if isfield(groupColors, group)
                    color = groupColors.(group);
                else
                    color = rand(1, 3);
                end
                
                % Create overlay
                overlay = zeros(size(N,1), size(N,2), 3);
                for c = 1:3
                    overlay(:,:,c) = color(c);
                end
                
                if max(N(:)) > 0
                    h = imagesc(xedges(1:end-1), yedges(1:end-1), overlay);
                    set(h, 'AlphaData', N/max(N(:))*0.6);
                end
                
                % Add to legend
                h_legend = scatter(nan, nan, 100, color, 'filled', 's');
                legend_handles1 = [legend_handles1, h_legend];
                legend_labels1 = [legend_labels1, {group}];
            end
        end
        
        scatter(results.llbwb(:,2), results.llbwb(:,1), 0.5, '.', 'k', 'MarkerEdgeAlpha', 0.3);
        axis equal off;
        colormap(gca, flipud(gray));
        caxis([0 max(results.D(:))*0.8]);
        title('Week 1');
        legend(legend_handles1, legend_labels1, 'Location', 'best');
        
        % Week 4
        subplot(1, 2, 2);
        imagesc(results.D);
        hold on;
        
        % Group week4 data
        week4_groups = struct();
        for idx = week4_indices
            if ~isempty(results.reembedding_metadata_all{idx})
                group = results.reembedding_metadata_all{idx}.group;
                clean_group = strrep(group, '_flip', '');
                
                if ~isfield(week4_groups, clean_group)
                    week4_groups.(clean_group) = [];
                end
                week4_groups.(clean_group) = [week4_groups.(clean_group), idx];
            end
        end
        
        % Plot each group
        legend_handles2 = [];
        legend_labels2 = {};
        group_names = fieldnames(week4_groups);
        
        for g = 1:length(group_names)
            group = group_names{g};
            indices = week4_groups.(group);
            
            % Collect all points
            all_points = [];
            for idx = indices
                z = results.zEmbeddings_all{idx};
                z_img = (z + 65) * 501 / 130;
                all_points = [all_points; z_img];
            end
            
            if size(all_points, 1) > 10
                [N, xedges, yedges] = histcounts2(all_points(:,1), all_points(:,2), ...
                    linspace(1, 501, 50), linspace(1, 501, 50));
                
                if exist('imgaussfilt', 'file')
                    N = imgaussfilt(N', 2);
                else
                    N = conv2(N', ones(3)/9, 'same');
                end
                
                % Get color (use week4 color scheme if available)
                week4_group_name = ['week4_' group];
                if isfield(groupColors, week4_group_name)
                    color = groupColors.(week4_group_name);
                elseif isfield(groupColors, group)
                    color = groupColors.(group) * 0.7;  % Darker version
                else
                    color = rand(1, 3);
                end
                
                % Create overlay
                overlay = zeros(size(N,1), size(N,2), 3);
                for c = 1:3
                    overlay(:,:,c) = color(c);
                end
                
                if max(N(:)) > 0
                    h = imagesc(xedges(1:end-1), yedges(1:end-1), overlay);
                    set(h, 'AlphaData', N/max(N(:))*0.6);
                end
                
                % Add to legend
                h_legend = scatter(nan, nan, 100, color, 'filled', 's');
                legend_handles2 = [legend_handles2, h_legend];
                legend_labels2 = [legend_labels2, {group}];
            end
        end
        
        scatter(results.llbwb(:,2), results.llbwb(:,1), 0.5, '.', 'k', 'MarkerEdgeAlpha', 0.3);
        axis equal off;
        colormap(gca, flipud(gray));
        caxis([0 max(results.D(:))*0.8]);
        title('Week 4');
        legend(legend_handles2, legend_labels2, 'Location', 'best');
        
        sgtitle('Temporal Analysis: Week1 vs Week4 (Re-embedded using SNI\_2 + week4-TBI\_3)', 'FontSize', 14);
    end
end
