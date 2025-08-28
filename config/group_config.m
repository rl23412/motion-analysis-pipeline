function config = group_config()
%% Group Configuration for Spontaneous Pain Analysis Pipeline
% This file defines experimental groups, data files, and group-specific parameters
% Modify this file to add new groups or change experimental parameters

%% Experimental Groups Configuration
config = struct();

% Version and metadata
config.version = '1.0';
config.description = 'Spontaneous pain behavioral analysis configuration';
config.created = datestr(now);

%% Group Definitions
% Define experimental groups with their associated data files and parameters

% Week 1 Groups
config.groups.week1.DRG = struct();
config.groups.week1.DRG.name = 'DRG';
config.groups.week1.DRG.description = 'Dorsal Root Ganglion - Week 1';
config.groups.week1.DRG.files = {
    'DRG_1.mat', 'DRG_2.mat', 'DRG_3.mat', 'DRG_4.mat', 'DRG_5.mat'
};
config.groups.week1.DRG.color = [1 0 0];  % Red
config.groups.week1.DRG.week = 'week1';
config.groups.week1.DRG.indices = [1, 2, 3, 4, 5];

config.groups.week1.SC = struct();
config.groups.week1.SC.name = 'SC';
config.groups.week1.SC.description = 'Spinal Cord - Week 1'; 
config.groups.week1.SC.files = {
    'SC_1.mat', 'SC_2.mat', 'SC_3.mat', 'SC_4.mat', 'SC_5.mat', 'SC_6.mat'
};
config.groups.week1.SC.color = [0 0 1];  % Blue
config.groups.week1.SC.week = 'week1';
config.groups.week1.SC.indices = [9, 10, 11, 12, 13, 14];

% Week 2 Groups  
config.groups.week2.DRG = struct();
config.groups.week2.DRG.name = 'DRG';
config.groups.week2.DRG.description = 'Dorsal Root Ganglion - Week 2';
config.groups.week2.DRG.files = {
    'week4-DRG_1.mat', 'week4-DRG_2.mat', 'week4-DRG_3.mat'
};
config.groups.week2.DRG.color = [1 0.5 0.5];  % Light Red
config.groups.week2.DRG.week = 'week2';
config.groups.week2.DRG.indices = [18, 19, 20];

config.groups.week2.IT = struct();
config.groups.week2.IT.name = 'IT';
config.groups.week2.IT.description = 'Intrathecal - Week 2';
config.groups.week2.IT.files = {
    'IT_1.mat', 'IT_2.mat'
};
config.groups.week2.IT.color = [0 1 0];  % Green
config.groups.week2.IT.week = 'week2';
config.groups.week2.IT.indices = [6, 7];

config.groups.week2.SC = struct();
config.groups.week2.SC.name = 'SC';
config.groups.week2.SC.description = 'Spinal Cord - Week 2';
config.groups.week2.SC.files = {
    'week4-SC_1.mat', 'week4-SC_2.mat', 'week4-SC_3.mat'
};
config.groups.week2.SC.color = [0.5 0.5 1];  % Light Blue
config.groups.week2.SC.week = 'week2';
config.groups.week2.SC.indices = [21, 22, 23];

config.groups.week2.SNI = struct();
config.groups.week2.SNI.name = 'SNI';
config.groups.week2.SNI.description = 'Spinal Nerve Injury - Week 2';
config.groups.week2.SNI.files = {
    'SNI_1.mat', 'SNI_3.mat'  % SNI_2.mat used for training
};
config.groups.week2.SNI.color = [1 0.5 0];  % Orange
config.groups.week2.SNI.week = 'week2';
config.groups.week2.SNI.indices = [15, 17];

config.groups.week2.TBI = struct();
config.groups.week2.TBI.name = 'TBI';
config.groups.week2.TBI.description = 'Traumatic Brain Injury - Week 2';
config.groups.week2.TBI.files = {
    'week4-TBI_1.mat', 'week4-TBI_2.mat', 'week4-TBI_4.mat'  % TBI_3.mat used for training
};
config.groups.week2.TBI.color = [1 0 1];  % Magenta
config.groups.week2.TBI.week = 'week2';
config.groups.week2.TBI.indices = [27, 28, 30];

%% Training Data Configuration
config.training = struct();
config.training.files = {'SNI_2.mat', 'week4-TBI_3.mat'};
config.training.description = 'Files used for training the behavioral embedding model';

%% Pipeline Parameters
config.parameters = struct();

% PCA parameters
config.parameters.pca = struct();
config.parameters.pca.numComponents = 15;  % Number of PCA components to use
config.parameters.pca.batchSize = 30000;   % Batch size for PCA computation

% Wavelet parameters  
config.parameters.wavelets = struct();
config.parameters.wavelets.minF = 0.5;    % Minimum frequency
config.parameters.wavelets.maxF = 20;     % Maximum frequency
config.parameters.wavelets.pcaModes = 20; % Number of PCA modes
config.parameters.wavelets.samplingFreq = 50; % Sampling frequency (Hz)

% t-SNE parameters
config.parameters.tsne = struct();
config.parameters.tsne.numPerDataSet = 320; % Templates per dataset

% Watershed parameters
config.parameters.watershed = struct();
config.parameters.watershed.connectivity = 18;       % Watershed connectivity
config.parameters.watershed.densityRange = [-65 65]; % Density computation range
config.parameters.watershed.gridSize = 501;          % Grid size for density computation

%% Data Format Parameters
config.dataFormat = struct();
config.dataFormat.jointFormat = 'rat23';  % 'mouse14' or 'rat23'
config.dataFormat.expectedDimensions = [NaN, 3, 23]; % [frames, coords, joints]
config.dataFormat.coordinateOrder = {'x', 'y', 'z'};

% Joint mapping for rat23 format
config.dataFormat.jointNames = {
    'Snout', 'EarL', 'EarR', 'Spine1', 'Spine2', 'Spine3', 'TailBase', ...
    'ArmL1', 'ArmL2', 'ArmL3', 'ArmL4', 'ArmR1', 'ArmR2', 'ArmR3', 'ArmR4', ...
    'LegL1', 'LegL2', 'LegL3', 'LegL4', 'LegR1', 'LegR2', 'LegR3', 'LegR4'
};

% Skeleton connections for visualization
config.dataFormat.skeleton.joints_idx = [
    1 2; 1 3; 2 3; ...                    % head connections
    1 4; 4 5; 5 6; 6 7; ...               % spine
    4 8; 8 9; 9 10; 10 11; ...            % left arm
    4 12; 12 13; 13 14; 14 15; ...        % right arm
    6 16; 16 17; 17 18; 18 19; ...        % left leg  
    6 20; 20 21; 21 22; 22 23             % right leg
];

% Colors for skeleton visualization
config.dataFormat.skeleton.colors = struct();
config.dataFormat.skeleton.colors.head = [1 0.6 0.2];   % Orange
config.dataFormat.skeleton.colors.spine = [0.2 0.635 0.172]; % Green
config.dataFormat.skeleton.colors.leftFront = [0 0 1];   % Blue
config.dataFormat.skeleton.colors.rightFront = [1 0 0];  % Red
config.dataFormat.skeleton.colors.leftHind = [0 1 1];    % Cyan
config.dataFormat.skeleton.colors.rightHind = [1 0 1];   % Magenta

%% Output Configuration
config.output = struct();
config.output.baseDir = 'outputs';
config.output.saveIntermediate = true;  % Save intermediate results
config.output.generateFigures = true;   % Generate visualization figures
config.output.exportCSV = true;         % Export CSV files
config.output.compressionLevel = 6;     % MAT file compression level

%% Validation Parameters
config.validation = struct();
config.validation.checkDataIntegrity = true;
config.validation.validateResults = true;
config.validation.generateReports = true;

end

%% Helper Functions for Group Management

function groupList = get_all_groups(config)
%GET_ALL_GROUPS Get list of all defined groups
    groupList = {};
    weeks = fieldnames(config.groups);
    for w = 1:length(weeks)
        groupNames = fieldnames(config.groups.(weeks{w}));
        for g = 1:length(groupNames)
            groupInfo = config.groups.(weeks{w}).(groupNames{g});
            groupList{end+1} = struct('week', weeks{w}, 'name', groupNames{g}, 'info', groupInfo);
        end
    end
end

function config = add_group(config, week, groupName, files, color, description)
%ADD_GROUP Add a new experimental group
%   config = add_group(config, week, groupName, files, color, description)
%
% Inputs:
%   config - Current configuration structure
%   week - Week identifier ('week1', 'week2', etc.)
%   groupName - Name of the group (e.g., 'Control', 'Treatment')
%   files - Cell array of data files for this group
%   color - RGB color for visualization [R G B]
%   description - Text description of the group

    if nargin < 6
        description = sprintf('%s - %s', groupName, week);
    end
    
    % Create group structure
    newGroup = struct();
    newGroup.name = groupName;
    newGroup.description = description;
    newGroup.files = files;
    newGroup.color = color;
    newGroup.week = week;
    
    % Assign indices (this should be updated based on actual file order)
    if ~isfield(config.groups, week)
        config.groups.(week) = struct();
    end
    
    % Add to configuration
    config.groups.(week).(groupName) = newGroup;
    
    fprintf('Added group: %s/%s with %d files\n', week, groupName, length(files));
end

function config = remove_group(config, week, groupName)
%REMOVE_GROUP Remove an experimental group
    if isfield(config.groups, week) && isfield(config.groups.(week), groupName)
        config.groups.(week) = rmfield(config.groups.(week), groupName);
        fprintf('Removed group: %s/%s\n', week, groupName);
    else
        warning('Group %s/%s not found', week, groupName);
    end
end

function validate_config(config)
%VALIDATE_CONFIG Validate the configuration structure
    fprintf('Validating configuration...\n');
    
    % Check required fields
    requiredFields = {'groups', 'training', 'parameters', 'dataFormat'};
    for i = 1:length(requiredFields)
        if ~isfield(config, requiredFields{i})
            error('Missing required configuration field: %s', requiredFields{i});
        end
    end
    
    % Validate groups
    weeks = fieldnames(config.groups);
    totalFiles = 0;
    for w = 1:length(weeks)
        groupNames = fieldnames(config.groups.(weeks{w}));
        for g = 1:length(groupNames)
            group = config.groups.(weeks{w}).(groupNames{g});
            totalFiles = totalFiles + length(group.files);
            
            % Check color format
            if length(group.color) ~= 3 || any(group.color < 0) || any(group.color > 1)
                warning('Invalid color for group %s/%s', weeks{w}, groupNames{g});
            end
        end
    end
    
    fprintf('Configuration validated: %d groups, %d total files\n', ...
        sum(cellfun(@(w) length(fieldnames(config.groups.(w))), weeks)), totalFiles);
end