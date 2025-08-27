function [skeleton, parameters] = setup_skeleton_and_parameters(config)
% SETUP_SKELETON_AND_PARAMETERS - Initialize skeleton and algorithm parameters
%
% Sets up the mouse skeleton structure and algorithm parameters from config
%
% Inputs:
%   config - Configuration structure containing skeleton and parameter settings
%
% Outputs:
%   skeleton   - Skeleton structure with joints and colors
%   parameters - Algorithm parameters structure

% Setup skeleton from config
skeleton = struct();
skeleton.joints_idx = config.skeleton.joints_idx;
skeleton.color = config.skeleton.color;

% Setup parameters using SocialMapper function
parameters = setRunParameters([]);

% Override with config values
parameters.samplingFreq = config.parameters.samplingFreq;
parameters.minF = config.parameters.minF;
parameters.maxF = config.parameters.maxF;
parameters.numModes = config.parameters.numModes;

% Add batch size for processing
if isfield(config.parameters, 'reembedding') && isfield(config.parameters.reembedding, 'batchSize')
    parameters.batchSize = config.parameters.reembedding.batchSize;
else
    parameters.batchSize = 10000;  % Default
end

% Add additional parameters needed for the pipeline
if ~isfield(parameters, 'sigmaTolerance')
    parameters.sigmaTolerance = 1e-5;
end

if ~isfield(parameters, 'maxNeighbors')
    parameters.maxNeighbors = 200;
end

if ~isfield(parameters, 'kdNeighbors')
    parameters.kdNeighbors = 5;
end

if ~isfield(parameters, 'templateLength')
    parameters.templateLength = 25;
end

if ~isfield(parameters, 'minTemplateLength')
    parameters.minTemplateLength = 10;
end

fprintf('Skeleton configured with %d joints and %d connections\n', ...
    14, size(skeleton.joints_idx, 1));
fprintf('Parameters: samplingFreq=%d, minF=%.1f, maxF=%.1f\n', ...
    parameters.samplingFreq, parameters.minF, parameters.maxF);

end