function [training_data, training_labels] = load_training_data(data_dir, training_files, verbose)
% LOAD_TRAINING_DATA - Load training data from .mat files
%
% Loads pose estimation data from specified training files
%
% Inputs:
%   data_dir        - Directory containing .mat files
%   training_files  - Cell array of filenames to load
%   verbose         - Display progress messages (default: true)
%
% Outputs:
%   training_data   - Cell array of pose data matrices
%   training_labels - Cell array of corresponding filenames

if nargin < 3
    verbose = true;
end

training_data = {};
training_labels = {};
total_training_frames = 0;

if verbose
    fprintf('Loading training data from %d files...\n', length(training_files));
end

for i = 1:length(training_files)
    file_path = fullfile(data_dir, training_files{i});
    
    if verbose && mod(i, 5) == 0
        fprintf('Loading training file %d/%d: %s\n', i, length(training_files), training_files{i});
    end
    
    if exist(file_path, 'file')
        try
            data = load(file_path);
            if isfield(data, 'pred')
                training_data{i} = data.pred;
                training_labels{i} = training_files{i};
                total_training_frames = total_training_frames + size(data.pred, 1);
                
                if verbose && i <= 10  % Show details for first 10 files
                    fprintf('  Loaded: %s with %d frames\n', training_files{i}, size(data.pred, 1));
                end
            else
                warning('File %s does not contain pred field', training_files{i});
            end
        catch ME
            warning('Failed to load file %s: %s', training_files{i}, ME.message);
        end
    else
        warning('Training file not found: %s', file_path);
    end
end

% Remove empty entries
valid_idx = ~cellfun(@isempty, training_data);
training_data = training_data(valid_idx);
training_labels = training_labels(valid_idx);

if verbose
    fprintf('Successfully loaded %d training files\n', length(training_data));
    fprintf('Total training frames: %d\n', total_training_frames);
end

if isempty(training_data)
    error('No training data could be loaded');
end

end