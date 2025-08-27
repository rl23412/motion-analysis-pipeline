function test_results = test_git_functionality()
%% TEST_GIT_FUNCTIONALITY Comprehensive Git repository functionality tests
% This script tests all Git functionality and repository integrity

fprintf('\n=== Git Functionality Test Suite ===\n\n');

test_results = struct();
test_results.timestamp = datestr(now);
test_results.tests = {};
test_results.passed = 0;
test_results.failed = 0;
test_results.total = 0;

%% Change to project directory
project_dir = fileparts(fileparts(mfilename('fullpath')));
original_dir = pwd;
cd(project_dir);

fprintf('Testing Git repository at: %s\n\n', project_dir);

try
    %% Test 1: Git Repository Initialization
    test_name = 'Git Repository Initialization';
    fprintf('Running test: %s\n', test_name);
    
    try
        [status, result] = system('git rev-parse --git-dir');
        if status == 0
            fprintf('  ✓ Git repository properly initialized\n');
            fprintf('    Git directory: %s\n', strtrim(result));
            test_result = true;
        else
            fprintf('  ✗ Git repository not found\n');
            test_result = false;
        end
    catch ME
        fprintf('  ✗ Error checking Git repository: %s\n', ME.message);
        test_result = false;
    end
    
    test_results = add_test_result(test_results, test_name, test_result, 'Check if Git repository is properly initialized');
    
    %% Test 2: Git Configuration
    test_name = 'Git Configuration';
    fprintf('\nRunning test: %s\n', test_name);
    
    try
        [status1, user_name] = system('git config user.name');
        [status2, user_email] = system('git config user.email');
        
        if status1 == 0 && status2 == 0 && ~isempty(strtrim(user_name)) && ~isempty(strtrim(user_email))
            fprintf('  ✓ Git user configuration found\n');
            fprintf('    User name: %s', user_name);
            fprintf('    User email: %s', user_email);
            test_result = true;
        else
            fprintf('  ✗ Git user configuration missing\n');
            test_result = false;
        end
    catch ME
        fprintf('  ✗ Error checking Git configuration: %s\n', ME.message);
        test_result = false;
    end
    
    test_results = add_test_result(test_results, test_name, test_result, 'Check Git user configuration');
    
    %% Test 3: Repository Status
    test_name = 'Repository Status';
    fprintf('\nRunning test: %s\n', test_name);
    
    try
        [status, result] = system('git status --porcelain');
        if status == 0
            fprintf('  ✓ Git status command successful\n');
            if isempty(strtrim(result))
                fprintf('    Repository is clean (no uncommitted changes)\n');
            else
                fprintf('    Repository has uncommitted changes:\n');
                fprintf('    %s\n', result);
            end
            test_result = true;
        else
            fprintf('  ✗ Git status command failed\n');
            test_result = false;
        end
    catch ME
        fprintf('  ✗ Error checking Git status: %s\n', ME.message);
        test_result = false;
    end
    
    test_results = add_test_result(test_results, test_name, test_result, 'Check repository status');
    
    %% Test 4: Commit History
    test_name = 'Commit History';
    fprintf('\nRunning test: %s\n', test_name);
    
    try
        [status, result] = system('git log --oneline -n 5');
        if status == 0 && ~isempty(strtrim(result))
            fprintf('  ✓ Commit history exists\n');
            fprintf('    Recent commits:\n');
            commits = strsplit(strtrim(result), '\n');
            for i = 1:min(3, length(commits))
                fprintf('    %s\n', commits{i});
            end
            test_result = true;
        else
            fprintf('  ✗ No commit history found\n');
            test_result = false;
        end
    catch ME
        fprintf('  ✗ Error checking commit history: %s\n', ME.message);
        test_result = false;
    end
    
    test_results = add_test_result(test_results, test_name, test_result, 'Check commit history exists');
    
    %% Test 5: Branch Information
    test_name = 'Branch Information';
    fprintf('\nRunning test: %s\n', test_name);
    
    try
        [status, result] = system('git branch');
        if status == 0
            fprintf('  ✓ Branch information available\n');
            fprintf('    Current branch: %s\n', strtrim(result));
            test_result = true;
        else
            fprintf('  ✗ Could not get branch information\n');
            test_result = false;
        end
    catch ME
        fprintf('  ✗ Error checking branch information: %s\n', ME.message);
        test_result = false;
    end
    
    test_results = add_test_result(test_results, test_name, test_result, 'Check branch information');
    
    %% Test 6: .gitignore Functionality
    test_name = 'Gitignore Functionality';
    fprintf('\nRunning test: %s\n', test_name);
    
    try
        % Create a test file that should be ignored
        test_file = 'test_ignored_file.log';
        fid = fopen(test_file, 'w');
        if fid ~= -1
            fprintf(fid, 'This is a test file that should be ignored by Git');
            fclose(fid);
            
            % Check if file is ignored
            [status, result] = system('git status --porcelain');
            if status == 0 && ~contains(result, test_file)
                fprintf('  ✓ .gitignore is working correctly\n');
                fprintf('    Test file %s is properly ignored\n', test_file);
                test_result = true;
            else
                fprintf('  ! .gitignore may not be working properly\n');
                fprintf('    Test file %s appears in git status\n', test_file);
                test_result = false;
            end
            
            % Clean up test file
            delete(test_file);
        else
            fprintf('  ✗ Could not create test file for .gitignore test\n');
            test_result = false;
        end
    catch ME
        fprintf('  ✗ Error testing .gitignore: %s\n', ME.message);
        test_result = false;
    end
    
    test_results = add_test_result(test_results, test_name, test_result, 'Test .gitignore functionality');
    
    %% Test 7: File Tracking
    test_name = 'File Tracking';
    fprintf('\nRunning test: %s\n', test_name);
    
    try
        [status, result] = system('git ls-files | wc -l');
        if status == 0
            num_tracked = str2double(strtrim(result));
            if num_tracked > 0
                fprintf('  ✓ Files are being tracked by Git\n');
                fprintf('    Number of tracked files: %d\n', num_tracked);
                
                % Show some tracked files
                [~, tracked_files] = system('git ls-files');
                files = strsplit(strtrim(tracked_files), '\n');
                fprintf('    Sample tracked files:\n');
                for i = 1:min(5, length(files))
                    fprintf('      %s\n', files{i});
                end
                test_result = true;
            else
                fprintf('  ✗ No files are being tracked by Git\n');
                test_result = false;
            end
        else
            fprintf('  ✗ Could not check tracked files\n');
            test_result = false;
        end
    catch ME
        fprintf('  ✗ Error checking file tracking: %s\n', ME.message);
        test_result = false;
    end
    
    test_results = add_test_result(test_results, test_name, test_result, 'Check file tracking');
    
    %% Test 8: Test Git Operations (Add, Commit)
    test_name = 'Git Operations Test';
    fprintf('\nRunning test: %s\n', test_name);
    
    try
        % Create a test file
        test_file = 'git_test_file.txt';
        fid = fopen(test_file, 'w');
        if fid ~= -1
            fprintf(fid, 'Git functionality test file\nCreated: %s\n', datestr(now));
            fclose(fid);
            
            % Add the file
            [status1, ~] = system(['git add ' test_file]);
            
            % Check if file is staged
            [status2, result] = system('git status --porcelain');
            is_staged = contains(result, test_file) && contains(result, 'A ');
            
            if status1 == 0 && is_staged
                fprintf('  ✓ Git add operation successful\n');
                
                % Test commit
                commit_msg = sprintf('Test commit from automated test suite - %s', datestr(now));
                [status3, ~] = system(['git commit -m "' commit_msg '"']);
                
                if status3 == 0
                    fprintf('  ✓ Git commit operation successful\n');
                    test_result = true;
                    
                    % Clean up - remove the test file and commit the removal
                    delete(test_file);
                    system(['git add ' test_file]);
                    system('git commit -m "Clean up test file from automated test"');
                else
                    fprintf('  ✗ Git commit operation failed\n');
                    test_result = false;
                end
            else
                fprintf('  ✗ Git add operation failed\n');
                test_result = false;
            end
        else
            fprintf('  ✗ Could not create test file\n');
            test_result = false;
        end
    catch ME
        fprintf('  ✗ Error testing Git operations: %s\n', ME.message);
        test_result = false;
    end
    
    test_results = add_test_result(test_results, test_name, test_result, 'Test basic Git operations (add, commit)');
    
    %% Test 9: Repository Integrity
    test_name = 'Repository Integrity';
    fprintf('\nRunning test: %s\n', test_name);
    
    try
        [status, result] = system('git fsck --full');
        if status == 0 && ~contains(lower(result), 'error')
            fprintf('  ✓ Repository integrity check passed\n');
            if isempty(strtrim(result))
                fprintf('    No issues found\n');
            else
                fprintf('    Fsck output: %s\n', strtrim(result));
            end
            test_result = true;
        else
            fprintf('  ✗ Repository integrity issues found\n');
            fprintf('    %s\n', result);
            test_result = false;
        end
    catch ME
        fprintf('  ✗ Error checking repository integrity: %s\n', ME.message);
        test_result = false;
    end
    
    test_results = add_test_result(test_results, test_name, test_result, 'Check repository integrity');
    
    %% Test 10: Remote Repository Readiness
    test_name = 'Remote Repository Readiness';
    fprintf('\nRunning test: %s\n', test_name);
    
    try
        [status, result] = system('git remote -v');
        
        % Repository should be ready for remote even if no remote is configured yet
        fprintf('  ✓ Repository is ready for remote configuration\n');
        if status == 0 && ~isempty(strtrim(result))
            fprintf('    Configured remotes:\n%s\n', result);
        else
            fprintf('    No remotes configured (this is normal for new repositories)\n');
        end
        
        % Test that we can show the current branch (needed for push)
        [status2, branch] = system('git symbolic-ref --short HEAD');
        if status2 == 0
            fprintf('    Current branch: %s', branch);
            fprintf('    Repository is ready for GitHub/remote setup\n');
            test_result = true;
        else
            fprintf('    Could not determine current branch\n');
            test_result = false;
        end
    catch ME
        fprintf('  ✗ Error checking remote readiness: %s\n', ME.message);
        test_result = false;
    end
    
    test_results = add_test_result(test_results, test_name, test_result, 'Check readiness for remote repository setup');
    
finally
    % Return to original directory
    cd(original_dir);
end

%% Test Summary
fprintf('\n=== Git Test Summary ===\n');
fprintf('Total tests: %d\n', test_results.total);
fprintf('Passed: %d\n', test_results.passed);
fprintf('Failed: %d\n', test_results.failed);
fprintf('Success rate: %.1f%%\n', 100 * test_results.passed / test_results.total);

test_results.success_rate = test_results.passed / test_results.total;
test_results.overall_success = test_results.failed == 0;

if test_results.overall_success
    fprintf('\n✅ All Git functionality tests PASSED!\n');
    fprintf('Repository is fully functional and ready for use.\n');
else
    fprintf('\n❌ Some Git functionality tests FAILED.\n');
    fprintf('Review the failed tests above and resolve issues.\n');
end

% Save test results
results_file = fullfile(project_dir, 'tests', 'git_test_results.mat');
try
    save(results_file, 'test_results');
    fprintf('\nTest results saved to: %s\n', results_file);
catch
    fprintf('\nWarning: Could not save test results to file\n');
end

fprintf('\n=== Git Functionality Test Complete ===\n\n');

end

%% Helper function to add test results
function test_results = add_test_result(test_results, test_name, passed, description)
    test_info = struct();
    test_info.name = test_name;
    test_info.passed = passed;
    test_info.description = description;
    test_info.timestamp = datestr(now);
    
    test_results.tests{end+1} = test_info;
    test_results.total = test_results.total + 1;
    
    if passed
        test_results.passed = test_results.passed + 1;
    else
        test_results.failed = test_results.failed + 1;
    end
end