function Y = planar_truss_opensees_uqlab(X)
%PLANAR_TRUSS_OPENSEES_UQLAB Evaluate physical truss inputs in OpenSeesPy.
%   X contains A1, A2, E1, E2, P1, ..., P6 in its ten columns. Y is the
%   conventional limit-state response 0.12 - abs(midspan displacement).

    arguments
        X (:,10) double {mustBeFinite, mustBePositive}
    end

    thisFile = mfilename('fullpath');
    uqlabDir = fileparts(thisFile);
    planarTrussDir = fileparts(uqlabDir);
    studiesDir = fileparts(planarTrussDir);
    repoRoot = fileparts(studiesDir);

    pythonExecutable = strtrim(getenv('ABSVR_PYTHON'));
    if isempty(pythonExecutable)
        if ispc
            pythonExecutable = fullfile(repoRoot, '.venv', 'Scripts', 'python.exe');
        else
            pythonExecutable = fullfile(repoRoot, '.venv', 'bin', 'python');
        end
    end
    if ~isfile(pythonExecutable)
        error('ABSVR:PythonNotFound', ...
            ['Python was not found at "%s". Set ABSVR_PYTHON to the ' ...
             'interpreter containing OpenSeesPy.'], pythonExecutable);
    end

    inputFile = [tempname, '.csv'];
    outputFile = [tempname, '.csv'];
    cleanupFiles = onCleanup(@() deleteIfPresent(inputFile, outputFile));
    writematrix(X, inputFile, 'Delimiter', ',');

    ledgerPath = strtrim(getenv('ABSVR_UQLAB_CALL_LEDGER'));
    ledgerArgument = '';
    if ~isempty(ledgerPath)
        ledgerArgument = sprintf(' --ledger "%s"', ledgerPath);
    end
    command = sprintf( ...
        '"%s" -m studies.planar_truss.opensees_batch_bridge --input "%s" --output "%s"%s', ...
        pythonExecutable, inputFile, outputFile, ledgerArgument);
    oldDirectory = pwd;
    restoreDirectory = onCleanup(@() cd(oldDirectory));
    cd(repoRoot);
    [status, commandOutput] = system(command);
    if status ~= 0
        error('ABSVR:OpenSeesBridgeFailed', ...
            'OpenSeesPy bridge failed (status %d):\n%s', status, commandOutput);
    end

    Y = readmatrix(outputFile);
    Y = Y(:);
    if numel(Y) ~= size(X, 1) || any(~isfinite(Y))
        error('ABSVR:InvalidBridgeOutput', ...
            'Expected %d finite OpenSees responses, received %d.', ...
            size(X, 1), numel(Y));
    end
end


function deleteIfPresent(varargin)
    for index = 1:nargin
        if isfile(varargin{index})
            delete(varargin{index});
        end
    end
end
