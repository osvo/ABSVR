function summary = run_uqlab_abpce(seed, outputFile, runOptions)
%RUN_UQLAB_ABPCE Reproduce the published active bootstrap-PCE algorithm.
%
% The active-learning controller implements Eqs. (8)--(11) in Marelli and
% Sudret (2018). PCE calibration, LARS basis selection, and fast bootstrap
% replications are provided directly by UQLab.

    arguments
        seed (1,1) double {mustBeInteger, mustBeNonnegative} = 11
        outputFile (1,1) string = ""
        runOptions.InitialDesignSize (1,1) double {mustBeInteger,mustBePositive} = 40
        runOptions.InternalSampleSize (1,1) double {mustBeInteger,mustBePositive} = 1e6
        runOptions.BootstrapReplications (1,1) double {mustBeInteger,mustBePositive} = 100
        runOptions.ChunkSize (1,1) double {mustBeInteger,mustBePositive} = 20000
        runOptions.ConvergenceTolerance (1,1) double {mustBePositive} = 0.15
        runOptions.ConsecutiveConvergedIterations (1,1) double {mustBeInteger,mustBePositive} = 2
        runOptions.MaximumTotalEvaluations (1,1) double {mustBeInteger,mustBePositive} = 300
        runOptions.Degree (1,:) double {mustBeInteger,mustBePositive} = 1:10
        runOptions.QNorm (1,1) double {mustBePositive} = 0.75
        runOptions.MaximumInteraction (1,1) double {mustBeInteger,mustBePositive} = 2
    end

    if runOptions.InitialDesignSize >= runOptions.MaximumTotalEvaluations
        error('ABSVR:InvalidABPCEBudget', ...
            'MaximumTotalEvaluations must exceed InitialDesignSize.');
    end
    if runOptions.QNorm > 1
        error('ABSVR:InvalidABPCEQNorm', 'QNorm must not exceed one.');
    end

    uqlabLocation = which('uqlab');
    if isempty(uqlabLocation)
        error('ABSVR:UQLabNotFound', 'UQLab is not on the MATLAB path.');
    end
    uqlab -nosplash
    rng(seed, 'twister');

    thisDirectory = fileparts(mfilename('fullpath'));
    addpath(thisDirectory);
    if strlength(outputFile) == 0
        repoRoot = fileparts(fileparts(fileparts(thisDirectory)));
        outputFile = fullfile(repoRoot, 'results', 'five_story_frame', ...
            sprintf('uqlab_abpce_seed_%d.json', seed));
    end
    outputFile = string(outputFile);
    outputDirectory = fileparts(outputFile);
    if ~isfolder(outputDirectory)
        mkdir(outputDirectory);
    end
    ledgerFile = replace(outputFile, ".json", "_calls.jsonl");
    if isfile(ledgerFile)
        delete(ledgerFile);
    end
    setenv('ABSVR_UQLAB_CALL_LEDGER', ledgerFile);
    clearLedgerEnvironment = onCleanup( ...
        @() setenv('ABSVR_UQLAB_CALL_LEDGER', '')); %#ok<NASGU>

    myInput = create_five_story_frame_input();
    modelOptions.mFile = 'five_story_frame_opensees_uqlab';
    modelOptions.isVectorized = true;
    myModel = uq_createModel(modelOptions);

    experimentalX = uq_getSample( ...
        myInput, runOptions.InitialDesignSize, 'LHS');
    experimentalY = uq_evalModel(myModel, experimentalX);
    internalX = uq_getSample(myInput, runOptions.InternalSampleSize, 'MC');
    selectedInternal = false(runOptions.InternalSampleSize, 1);

    stableIterations = 0;
    converged = false;
    iteration = 0;
    history = struct([]);
    finalPCE = [];
    while true
        iteration = iteration + 1;
        pceOptions.Type = 'Metamodel';
        pceOptions.MetaType = 'PCE';
        pceOptions.Method = 'LARS';
        pceOptions.Input = myInput;
        pceOptions.Degree = runOptions.Degree;
        pceOptions.TruncOptions.qNorm = runOptions.QNorm;
        pceOptions.TruncOptions.MaxInteraction = runOptions.MaximumInteraction;
        pceOptions.ExpDesign.X = experimentalX;
        pceOptions.ExpDesign.Y = experimentalY;
        pceOptions.Bootstrap.Replications = runOptions.BootstrapReplications;
        pceOptions.Display = 0;
        finalPCE = uq_createModel(pceOptions, '-private');

        [pfCentral, pfBootstrap, candidateIndex, minimumLearningScore] = ...
            evaluateActiveQuantities(finalPCE, internalX, selectedInternal, ...
                runOptions.BootstrapReplications, runOptions.ChunkSize);
        pfLower = min(pfBootstrap);
        pfUpper = max(pfBootstrap);
        if pfCentral > 0
            relativeRange = (pfUpper - pfLower) / pfCentral;
        else
            relativeRange = inf;
        end
        if relativeRange <= runOptions.ConvergenceTolerance
            stableIterations = stableIterations + 1;
        else
            stableIterations = 0;
        end
        converged = stableIterations >= ...
            runOptions.ConsecutiveConvergedIterations;

        basisIndices = finalPCE.PCE.Basis.Indices;
        coefficients = finalPCE.PCE.Coefficients;
        history(iteration).iteration = iteration;
        history(iteration).model_evaluations = size(experimentalX, 1);
        history(iteration).pf = pfCentral;
        history(iteration).pf_lower = pfLower;
        history(iteration).pf_upper = pfUpper;
        history(iteration).relative_bootstrap_range = relativeRange;
        history(iteration).stable_iterations = stableIterations;
        history(iteration).minimum_learning_score = minimumLearningScore;
        history(iteration).selected_degree = full(max(sum(basisIndices, 2)));
        history(iteration).nonzero_coefficients = nnz(coefficients);
        fprintf(['A-bPCE: N=%d, Pf=%.8g, range=[%.8g, %.8g], ' ...
            'relative range=%.5g, stable=%d\n'], size(experimentalX, 1), ...
            pfCentral, pfLower, pfUpper, relativeRange, stableIterations);

        if converged || size(experimentalX, 1) >= ...
                runOptions.MaximumTotalEvaluations
            break
        end
        selectedInternal(candidateIndex) = true;
        newX = internalX(candidateIndex, :);
        newY = uq_evalModel(myModel, newX);
        experimentalX(end + 1, :) = newX; %#ok<AGROW>
        experimentalY(end + 1, :) = newY; %#ok<AGROW>
    end

    ledgerEvaluations = countLedgerEvaluations(ledgerFile);
    if ledgerEvaluations ~= size(experimentalX, 1)
        error('ABSVR:CallCountMismatch', ...
            ['The A-bPCE experimental design contains %d points, but the ' ...
             'OpenSees ledger contains %d.'], size(experimentalX, 1), ...
            ledgerEvaluations);
    end

    summary.schema_version = 1;
    summary.method = 'direct UQLab A-bPCE with OpenSeesPy';
    summary.profile = 'published_abpce';
    summary.seed = seed;
    summary.status = ternary(converged, 'converged', 'maximum_evaluations_reached');
    summary.converged = converged;
    summary.pf = history(end).pf;
    summary.beta = -norminv(history(end).pf);
    summary.pf_bounds = [history(end).pf_lower, history(end).pf_upper];
    summary.beta_bounds = [-norminv(history(end).pf_upper), ...
        -norminv(history(end).pf_lower)];
    summary.model_evaluations_uqlab_design = size(experimentalX, 1);
    summary.model_evaluations_opensees_ledger = ledgerEvaluations;
    summary.initial_design = struct('sampling', 'LHS', ...
        'size', runOptions.InitialDesignSize);
    summary.internal_mcs = struct('sampling', 'MC', ...
        'size', runOptions.InternalSampleSize, 'reused_across_iterations', true);
    summary.bootstrap_replications = runOptions.BootstrapReplications;
    summary.pce = struct('method', 'LARS', 'degree', runOptions.Degree, ...
        'q_norm', runOptions.QNorm, ...
        'maximum_interaction', runOptions.MaximumInteraction, ...
        'bootstrap_implementation', 'UQLab fast bootstrap on selected sparse basis');
    summary.enrichment = struct('points_per_iteration', 1, ...
        'learning_function', 'absolute safe-minus-failed bootstrap count divided by B');
    summary.convergence = struct('relative_pf_range_tolerance', ...
        runOptions.ConvergenceTolerance, 'required_consecutive_iterations', ...
        runOptions.ConsecutiveConvergedIterations);
    summary.maximum_total_evaluations = runOptions.MaximumTotalEvaluations;
    summary.history = history;
    summary.matlab_release = version('-release');
    summary.matlab_version = version;
    summary.uqlab_entrypoint = uqlabLocation;
    summary.ledger_file = char(ledgerFile);

    temporaryFile = outputFile + ".tmp";
    fileIdentifier = fopen(temporaryFile, 'w');
    if fileIdentifier < 0
        error('ABSVR:OutputOpenFailed', 'Could not open %s.', temporaryFile);
    end
    closeOutput = onCleanup(@() fcloseIfOpen(fileIdentifier)); %#ok<NASGU>
    fprintf(fileIdentifier, '%s\n', jsonencode(summary, PrettyPrint=true));
    fclose(fileIdentifier);
    movefile(temporaryFile, outputFile, 'f');
    matFile = replace(outputFile, ".json", ".mat");
    save(matFile, 'summary', 'experimentalX', 'experimentalY', 'history');
end


function [pfCentral, pfBootstrap, bestIndex, bestScore] = ...
        evaluateActiveQuantities(pceModel, sample, excluded, replications, chunkSize)
    sampleSize = size(sample, 1);
    centralFailures = 0;
    bootstrapFailures = zeros(1, replications);
    bestIndex = 0;
    bestScore = inf;
    for first = 1:chunkSize:sampleSize
        last = min(first + chunkSize - 1, sampleSize);
        [centralPrediction, ~, bootstrapPrediction] = ...
            uq_evalModel(pceModel, sample(first:last, :));
        centralFailures = centralFailures + sum(centralPrediction <= 0);
        bootstrapFailure = bootstrapPrediction <= 0;
        bootstrapFailures = bootstrapFailures + sum(bootstrapFailure, 1);
        safeCount = replications - sum(bootstrapFailure, 2);
        score = abs(safeCount - sum(bootstrapFailure, 2)) / replications;
        score(excluded(first:last)) = inf;
        [chunkBest, relativeIndex] = min(score);
        if chunkBest < bestScore
            bestScore = chunkBest;
            bestIndex = first + relativeIndex - 1;
        end
    end
    if bestIndex == 0 || ~isfinite(bestScore)
        error('ABSVR:NoABPCECandidate', 'No unused enrichment candidate remains.');
    end
    pfCentral = centralFailures / sampleSize;
    pfBootstrap = bootstrapFailures / sampleSize;
end


function total = countLedgerEvaluations(ledgerFile)
    if ~isfile(ledgerFile)
        total = 0;
        return
    end
    lines = regexp(strtrim(fileread(ledgerFile)), '\r\n|\n|\r', 'split');
    total = 0;
    for index = 1:numel(lines)
        line = strtrim(lines{index});
        if ~isempty(line)
            entry = jsondecode(line);
            total = total + entry.evaluations;
        end
    end
end


function value = ternary(condition, trueValue, falseValue)
    if condition
        value = trueValue;
    else
        value = falseValue;
    end
end


function fcloseIfOpen(fileIdentifier)
    if fileIdentifier >= 0 && ~isempty(fopen(fileIdentifier))
        fclose(fileIdentifier);
    end
end
