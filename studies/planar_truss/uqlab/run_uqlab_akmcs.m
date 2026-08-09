function summary = run_uqlab_akmcs(seed, outputFile, profile)
%RUN_UQLAB_AKMCS Run one independently seeded UQLab AK-MCS replication.
%   The default profile is a transparent paper-like reconstruction: Kriging
%   with a Gaussian covariance, U learning, Pf-bound stopping at 10%, an
%   initial LHS of size 30, and an internal MCS population of 1e6.
%
%   The 2018 article used the same settings except that its initial design was
%   sampled uniformly in a ball using details not fully specified in the
%   article. Therefore this local profile must not be called an exact
%   reproduction of its published 300-call result.

    arguments
        seed (1,1) double {mustBeInteger, mustBeNonnegative} = 11
        outputFile (1,1) string = ""
        profile (1,1) string {mustBeMember(profile, ["paper_like", "original_akmcs"])} = "paper_like"
    end

    uqlabLocation = which('uqlab');
    if isempty(uqlabLocation)
        error('ABSVR:UQLabNotFound', ...
            ['UQLab is not on the MATLAB path. Install it and add its core ' ...
             'directory before running this function.']);
    end
    uqlab -nosplash
    rng(seed, 'twister');

    thisDirectory = fileparts(mfilename('fullpath'));
    addpath(thisDirectory);
    if strlength(outputFile) == 0
        repoRoot = fileparts(fileparts(fileparts(thisDirectory)));
        outputFile = fullfile(repoRoot, 'results', 'planar_truss', ...
            sprintf('uqlab_%s_seed_%d.json', profile, seed));
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
        @() setenv('ABSVR_UQLAB_CALL_LEDGER', ''));

    inputOptions.Marginals(1).Name = 'A1';
    inputOptions.Marginals(1).Type = 'Lognormal';
    inputOptions.Marginals(1).Moments = [2.0e-3, 2.0e-4];
    inputOptions.Marginals(2).Name = 'A2';
    inputOptions.Marginals(2).Type = 'Lognormal';
    inputOptions.Marginals(2).Moments = [1.0e-3, 1.0e-4];
    inputOptions.Marginals(3).Name = 'E1';
    inputOptions.Marginals(3).Type = 'Lognormal';
    inputOptions.Marginals(3).Moments = [2.1e11, 2.1e10];
    inputOptions.Marginals(4).Name = 'E2';
    inputOptions.Marginals(4).Type = 'Lognormal';
    inputOptions.Marginals(4).Moments = [2.1e11, 2.1e10];
    for loadIndex = 1:6
        marginalIndex = 4 + loadIndex;
        inputOptions.Marginals(marginalIndex).Name = sprintf('P%d', loadIndex);
        inputOptions.Marginals(marginalIndex).Type = 'Gumbel';
        inputOptions.Marginals(marginalIndex).Moments = [5.0e4, 7.5e3];
    end
    myInput = uq_createInput(inputOptions);

    modelOptions.mFile = 'planar_truss_opensees_uqlab';
    modelOptions.isVectorized = true;
    myModel = uq_createModel(modelOptions);

    analysisOptions.Type = 'Reliability';
    analysisOptions.Method = 'ALR';
    analysisOptions.Input = myInput;
    analysisOptions.Model = myModel;
    analysisOptions.ALR.Metamodel = 'Kriging';
    analysisOptions.ALR.Reliability = 'MCS';
    analysisOptions.ALR.LearningFunction = 'U';
    analysisOptions.ALR.IExpDesign.Sampling = 'LHS';
    analysisOptions.ALR.IExpDesign.N = 30;
    analysisOptions.ALR.Kriging.Corr.Family = 'Gaussian';
    analysisOptions.ALR.MaxCandidateSize = 1e6;
    analysisOptions.ALR.MaxAddedED = 970;
    analysisOptions.Simulation.MaxSampleSize = 1e6;
    analysisOptions.Simulation.BatchSize = 1e6;

    if profile == "paper_like"
        analysisOptions.ALR.Convergence = 'StopPfBound';
        analysisOptions.ALR.ConvThres = 0.10;
        analysisOptions.ALR.ConvIter = 2;
    else
        analysisOptions.ALR.Convergence = 'StopLF';
        analysisOptions.ALR.ConvThres = 2.0;
        analysisOptions.ALR.ConvIter = 1;
    end

    analysis = uq_createAnalysis(analysisOptions);
    result = analysis.Results;
    ledgerEvaluations = countLedgerEvaluations(ledgerFile);
    if ledgerEvaluations ~= result.ModelEvaluations
        error('ABSVR:CallCountMismatch', ...
            ['UQLab reports %d model evaluations, but the OpenSees ledger ' ...
             'contains %d. The run is not auditable.'], ...
            result.ModelEvaluations, ledgerEvaluations);
    end

    summary.schema_version = 1;
    summary.method = 'UQLab ALR / AK-MCS';
    summary.profile = char(profile);
    summary.seed = seed;
    summary.pf = result.Pf;
    summary.beta = result.Beta;
    summary.cov = result.CoV;
    summary.pf_ci = result.PfCI;
    summary.beta_ci = result.BetaCI;
    summary.model_evaluations_uqlab = result.ModelEvaluations;
    summary.model_evaluations_opensees_ledger = ledgerEvaluations;
    summary.input_order = {'A1','A2','E1','E2','P1','P2','P3','P4','P5','P6'};
    summary.initial_design = struct('sampling', 'LHS', 'size', 30);
    summary.internal_mcs_size = 1e6;
    summary.kriging_covariance = 'Gaussian';
    summary.learning_function = 'U';
    summary.convergence = analysisOptions.ALR.Convergence;
    summary.convergence_threshold = analysisOptions.ALR.ConvThres;
    summary.convergence_iterations = analysisOptions.ALR.ConvIter;
    summary.max_candidate_size = analysisOptions.ALR.MaxCandidateSize;
    summary.matlab_release = version('-release');
    summary.matlab_version = version;
    summary.uqlab_entrypoint = uqlabLocation;
    summary.ledger_file = char(ledgerFile);
    summary.comparability_note = [ ...
        'Direct UQLab run with the published probabilistic model and OpenSees. ' ...
        'The 2018 initial design was sampled uniformly in an insufficiently ' ...
        'specified ball; this local run uses a disclosed LHS and is paper-like, ' ...
        'not an exact reproduction of the published run.'];

    temporaryFile = outputFile + ".tmp";
    fileIdentifier = fopen(temporaryFile, 'w');
    if fileIdentifier < 0
        error('ABSVR:OutputOpenFailed', 'Could not open %s.', temporaryFile);
    end
    closeOutput = onCleanup(@() fcloseIfOpen(fileIdentifier));
    fprintf(fileIdentifier, '%s\n', jsonencode(summary, PrettyPrint=true));
    fclose(fileIdentifier);
    movefile(temporaryFile, outputFile, 'f');

    matFile = replace(outputFile, ".json", ".mat");
    save(matFile, 'summary', 'analysisOptions', 'result');
end


function total = countLedgerEvaluations(ledgerFile)
    if ~isfile(ledgerFile)
        total = 0;
        return
    end
    text = fileread(ledgerFile);
    % REGEXP returns a cell array for character input across supported MATLAB
    % releases. SPLITLINES changed its output container semantics, which made
    % JSONDECODE receive a cell instead of a scalar string in R2025b.
    lines = regexp(strtrim(text), '\r\n|\n|\r', 'split');
    total = 0;
    for index = 1:numel(lines)
        line = strtrim(lines{index});
        if ~isempty(line)
            entry = jsondecode(line);
            total = total + entry.evaluations;
        end
    end
end


function fcloseIfOpen(fileIdentifier)
    if fileIdentifier >= 0 && ~isempty(fopen(fileIdentifier))
        fclose(fileIdentifier);
    end
end
