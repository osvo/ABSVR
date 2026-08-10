function summary = run_uqlab_akmcs(seed, outputFile)
%RUN_UQLAB_AKMCS Run one independently seeded native UQLab AK-MCS analysis.

    arguments
        seed (1,1) double {mustBeInteger, mustBeNonnegative} = 11
        outputFile (1,1) string = ""
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
            sprintf('uqlab_akmcs_seed_%d.json', seed));
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

    myInput = create_five_story_frame_input();
    modelOptions.mFile = 'five_story_frame_opensees_uqlab';
    modelOptions.isVectorized = true;
    myModel = uq_createModel(modelOptions);

    analysisOptions.Type = 'Reliability';
    analysisOptions.Method = 'AKMCS';
    analysisOptions.Input = myInput;
    analysisOptions.Model = myModel;
    analysisOptions.Simulation.MaxSampleSize = 1e6;
    analysisOptions.Simulation.BatchSize = 1e6;
    analysisOptions.AKMCS.MetaModel = 'Kriging';
    analysisOptions.AKMCS.LearningFunction = 'U';
    analysisOptions.AKMCS.IExpDesign.Sampling = 'LHS';
    analysisOptions.AKMCS.IExpDesign.N = 40;
    analysisOptions.AKMCS.Kriging.Corr.Family = 'Gaussian';
    analysisOptions.AKMCS.MaxAddedED = 260;
    analysisOptions.AKMCS.Convergence = 'stopPf15';

    analysis = uq_createAnalysis(analysisOptions);
    result = analysis.Results;
    ledgerEvaluations = countLedgerEvaluations(ledgerFile);
    if ledgerEvaluations ~= result.ModelEvaluations
        error('ABSVR:CallCountMismatch', ...
            ['UQLab reports %d model evaluations, but the OpenSees ledger ' ...
             'contains %d.'], result.ModelEvaluations, ledgerEvaluations);
    end

    summary.schema_version = 1;
    historyPf = result.History.Pf;
    historyPfLower = result.History.PfLower;
    historyPfUpper = result.History.PfUpper;
    converged = isfield(analysis.Internal.Runtime, ...
        'ABSVR_stopPf15_converged') && ...
        analysis.Internal.Runtime.ABSVR_stopPf15_converged == 1;

    summary.method = ['direct UQLab native AK-MCS with OpenSeesPy and ' ...
        'published frame stopping criterion'];
    summary.profile = 'published_akmcs';
    summary.seed = seed;
    summary.status = ternary(converged, 'converged', ...
        'maximum_evaluations_reached');
    summary.converged = converged;
    summary.pf = result.Pf;
    summary.beta = result.Beta;
    summary.cov = result.CoV;
    summary.pf_binomial_ci = result.PfCI;
    summary.beta_binomial_ci = result.BetaCI;
    summary.pf_surrogate_bounds = [historyPfLower(end), historyPfUpper(end)];
    summary.beta_surrogate_bounds = [-norminv(historyPfUpper(end)), ...
        -norminv(historyPfLower(end))];
    summary.model_evaluations_uqlab = result.ModelEvaluations;
    summary.model_evaluations_opensees_ledger = ledgerEvaluations;
    summary.input_order = {'P1','P2','P3','E4','E5','I6','I7','I8','I9', ...
        'I10','I11','I12','I13','A14','A15','A16','A17','A18','A19','A20','A21'};
    summary.initial_design = struct('sampling', 'LHS', 'size', 40);
    summary.internal_mcs_size = 1e6;
    summary.kriging_covariance = 'Gaussian';
    summary.learning_function = 'U';
    summary.convergence = 'stopPf15';
    summary.convergence_relative_pf_range_tolerance = 0.15;
    summary.convergence_required_consecutive_iterations = 2;
    summary.max_added_design = 260;
    summary.maximum_total_evaluations = 300;
    summary.matlab_release = version('-release');
    summary.matlab_version = version;
    summary.uqlab_entrypoint = uqlabLocation;
    summary.ledger_file = char(ledgerFile);

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


function value = ternary(condition, trueValue, falseValue)
    if condition
        value = trueValue;
    else
        value = falseValue;
    end
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


function fcloseIfOpen(fileIdentifier)
    if fileIdentifier >= 0 && ~isempty(fopen(fileIdentifier))
        fclose(fileIdentifier);
    end
end
