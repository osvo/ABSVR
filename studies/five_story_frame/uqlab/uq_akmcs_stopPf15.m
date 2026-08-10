function status = uq_akmcs_stopPf15(currentAnalysis)
%UQ_AKMCS_STOPPF15 Published two-step AK-MCS probability-range criterion.
%
% This is the UQLab stopPf definition with delta = 0.15, as specified for
% the five-storey frame by Marelli and Sudret (2018), and an explicit reset
% when an iteration does not satisfy the criterion.

    runtime = currentAnalysis.Internal.Runtime;
    initialSize = currentAnalysis.Internal.AKMCS.IExpDesign.N;
    sampleSize = length(runtime.gmean) + ...
        length(runtime.g(initialSize + 1:end));
    exactAdded = runtime.g(initialSize + 1:end);
    pf = (sum(runtime.gmean <= 0) + sum(exactAdded <= 0)) / sampleSize;
    pfUpper = (sum(runtime.gmean - 2 * runtime.gs <= 0) + ...
        sum(exactAdded <= 0)) / sampleSize;
    pfLower = (sum(runtime.gmean + 2 * runtime.gs <= 0) + ...
        sum(exactAdded <= 0)) / sampleSize;
    satisfiesRange = pf > 0 && (pfUpper - pfLower) / pf <= 0.15;

    previousSatisfied = isfield(runtime, 'Pfstop') && runtime.Pfstop == 1;
    status = double(satisfiesRange && previousSatisfied);
    currentAnalysis.Internal.Runtime.Pfstop = double(satisfiesRange);
    currentAnalysis.Internal.Runtime.ABSVR_stopPf15_satisfied = ...
        double(satisfiesRange);
    currentAnalysis.Internal.Runtime.ABSVR_stopPf15_converged = status;
end
