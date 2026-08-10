function myInput = create_five_story_frame_input()
%CREATE_FIVE_STORY_FRAME_INPUT Create the published 21-variable UQLab input.

    names = {'P1','P2','P3','E4','E5','I6','I7','I8','I9','I10','I11', ...
        'I12','I13','A14','A15','A16','A17','A18','A19','A20','A21'};
    means = [133.454, 88.97, 71.175, 2.1738e7, 2.3796e7, ...
        8.1344e-3, 1.1509e-2, 2.1375e-2, 2.5961e-2, ...
        1.0812e-2, 1.4105e-2, 2.3279e-2, 2.5961e-2, ...
        0.31256, 0.37210, 0.50606, 0.55815, ...
        0.25302, 0.29117, 0.37303, 0.41860];
    standardDeviations = [40.04, 35.59, 28.47, 1.9152e6, 1.9152e6, ...
        1.0834e-3, 1.2980e-3, 2.5961e-3, 3.0288e-3, ...
        2.5961e-3, 3.4615e-3, 5.6249e-3, 6.4902e-3, ...
        0.055815, 0.074420, 0.093025, 0.11163, ...
        0.093025, 0.10232, 0.12093, 0.19537];

    for index = 1:numel(names)
        inputOptions.Marginals(index).Name = names{index};
        if index <= 3
            inputOptions.Marginals(index).Type = 'Lognormal';
            inputOptions.Marginals(index).Moments = [ ...
                means(index), standardDeviations(index)];
        else
            % The paper tabulates the underlying Gaussian parameters. UQLab
            % must therefore receive Parameters, not post-truncation Moments.
            inputOptions.Marginals(index).Type = 'Gaussian';
            inputOptions.Marginals(index).Parameters = [ ...
                means(index), standardDeviations(index)];
            inputOptions.Marginals(index).Bounds = [0, inf];
        end
    end

    correlation = eye(21);
    correlation(4,5) = 0.90;
    correlation(5,4) = 0.90;
    for elementI = 1:8
        inertiaI = 5 + elementI;
        areaI = 13 + elementI;
        correlation(inertiaI, areaI) = 0.95;
        correlation(areaI, inertiaI) = 0.95;
        for elementJ = (elementI + 1):8
            inertiaJ = 5 + elementJ;
            areaJ = 13 + elementJ;
            pairs = [inertiaI inertiaJ; areaI areaJ; ...
                inertiaI areaJ; areaI inertiaJ];
            for pairIndex = 1:size(pairs, 1)
                row = pairs(pairIndex, 1);
                column = pairs(pairIndex, 2);
                correlation(row, column) = 0.13;
                correlation(column, row) = 0.13;
            end
        end
    end
    inputOptions.Copula = uq_GaussianCopula(correlation);
    myInput = uq_createInput(inputOptions);
end
