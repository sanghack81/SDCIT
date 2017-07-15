function run_kcit_postnonlinear()
    addpath('gpml-matlab/gpml');
    addpath('algorithms');
    addpath('data');
    addpath('experiments');
    % Main experiments
    for noise=0:4
        for trial=0:299
            for independent=0:1
                for N=[200 400]
                    kcit2_postnonlinear(independent, noise, trial, N, 'results/kcit2_postnonlinear.csv');
                end
            end
        end
    end

    % For high-dimensional experiments
    N = 400;
    for noise=[9 19 49]
        for trial=0:299
            for independent=0:1
                kcit2_postnonlinear(independent, noise, trial, N, 'results/kcit2_postnonlinear.csv');
            end
        end
    end
end
