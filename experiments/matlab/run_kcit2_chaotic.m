function run_kcit_chaotic()
    addpath('gpml-matlab/gpml');
    addpath('algorithms');
    addpath('data');
    addpath('experiments');
    for gamma=0:0.1:0.5
        for trial=0:299
            for independent=0:1
                for N=[200 400]
                    kcit2_chaotic(independent, gamma, 2, trial, N, 'results/kcit2_chaotic.csv');
                end
            end
        end
    end
end
