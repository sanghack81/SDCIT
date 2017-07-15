function run_kcit_chaotic_timing()
    maxNumCompThreads = 1;
    addpath('gpml-matlab/gpml');
    addpath('algorithms');
    addpath('data');
    addpath('experiments');
    gamma = 0;
    independent = 1;
    N = 400;
    for trial=0:299
        kcit_chaotic(independent, gamma, 2, trial, N, 'results/kcit_chaotic_timing.csv');
    end
    
    N = 200;
    for trial=0:299
        kcit_chaotic(independent, gamma, 2, trial, N, 'results/kcit_chaotic_timing.csv');
    end
end
