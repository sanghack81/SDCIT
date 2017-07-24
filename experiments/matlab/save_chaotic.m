function save_chaotic()
    mkdir(char(getenv('HOME')+string('/kcipt_data')));
    addpath('data');
    for gamma=0:0.1:0.5
        for trial=0:299
            for independent=0:1
                for N=[200 400]
                    args.independent = independent;
                    args.gamma = gamma;
                    args.noise = 2;

                    data = synthetic('henon', trial, N, args);
                    
                    fname = sprintf(getenv('HOME')+string('/kcipt_data/%0.1f_%d_%d_%d_chaotic.mat'),gamma,trial,independent,N);
                    save(char(fname),'data','-v7')
                end
            end
        end
    end
end
