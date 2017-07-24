function save_postnonlinear_distance()
    mkdir(char(getenv('HOME')+string('/kcipt_data')));
    addpath('gpml-matlab/gpml');
    addpath('algorithms');
    addpath('kcipt');
    addpath('data');
    % Main experiments
    for noise=0:4
        for trial=0:299
            for independent=0:1
                for N=[200 400]
                    args.independent = independent;
                    args.dimensions = (noise + 1);
                    data = synthetic('caseI', trial, N, args);
                    
                    X = data.X;
                    Y = data.Y;
                    Z = data.Z;
                    D = regression_distance([X Y], Z);
                    
                    fname = sprintf(getenv('HOME')+string('/kcipt_data/dist_%d_%d_%d_%d_postnonlinear.mat'),noise,trial,independent,N);
                    save(char(fname),'D','-v7')
                end
            end
        end
    end

    % For high-dimensional experiments
    N = 400;
    for noise=[9 19 49]
        for trial=0:299
            for independent=0:1
                args.independent = independent;
                args.dimensions = (noise + 1);
                data = synthetic('caseI', trial, N, args);

                X = data.X;
                Y = data.Y;
                Z = data.Z;
                D = regression_distance([X Y], Z);
                
                fname = sprintf(getenv('HOME')+string('/kcipt_data/dist_%d_%d_%d_%d_postnonlinear.mat'),noise,trial,independent,N);
                save(char(fname),'D','-v7')
            end
        end
    end
end
