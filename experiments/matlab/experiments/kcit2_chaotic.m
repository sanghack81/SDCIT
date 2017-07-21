function kcit2_chaotic(independent, gamma, noise, trial, N, outputfile)
    args.gamma = gamma;
    args.noise = noise;
    data = synthetic('henon', trial, N, args);

    if independent
        X = data.Xt1;
        Y = data.Yt;
        Z = data.Xt(:,1:2);
    else
        X = data.Yt1;
        Y = data.Xt;
        Z = data.Yt(:,1:2);
    end

    start = tic;
    [statistic v2 boot_p_value v3 appr_p_value] =...
        CInd_test_new_withGP2(X, Y, Z, 0.01, 0);
    runtime = toc(start);

    fileid = fopen(outputfile, 'a+');
    line = sprintf('%d,%f,%d,%d,%d,%f,%f,%f,%f\n',...
            independent, gamma, noise, trial, N, runtime, statistic, boot_p_value, appr_p_value);
    fprintf(line);
    fprintf(fileid, line);
    fclose(fileid);
end
