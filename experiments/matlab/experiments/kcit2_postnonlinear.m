function kcit2_postnonlinear(independent, noise, trial, N, outputfile)

    args.independent = independent;
    args.dimensions = (noise + 1);
    data = synthetic('caseI', trial, N, args);

    start = tic;
    [statistic v2 boot_p_value v3 appr_p_value] =...
        CInd_test_new_withGP2(data.X, data.Y, data.Z, 0.01, 0);
    runtime = toc(start);

    fileid = fopen(outputfile, 'a+');
    line = sprintf('%d,%d,%d,%d,%f,%f,%f,%f\n',...
            independent, noise, trial, N, runtime, statistic, boot_p_value, appr_p_value);
    fprintf(line);
    fprintf(fileid, line);
    fclose(fileid);
end
