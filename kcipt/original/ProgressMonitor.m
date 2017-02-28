classdef ProgressMonitor < handle
    properties
        tstart
        total
        print_interval
        progress
        pending
        msg
    end
    methods
        function obj = ProgressMonitor(total, msg, print_interval)
            if nargin < 3
                print_interval = 1;
            end
            if nargin < 2
                msg = [];
            end
            if nargin < 1
                total = 100;
            end
            obj.tstart = tic;
            obj.total = total;
            obj.print_interval = print_interval;
            obj.progress = 0;
            obj.pending = 0;
            obj.msg = msg;
            obj.print_progress();
        end

        function increment(obj, amount)
            if nargin < 2
                amount = 1;
            end
            obj.pending = obj.pending + amount;
            if ((100*obj.pending/obj.total) >= obj.print_interval)
                obj.progress = obj.progress + obj.pending;
                obj.pending = 0;
                obj.print_progress();
            end
        end

        function print_progress(obj)
            elapsed = toc(obj.tstart);
            pc = floor(100*obj.progress/obj.total);
            if pc > 0
                remaining = elapsed*(100 - pc)/pc;
                if remaining >= 3600
                    rm_str = sprintf('%.1f hours', remaining/3600);
                elseif remaining >= 60
                    rm_str = sprintf('%.1f minutes', remaining/60);
                else
                    rm_str = sprintf('%.1f seconds', remaining);
                end
            else
                rm_str = '???';
            end
            if obj.msg
                fprintf('%s: %3d%% (~%s remaining)\n', obj.msg, pc, rm_str);
            else
                fprintf('%3d%% (~%s remaining)\n', pc, rm_str);
            end
        end
    end
end
