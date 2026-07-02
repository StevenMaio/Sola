function [sensors, val] = Lazy_Greedy_Solve_Cardinality_Cons(f, num_sensors, budget)
    sensors = [];
    val = 0;
    for t = 1:budget
        best_v = -1;
        best_gain = -1;
        for v = 1:num_sensors
            gain = f([sensors, v]) - val;
            if gain > best_gain
                best_gain = gain;
                best_v = v;
            end
        end
        sensors(t) = best_v;
        val = val + best_gain;
    end
end
