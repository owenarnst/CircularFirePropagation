function minimizer
    %% Simulation parameters
    p = 0.5;          % Given fire spread probability
    L = 0.1;          % Threshold for fire intensity
    maxt = 50;        % Total number of time steps in the simulation
    maxr = 100;       % Number of repetitions (for averaging)
    
    grid_size = 101;  % Define grid dimensions (101x101)
    mat = zeros(grid_size, grid_size);
    center = ceil([grid_size, grid_size] / 2);
    mat(center(1), center(2)) = 1;  % Ignite the center tile

    %% Define range of k values to test
    k_values = 0:0.1:2;  % Adjust the range/resolution as needed
    num_k = length(k_values);
    ratio_array = zeros(size(k_values));
    
    %% Loop over k values
    for idx = 1:num_k
        k = k_values(idx);
        % Compute the expected fire shape for this (p,k) combination.
        % expectedShape returns a 3D matrix with dimensions: grid x grid x time.
        Mavg = expectedShape(mat, p, maxt, maxr, k);
        finalShape = Mavg(:,:,end);  % Get the final state at time t = maxt
        
        % Determine m: count contiguous tiles along the northwards line 
        % (i.e., same column, moving upward from the center) that exceed L.
        m = 0;
        for i = center(1):-1:1
            if finalShape(i, center(2)) > L
                m = m + 1;
            else
                break;
            end
        end
        
        % Determine n: count contiguous tiles along the northeastern line 
        % (i.e., moving diagonally: row decreases, column increases) that exceed L.
        n = 0;
        i = center(1);
        j = center(2);
        while (i >= 1 && j <= grid_size)
            if finalShape(i, j) > L
                n = n + 1;
            else
                break;
            end
            i = i - 1;
            j = j + 1;
        end
        
        % Compute the ratio m/n for circularity.
        ratio_array(idx) = m / n;
    end
    
    %% Find the k value for which m/n is closest to sqrt2
    [~, best_idx] = min(abs(ratio_array - sqrt(2)));
    best_k = k_values(best_idx);
    
    fprintf('Best k value: %f, with m/n ratio = %f\n', best_k, ratio_array(best_idx));
    
    %% Plot the ratio vs. k values
    figure;
    plot(k_values, ratio_array, 'o-');
    xlabel('k');
    ylabel('m/n Ratio');
    title(sprintf('m/n Ratio vs. k for p = %f', p));
    grid on;
    
    %% Optionally, display the final fire shape for the best k
    Mavg_best = expectedShape(mat, p, maxt, maxr, best_k);
    finalShape_best = Mavg_best(:,:,end);
    figure;
    imagesc(finalShape_best);
    colorbar;
    title(sprintf('Final Fire Shape for Best k = %f', best_k));
end
