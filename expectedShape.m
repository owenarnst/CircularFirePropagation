function Mavg = expectedShape(mat, p, maxt, maxr, k)
    % expectedShape.m simulates the growth of fire on a grid and 
    % computes the expected shape of the fire over multiple repetitions.
    % Inputs:
    %   mat   - initial grid state
    %   p     - probability parameter for fire spread
    %   maxt  - maximum number of time steps
    %   maxr  - maximum number of repetitions
    %   k     - scaling factor for intercardinal (diagonal) spread

    [sx, sy] = size(mat); % grid dimensions
    Mavg = zeros(sx, sy, maxt); % initialize matrix for averages

    % Reduce number of samples for efficiency
    num_samples = min(100, maxr); % use a subset of repetitions
    scale_factor = maxr / num_samples;

    % Parallel loop (using parfor for efficiency)
    Msum = zeros(sx, sy, maxt); % accumulate results
    parfor i = 1:num_samples 
        Mtemp = zeros(sx, sy, maxt); % temporary matrix for each repetition
        Mtemp(:,:,1) = mat; % initial condition

        % Precompute random numbers for all timesteps
        random_numbers = rand(sx, sy, maxt - 1); % random matrix for fire spread

        % Simulate fire spread over time
        for j = 1:maxt-1
            Mtemp(:,:,j+1) = Mtemp(:,:,j) + prop1round(p, Mtemp(:,:,j), random_numbers(:,:,j), k);
        end

        Msum = Msum + Mtemp; % accumulate results
    end

    % Compute the average fire spread
    Mavg = Msum * scale_factor / num_samples;
end

function fire_next = prop1round(p, fire_current, random_numbers, k)
    % prop1round computes the next state of the fire grid based on spread probabilities.
    % Inputs:
    %   p            - probability parameter for fire spread
    %   fire_current - current state of the fire grid (matrix)
    %   random_numbers - precomputed random matrix for the current time step
    %   k            - scaling factor for intercardinal (diagonal) spread

    % Define the cardinal kernel (spread to adjacent cells)
    cardinal_kernel = [0, 1, 0; 1, 0, 1; 0, 1, 0];
    % Define the intercardinal kernel (spread to diagonal cells)
    diagonal_kernel = [1, 0, 1; 0, 0, 0; 1, 0, 1];

    % Count burning neighbors: cardinal plus k-scaled diagonal contributions.
    neighbor_fire = conv2(fire_current > 0, cardinal_kernel, 'same') + ...
                    k * conv2(fire_current > 0, diagonal_kernel, 'same');
    
    % Determine the next fire state based on the probability condition
    fire_next = (random_numbers < (p .* neighbor_fire)) | fire_current;
end
