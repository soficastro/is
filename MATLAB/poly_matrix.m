function result = poly_matrix(matrix, poly)
    %POLY_MATRIX Generate polynomial terms from columns of a matrix.
    %   result = POLY_MATRIX(matrix, poly) takes an input matrix and a polynomial degree.
    %   It returns a new matrix containing:
    %       1. Original columns of the input matrix.
    %       2. All unique products of columns (including self-multiplication) up to the specified degree.

    % Validate inputs
    if poly < 1
        error('Polynomial degree must be at least 1.');
    end

    [numRows, numCols] = size(matrix);
    result = matrix; % Start with the original columns

    % Generate polynomial terms
    for degree = 2:poly
        % Generate all combinations with repetition (unique terms only)
        combinations = generate_combinations(numCols, degree);
        for i = 1:size(combinations, 1)
            term = prod(matrix(:, combinations(i, :)), 2); % Multiply selected columns
            result = [result, term]; % Append the new term
        end
    end
end

function combinations = generate_combinations(numCols, degree)
    % Generate all unique combinations with repetition for a given degree
    % This ensures terms are ordered and avoids duplicates.
    if degree == 1
        combinations = (1:numCols)';
    else
        previous_combinations = generate_combinations(numCols, degree - 1);
        combinations = [];
        for i = 1:numCols
            % Only add combinations where the current index is >= the previous indices
            valid_rows = all(previous_combinations >= i, 2);
            new_rows = [repmat(i, sum(valid_rows), 1), previous_combinations(valid_rows, :)];
            combinations = [combinations; new_rows];
        end
    end
end
