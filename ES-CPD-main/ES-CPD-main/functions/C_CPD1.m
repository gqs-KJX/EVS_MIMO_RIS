function [X, A, B, C, D,initer] = C_CPD1(Z, rank, sigma_2)
% Complex ALS for 4th-order tensor CP decomposition with regularization
% --------------------------------------------------------
% Input:
%   Z :      4D complex tensor of size I1 x I2 x I3 x I4
%   rank :   CP rank R
%   sigma_2: Tikhonov regularization parameter (e.g., noise variance)
%
% Output:
%   X      : Reconstructed tensor
%   A,B,C,D: Factor matrices (I1xR, I2xR, I3xR, I4xR)
%   res    : Relative residual norm: ||Z - X||_F / ||Z||_F
%   initer : Number of iterations performed
% --------------------------------------------------------
% Author: Modified from original by Lin Chen, extended to 4D.
% --------------------------------------------------------
    Z=tensor(Z);
    % Get dimensions
    I = size(Z);
    maxiter = 3e3; % Maximum number of iterations
    
    outiter = 0;
    initer = 0;
    terminated=0;
    % Initialize factor matrices using SVD of unfoldings
    A = tenmat(Z, 1).data;
    A = take_svd(A, rank);
    
    B = tenmat(Z, 2).data;
    B = take_svd(B, rank);
    
    C = tenmat(Z, 3).data;
    C = take_svd(C, rank);
    
    D = tenmat(Z, 4).data;
    D = take_svd(D, rank);
    
    % Optional: random initialization instead
    % rng(20); A = randn(I(1),rank) + 1j*randn(I(1),rank);
    % rng(20); B = randn(I(2),rank) + 1j*randn(I(2),rank);
    % rng(20); C = randn(I(3),rank) + 1j*randn(I(3),rank);
    % rng(20); D = randn(I(4),rank) + 1j*randn(I(4),rank);
    
    % Initial reconstruction
    X = ktensor(ones(rank,1), A, B, C, D);
    X = tensor(X);
    global verbose
    while ~terminated
        X_old = X;
    
        % === Update A ===
        PI_BC_D = khatrirao(D, C, B); % D ⊙ C ⊙ B
        A = updatafactor(1, PI_BC_D);
        [A, ~] = normalization_column(A);
    
        % === Update B ===
        PI_AC_D = khatrirao(D, C, A); % D ⊙ C ⊙ A
        B = updatafactor(2, PI_AC_D);
        [B, ~] = normalization_column(B);
    
        % === Update C ===
        PI_AB_D = khatrirao(D, B, A); % D ⊙ B ⊙ A
        C = updatafactor(3, PI_AB_D);
        [C, ~] = normalization_column(C);
    
        % === Update D ===
        PI_AB_C = khatrirao(C, B, A); % C ⊙ B ⊙ A
        D = updatafactor(4, PI_AB_C);
        [D, lambda] = normalization_column(D); % Normalize D and absorb scale into lambda
    
        % Reconstruct tensor with updated factors and weights
        X = ktensor(lambda.', A, B, C, D);
        X = tensor(X);
    
        initer = initer + 1;
        
    
    
        % Check convergence: small change or max iter reached
        if norm(X - X_old)/norm(X) < 1e-8 || outiter >= maxiter
            if outiter < maxiter
                converge = 1;
            else
                converge = 0;
            end
            % Absorb final lambda into D
            D = D * diag(lambda);
            terminated=1;
        end
    
        outiter = outiter + 1;
        if verbose==1
            fprintf('The number of iteration is %6.2f\n',outiter);
            fprintf('The residuals are %2.10f\n',norm(Z-X)/norm(Z));
            if  terminated==1
                  fprintf('\n\n');
            end
        end
        % Optional verbose output (uncomment if needed)
        % if exist('verbose','var') && verbose
        %     fprintf('Iteration %6d, Residual: %2.10f\n', outiter, res);
        % end
    end
    
    %% Nested functions
    function A=updatafactor(n_mode,PI)
        Z_fold=tenmat(Z,n_mode);
        nn=size(PI,2);
        new_A=(conj(double(Z_fold))*PI)/(PI'*PI+sigma_2*eye(nn));
        A=conj(new_A);
    end

    function U_matr=take_svd(Matr,rank)
        [U_matr,S,~]=svd(Matr,'econ');
        [m,n]=size(U_matr);
        if n>=rank
            U_matr=U_matr(:,1:rank);%*sqrt(S(1:rank,1:rank))
        else
            rng(10,'v5uniform');
            U_matr=[U_matr,randn(m,rank-n)+1j*randn(m,rank-n)];
        end
    end

    function [A,lambda]=normalization_column(A)
            lambda=sqrt(sum(abs(A).^2,1));
            M=size(A,1);
            A=A./(ones(M,1)*lambda);
        end

end