clc
clear
H = tensor(reshape(1:256, 4, 4, 4,4));
A=tenmat(H,1);
A=double(A);
H=double(H);
[I1,I2,I3,I4]=size(H);
H1 = reshape(permute(H,[4,3,2,1]),I2*I3*I4,I1); % mode-1 unfolding of H
H1_T = H1.';