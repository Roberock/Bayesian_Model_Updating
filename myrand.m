function Vrand = myrand(Nsample)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
load('ranstream.mat');

stream = RandStream('mt19937ar','Seed',0);
stream.State = savedState;

Vrand = rand(stream,Nsample,1);

end
