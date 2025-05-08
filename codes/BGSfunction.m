function gainSignal = BGSfunction(deltaT,pulseWidth,fiberLength,BFS,SW,Intensity,sweepFreq)
%% 用于计算光纤的BGS，具体公式参考2019_Frequency shift estimation technique near the hotspot in BOTDA sensor
T = pulseWidth:-2*deltaT:0;

%% 脉冲扫频输出信号
gainSignal = zeros(length(sweepFreq),fiberLength+length(T)-1);
for m = 1:length(sweepFreq)
    tau = 1i*pi*(BFS.^2-sweepFreq(m)^2-1i*sweepFreq(m)*SW)/sweepFreq(m);
    Gain = Intensity.*real((1-exp(-T'*conj(tau)))./conj(tau));
    for n = 1:size(gainSignal,2)
        gainSignal(m,n) = sum(diag(Gain,n-length(T)));
    end
end

end