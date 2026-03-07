
atlas='dbs80';
load([group '_' atlas '.mat']);

N=size(data{1},1);

indexN=1:N;
NSUB=size(data,2);
TR=double(TR);
% Parameters of the data
% Bandpass filter settings
fnq=1/(2*TR);                 % Nyquist frequency
flp = 0.008;                    % lowpass frequency of filter (Hz)
fhi = 0.08;                    % highpass
Wn=[flp/fnq fhi/fnq] ;        % butterworth bandpass non-dimensional frequency
k=2;                          % 2nd order butterworth filter
[bfilt,afilt]=butter(k,Wn);   % construct the filter

for sub=1:NSUB
    sub    
    clear signal_filt Power_Areas;
    ts=data{sub};
    ts=ts(indexN,:);

    for seed=1:N
        ts(seed,:)=detrend(ts(seed,:)-mean(ts(seed,:)));
        signal_filt(seed,:) =filtfilt(bfilt,afilt,ts(seed,:));
    end
    signal_filt=signal_filt(:,10:end-10);
   
    [Ns, Tmaxred]=size(signal_filt);
    TT=Tmaxred;
    Ts = TT*TR;
    freq = (0:TT/2-1)/Ts;
    nfreqs=length(freq);
    for seed=1:N
        pw = abs(fft(zscore(signal_filt(seed,:))));
        PowSpect = pw(1:floor(TT/2)).^2/(TT/TR);
        Power_Areas=gaussfilt(freq,PowSpect,0.005);
        [~,index]=max(Power_Areas);
        index=squeeze(index);
        f_diff_sub(sub,seed)=freq(index);
    end
    
end

f_diff = mean(f_diff_sub,1);

save(['fdiff_' group '_' atlas '.mat'], 'f_diff');

