clear all;
close all;
clc

%%
figure(1)
[S, Fs] = audioread('Floyd.m4a'); % S = sample data, Fs = number of samples per sec in Hertz
tr_gnr = length(S)/Fs; % record time in seconds
S = S(1: Fs * 58); 
plot((1:length(S))/Fs,S);
% title('Sweet Child O Mine');
xlabel('Time [sec]'); ylabel('Amplitude');
% p8 = audioplayer(S,Fs); playblocking(p8);

L = 58; % spatial domain
n = length(S); % Fourier modes
t2 = linspace(0,L,n+1); t = t2(1:n);
k = (2*pi/L)*[0:n/2-1 -n/2:-1]; ks = fftshift(k);

%%
test = transpose(abs(ks/ 2/ pi - 250));
[kmin, kind] = min(test);
instrument_filter = zeros(n, 1);
instrument_filter(1:kind, 1) = 1; 

%% Floyd
a = 1000;
tau = 0:2:58;
Sgt_spec = zeros(n, length(tau));
fund = zeros(length(tau),1);
for j = 1:length(tau)
   gabor = exp(-a*(t - tau(j)).^2); % Gabor window function
   Sg = transpose(gabor).*S;
   Sgt = fftshift(abs(fft(Sg))) .* instrument_filter;
   [max_Sgt,ind] = max(Sgt);
   central = abs(ks(ind)); 
   fund(j,1) = central;
   gauss_filter = exp(-0.001*(ks - central).^2);
%    gauss_filter = 1;
   Sgt_spec(:,j) = transpose(gauss_filter) .* Sgt;
end
%%
figure(2)
pcolor(tau,ks/(2 * pi),Sgt_spec/(2 * pi))
shading interp
yyaxis left
set(gca,'ylim',[250,1000],'Fontsize',16)
ylabel('frequency')
xlabel('time')
colormap(hot)

yyaxis right
ylabel('Notes')
set(gca,'ylim',[250,1000],'Fontsize',16)
yticklabels({'D','Eb','E','F#','B','F','#F'})
yticks([294,311.13,330,370,492,587,698,740])

% yticklabels({'E','A','B','E', 'B'})
% yticks([83.6842,110,123,164.81,246])

title('Floyd guitar')
%% GNR
a = 1000;
tau = 0:0.5:tr_gnr;
Sgt_spec = zeros(n, length(tau));
fund = zeros(length(tau),1);
for j = 1:length(tau)
   gabor = exp(-a*(t - tau(j)).^2); % Gabor window function
   Sg = transpose(gabor).*S;
   Sgt = fftshift(abs(fft(Sg)));
   [max_Sgt,ind] = max(Sgt);
   central = abs(ks(ind)); 
   fund(j,1) = central;
   gauss_filter = exp(-0.001*(ks - central).^2);
%    gauss_filter = 1;
   Sgt_spec(:,j) = transpose(gauss_filter) .* Sgt;
end
    
figure(2)
% pcolor(tau,ks,Sgt_spec)
pcolor(tau,ks/(2 * pi),Sgt_spec/(2 * pi))
shading interp
yyaxis left
set(gca,'ylim',[20,1000],'Fontsize',16)
ylabel('frequency')
xlabel('time')
colormap(hot)

yyaxis right
ylabel('Notes')
set(gca,'ylim',[20,1000],'Fontsize',16)
yticklabels({'C#4', 'F#4', 'Ab4', 'C#5','F5','F#5'})
yticks([277,370,415,554,698,740])
title('GNR Spectrogram')

%%
figure(3)
plot(log(abs(Sgt_spec)/max(abs(Sgt_spec)) + 1))










    
    