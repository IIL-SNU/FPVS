mag=3.3;
nSampR=200;
nObj=[600,600];
nImg=193;

% baseDir='/home/kcl724/data/data3/FPM_com/tissue NIR dataset/32.sto_S2F(U)_A3 1837852_sec1(pair22)/R';
% baseDir='/home/kcl724/data/data3/FPM_com/tissue NIR dataset/33.sto_U_A3 16-68231_sec1/R';
% baseDir='/home/kcl724/data/data3/FPM_com/tissue NIR dataset/34.bla_U_A4 23-36533_sec2/R';
% baseDir='/home/kcl724/data/data3/FPM_com/tissue NIR dataset/15.bladder_S_A4_23-36533_sec1/';
% baseDir='/home/kcl724/data/data3/FPM_com/tissue NIR dataset/14.bladder_S_A1_23-36533_sec1/';
% baseDir='/home/kcl724/data/data3/FPM_com/tissue NIR dataset/06.sto_S_10-805_sec1/';
% baseDir='/home/kcl724/data/data3/FPM_com/tissue NIR dataset/13.sto_S_15-711892_sec3/';
baseDir='/home/kcl724/data/data3/FPM_com/tissue NIR dataset/22.sto_S_A3 18-37852_sec1/';



% 14.bladder_S_A1_23-36533_sec1


stainDir = [baseDir,'Dataset/'];
% stainDir = [baseDir,'Dataset_newWB/'];
stainDir = [baseDir,'Dataset_new/'];

% outDir=[baseDir,'R/data/results/'];
outDir=[baseDir,'R/data/results_keep/'];
rawDir=[baseDir,'R/data/'];
saveDir=[baseDir,'R/Dataset_v2/'];

disp(outDir);
disp(saveDir);

formatOut = 'mmdd_HHMM';
nowDate=datestr(now,formatOut);


if ~exist(saveDir,'dir')
   mkdir(saveDir);   
end

file_list = dir([outDir,'*.mat']);
file_list_raw = dir([rawDir,'*.mat']);
file_stain = dir([stainDir,'*.mat']);

disp(length(file_stain));

disp('load start');




for m=1:length(file_stain)
    if mod(m,50)==0
        fprintf('%.2d / %.2d\n',m,length(file_stain));
    end
    
    
    
    [filepath,in_file,ext]=fileparts(file_stain(m).name);
    load(([stainDir,in_file,ext]));

    obj=cast(obj,"single");
    
    
    idx_curr = extractAfter(in_file,'FULLFOV_FPM_');
    idx_curr = extractBefore(idx_curr,4);
    idx_curr = str2num(idx_curr);

    [filepath_raw,in_file_raw,ext_raw]=fileparts(file_list_raw(idx_curr).name);
    dataset=load(([rawDir,in_file_raw,ext_raw]));

    dpix_cam=dataset.metadata.camera.pixelSize;
    
    lambda=dataset.metadata.illumination.wavelength;
    freqUV=dataset.metadata.source_list.na_init./lambda;
    
    con=nSampR.*dpix_cam./mag;
    pupilShiftY = round(freqUV(:,2)*con);
    pupilShiftX = round(freqUV(:,1)*con);
    
    % corresponding crop regions
    YXmid=floor(nObj./2)+1;
    hfSz=floor(nSampR/2);
    cropR=zeros(nImg,4);
    cropR(:,1) = YXmid(1)+pupilShiftY-hfSz; %y start
    cropR(:,2) = YXmid(1)+pupilShiftY+hfSz-1; %y end
    cropR(:,3) = YXmid(2)+pupilShiftX-hfSz; %x start
    cropR(:,4) = YXmid(2)+pupilShiftX+hfSz-1; %x end
    
    data=dataset.data;



    save([saveDir,in_file,ext],'obj','data','cropR');
    
end

disp('load finish')
clear;clc;


