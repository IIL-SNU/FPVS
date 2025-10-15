mag=3.3;
nSampR=200;
nObj=[600,600];
nImg=193;

% baseDir_tmp='/home/jupyter-kcl724/data/data3/FPM_com/tissue NIR dataset/32.sto_S2F(U)_A3 1837852_sec1(pair22)/R';
% baseDir_tmp='/mnt/data3/FPM_com/tissue NIR dataset/33.sto_U_A3 16-68231_sec1/';
% baseDir='/home/kcl724/data/data3/FPM_com/tissue NIR dataset/34.bla_U_A4 23-36533_sec2/R';
% baseDir='/home/kcl724/data/data3/FPM_com/tissue NIR dataset/15.bladder_S_A4_23-36533_sec1/';
% baseDir='Y:\jupyter-kcl724/data/data3/FPM_com/tissue NIR dataset/22.sto_S_A3 18-37852_sec1/';
% baseDir='Y:\jupyter-kcl724/data/data3/FPM_com/tissue NIR dataset/19.bladder_F_SS16-06440_sec1/';
% baseDir='/home/kcl724/jupyter-kcl724/data/data3/FPM_com/tissue NIR dataset/19.bladder_F_SS16-06440_sec1/';
% baseDir='/mnt/data3/FPM_com/tissue NIR dataset/35.sto_U_A3_16-68231_sec1/';
% baseDir='/mnt/data3/FPM_com/tissue NIR dataset/36.sto_U_A3_16-68231_sec2/';
% baseDir='/mnt/data3/FPM_com/tissue NIR dataset/36.sto_U_A3_16-68231_sec2/';
% baseDir='/mnt/data3/FPM_com/tissue NIR dataset/37.sto_U_A3_16-68231_sec3/';
% baseDir='/mnt/data3/FPM_com/tissue NIR dataset/38.bla_U_A12_23-35177_sec1/';
% baseDir='/mnt/data3/FPM_com/tissue NIR dataset/39.sto_F1_A3_16-68731_sec1/';
% baseDir='/mnt/data3/FPM_com/tissue NIR dataset/40.sto_F3_A3_18-37852_sec2/';
% baseDir='/mnt/data3/FPM_com/tissue NIR dataset/41.sto_F1_A3_18-37852_sec3/';
% baseDir='/mnt/data3/FPM_com/tissue NIR dataset/42.sto_F3_A3_18-37852_sec4/';
% baseDir='/mnt/data3/FPM_com/tissue NIR dataset/43.bla_F3_A1_23-36533_sec1/';
% baseDir='/mnt/data3/FPM_com/tissue NIR dataset/44.bla_F3_A1_23-36533_sec2/';
baseDir='/mnt/data3/FPM_com/tissue NIR dataset/45.bla_U_A4_23-36533_sec3/';
% baseDir='/mnt/data3/FPM_com/tissue NIR dataset/46.bla_U_A4_23-36533_sec4/';
% 45.bla_U_A4_23-36533_sec3
% 47.bla_F3_A1_23-36533_sec3
% 48.bla_F3_A1_23-36533_sec4


disp(baseDir)


% 14.bladder_S_A1_23-36533_sec1


outDir=[baseDir,'R/data/results/'];
% outDir=[baseDir,'R/data/results_keep/'];

rawDir=[baseDir,'R/data/'];
saveDir=[baseDir,'R/Dataset/'];

disp(outDir);
disp(saveDir);

formatOut = 'mmdd_HHMM';
nowDate=datestr(now,formatOut);


if ~exist(saveDir,'dir')
   mkdir(saveDir);   
end

file_list = dir([outDir,'*.mat']);
file_list_raw = dir([rawDir,'*.mat']);

disp(length(file_list));

disp('load start');

for m=1:length(file_list)
    if mod(m,50)==0
        fprintf('%.2d / %.2d\n',m,length(file_list));
    end
    
    [filepath,in_file,ext]=fileparts(file_list(m).name);
    load(([outDir,in_file,ext]));

    obj_temp=cast(obj,"single");
    obj=zeros(600,600,8);
    obj(:,:,1)=abs(obj_temp)./max(max(abs(obj_temp)));
    obj(:,:,5)=angle(obj_temp);

    [filepath_raw,in_file_raw,ext_raw]=fileparts(file_list_raw(m).name);
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
% clear;clc;


