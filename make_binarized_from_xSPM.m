function make_binarized_from_xSPM(xSPM)
[rootDir, compNum, prac] = fileparts(xSPM.swd);
[biDir, second_dir, prac] = fileparts(rootDir);
dataLoc = fullfile(strcat('3_',second_dir), compNum, xSPM.Vspm.fname);

% Read data
headerInfo = spm_vol(dataLoc);
data = spm_read_vols(headerInfo);

% z-scores in xSPM.XYZ
ravel_index_xyz = sub2ind(size(data), xSPM.XYZ(1,:),xSPM.XYZ(2,:),xSPM.XYZ(3,:));
z_scores_xyz = data(ravel_index_xyz);

% zero voxels apart from xSPM.XYZ
new_data = zeros(size(data));
new_data(ravel_index_xyz) = z_scores_xyz;

% grep title
title_split = strsplit(xSPM.title);

% new nifti name
new_fname = strcat(compNum, title_split(1), '.nii');
headerInfo.fname= new_fname{1}
headerInfo.private.dat.fname = headerInfo.fname;

spm_write_vol(headerInfo, new_data);




