audio_bitrates = ones(20,1) * 40;
vl = [1,1,1,1,1,9,9,9,9,9,1,1,1,1,1,6,6,6,6,6];
video_bitrates = vl' * 500
disRes = ones(20,1)*1920*1080;
codRes = ones(20,1)*1920*1080;
fps = ones(20,1)*25;
handheld = false;
ms = [1;5;7];
ls = [1;1;1];
mat_file = load('randomForest.mat');
forest = mat_file.forest;
P1203(audio_bitrates,video_bitrates,disRes,codRes,fps,handheld,ms,ls,forest)

