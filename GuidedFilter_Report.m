function output = GuidedFilter_Report(InputPic,GuidePic,WidSize,Epsilon)
% GuidedFilter_Report - Description
% 論文に載っていたガイデッドフィルターのアルゴリズムの再現
% Syntax: output = GuidedFilter_Report(InputPic[double],GuidePic[double],WidSize[double],Epsilon[double])
%
% InputPic[double]:入力画像
% GuidePic[double]:ガイド画像
% WidSize[double]:マスク処理の窓サイズ
% Epsilon[double]:正定数

dPic = InputPic;
dGuide = GuidePic;
    
BoxMat = ones(WidSize);
BoxMat = BoxMat./(WidSize^2);

%%%% 論文のアルゴリズムによる %%%%
meanGuide = imfilter(dGuide,BoxMat,'symmetric');
meanPic = imfilter(dPic,BoxMat,'symmetric');
corrGuidePic = imfilter(dGuide.*dPic,BoxMat,'symmetric');
covGuidePic = corrGuidePic - (meanGuide.*meanPic);

corrGuide = imfilter(dGuide.*dGuide,BoxMat,'symmetric');

varGuide = corrGuide - (meanGuide.*meanGuide);

a = covGuidePic ./ (varGuide + Epsilon);
b = meanPic -(a.*meanGuide);

mean_a = imfilter(a,BoxMat,'symmetric');
mean_b = imfilter(b,BoxMat,'symmetric');

q = (mean_a.*dGuide) + mean_b;

output = q;    
end