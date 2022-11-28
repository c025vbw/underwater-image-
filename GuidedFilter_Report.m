function output = GuidedFilter_Report(InputPic,GuidePic,WidSize,Epsilon)
% GuidedFilter_Report - Description
% �_���ɍڂ��Ă����K�C�f�b�h�t�B���^�[�̃A���S���Y���̍Č�
% Syntax: output = GuidedFilter_Report(InputPic[double],GuidePic[double],WidSize[double],Epsilon[double])
%
% InputPic[double]:���͉摜
% GuidePic[double]:�K�C�h�摜
% WidSize[double]:�}�X�N�����̑��T�C�Y
% Epsilon[double]:���萔

dPic = InputPic;
dGuide = GuidePic;
    
BoxMat = ones(WidSize);
BoxMat = BoxMat./(WidSize^2);

%%%% �_���̃A���S���Y���ɂ�� %%%%
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