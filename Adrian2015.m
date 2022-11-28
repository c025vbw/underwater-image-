%red-chael�@
% ��ʓI�ȃw�C�Y�����t�B���^�̊֐�
% Pic�͌��摜(uint8)
% WidSize�̓Ɂ~�ɂ̃}�X�N����������Ƃ��̃ɂ̒l(�_�[�N�`���l���ⓧ�ߗ��̕�����)
% �̒����pOmega�̓w�C�Y�����p�����[�^
% T0�͓��ߗ��̃[������h�����萔(��{��T0=0.1)
% GuideWidSize:GuidedFilter�ɓn������ �}�X�N�����̃T�C�Y
% Epsilon:GuidedFilter�̐��萔(�l��0�ɋ߂��قǌ��̉摜�ɋ߂�)

clc;
clear all;
close all;
%�摜���ۑ�����Ă���t�H���_
D = dir(['' '\*.png']);
%E_list{6} = dir(['C:\Users\Suzuki\Documents\MATLAB\result' '\*.png']);
imfolder = '';
rootname = 'file'; % �t�@�C�����Ɏg�p���镶����
extension = '.png'; % �g���q

%�ۑ���̃p�X
infolder = '';
%% red�`���l���摜����
k=1;
for i = 1: length(D)
    %�摜�̓ǂݍ���
    img = imread([imfolder '\' D(i).name]);
    image_name = D(i).name(1, [1:end-4]);
    image_name_result = D(i).name(1, [1:end-4]);
    %figure,imshow(img);
    Pic = double(img)./255;
    [N,M,C] = size(img);
    
    %Rchanel�t�]�摜�̍쐬
    R=Pic(:,:,1);
    G=Pic(:,:,2);
    B=Pic(:,:,3);
    onlyones=ones(size(R));
    A=abs(onlyones-R);
    dPic=cat(3, A, G, B);

    WidSize = 24;
    GuideWidSize = 41;
    Omega = 0.95;
    Epsilon = 0.001;
    T0=0.1;

    %% �摜�̃T�C�Y����ѐF�`���l���̎擾
    [m,n,rgb]=size(dPic);

    %% �ŏ��l�摜�̐���
    MinMat = zeros(m,n);
    % for i = 1:m
    %     for j = 1:n
    %         Temp = [0,0,0];
    %         for ch = 1:rgb
    %             Temp(ch) = dPic(i,j,ch);
    %         end
    %         MinMat(i,j) = min(Temp);
    %     end
    % end
    MinMat = min(dPic,[],3);
    
    %% ���b�h�`���l���摜�̐���
    % DarkMat = ordfilt2(MinMat,1,ones(WidSize,WidSize),'symmetric');
    % SE = strel('square', WidSize);
    % DarkMat = imerode(MinMat,SE);
    % figure,imshow(DarkMat,[])
    % imwrite(DarkMat,'.\new_images\Dark.jpg')
    
    %%%% �_�[�N�`���l���摜������0.1%�̋P�x��T�� %%%%

    %% sat�̌v�Z
    MaxMat = zeros(m,n);
    MaxMat = max(Pic,[],3);
    Min = zeros(m,n);
    Min = min(Pic,[],3);
    Sat=(MaxMat - Min)./MaxMat;
    %  figure,imshow(Sat)
    % imwrite(Sat,'.\new_images\Sat.jpg')
    DarkMat=min(Sat,MinMat);
    SE = strel('square', WidSize);
    DarkMat = imerode(DarkMat,SE);
    %% �_�[�N�`���l���摜���s�x�N�g���ɂ���
    vec = reshape(DarkMat,1,numel(DarkMat));

    %% �s�x�N�g���ɂ����_�[�N�`���l���摜���\�[�g����
    vec = sort(vec,'descend');
    % vec = sort(vec);

    %% Ambient�ɉ�f�l�̏��10%�����o���i�[
    % Ambient = vec(ceil((n*m)*0.1)+1:n*m);
    Ambient = vec(1:ceil((n*m)*0.1)+1);
    %% �����o������f�l�Ɠ������l�����_�[�N�`���l���摜�̈ʒu��񂩂�
    %% ���̈ʒu����1�ő���0�ɂȂ�悤�ȍs��AmbientTable�����
    AmbientTable = zeros(m,n);%������
    % AmbientTable(1:m,1:n)=0.0;%������
    % for i = 1:numel(Ambient)
    %     for j = 1:m
    %         for k = 1:n
    %             if  DarkMat(j,k) == Ambient(i)
    % %                 if  DarkMat(j,k) ==1
    %                 if  AmbientTable(j,k) ==1
    %                     AmbientTable(j,k) =AmbientTable(j,k) * 1.0;
    %                 else
    %                     AmbientTable(j,k) = 1.0;
    %                 end
    %                 else
    % %                 if  DarkMat(j,k) ==0
    %                 if  AmbientTable(j,k) ==0
    %                     AmbientTable(j,k) = 0;
    %                 else
    %                     AmbientTable(j,k) =AmbientTable(j,k) * 1.0;
    %                 end
    %                 
    %             end
    %         end
    %     end
    % end
    % for i = 1:numel(Ambient)
    idx = DarkMat >= Ambient(ceil((n*m)*0.1)+1);
    AmbientTable(idx) = 1;
    % end
    %%���ؗp�܂�������
    % [row,col] = find(DarkMat==Ambient(1));
    % AmbientTable(row,col)=Ambient(1);
    %% �P�x�摜���� �J���[����O���[�X�P�[���ɂ���
    LumiMat =mean(Pic,3);
    %  YCbCr = rgb2ycbcr(dPic);
    %  LumiMat =YCbCr(:,:,1);

    %% ���摜�̃��b�h�`���l�������10%�ɂȂ��f�����c��
    % AmbientMat = LumiMat.*AmbientTable;
    AmbientMat = Pic(:,:,1).*AmbientTable;
    AmbientMat(AmbientMat<=0) = 10;%���̍s�łO��I�΂Ȃ��悤�ɂ���
    %% �c�����P�x�̒��ň�ԍ������̂̈ʒu�����o��,���摜�̐F��������AmbientLumi�Ɏw�肷��
    [row,col]= find(AmbientMat == min(AmbientMat(:)),1);
    AmbientLumi=Pic(row,col,1:rgb);
    A = repmat(AmbientLumi,[m,n,1]);
    % A=uint8(A);
    %  figure,imshow(A)
    % imwrite(A,'.\new_images\A.jpg')
    %%%% ���ߗ��s��̐��� %%%%
    % MinT = zeros(m,n);
    % for i = 1:m
    %     for j = 1:n
    %         Temp=[0,0,0];
    %         for ch = 1:rgb
    %             Temp(ch) = dPic(i,j,ch)/(AmbientLumi(1,1,ch)+eps);
    %         end
    %          MinT(i,j) = min(Temp);
    %     end
    % end
    %����
    Ar= double(A(:,:,1));
    Ag= double(A(:,:,2));
    Ab= double(A(:,:,3));
    new_Ar=ones(size(Ar)) - Ar;
    % ���摜
    R_r=double(Pic(:,:,1));
    G_g=double(Pic(:,:,2));
    B_b=double(Pic(:,:,3));
    new_R=ones(size(R_r))-R_r;
    X=new_R./(new_Ar+Epsilon);
    Y=G_g./(Ag+Epsilon);
    Z=B_b./(Ab+Epsilon);
    Temp = cat(4, X, Y, Z,1.*DarkMat);
    % figure,imshow(Temp)
    MinT = min(Temp,[],4);

    %%%% ���ߗ��̌v�Z %%%%
    % MinFltMinT = ordfilt2(MinT,1,ones(WidSize,WidSize),'symmetric');
    SE = strel('square', WidSize);
    MinFltMinT = imerode(MinT,SE);
    %t���}�C�i�X�ɂȂ�Ȃ��悤�ɐ��K��
    % MinFltMinT=MinFltMinT./max(MinFltMinT(:));
    t = ones(m,n) - MinFltMinT;
    %figure,imshow(t,[])
    % imwrite(t,'.\new_images\t.jpg')
    %% ���ߗ��̕���������
    T = GuidedFilter_Report(t,LumiMat,GuideWidSize,Epsilon);
    %figure,imshow(T)
    % imwrite(T,'.\new_images\At.jpg')
    %%%% �N���摜���o %%%%
    %%%�����N�̏���(�ڐA)%%%
     T = repmat( T , [ 1 , 1 , 3 ] ); % ���s����3�����ɓ��ߗ���������
    % A = repmat(AmbientLumi,[m,n,1]);
    %% �f�w�C�Y�摜�̐���

    % idx = find(T < T0);
    idx = T < T0;
    T(idx) = T0;
    % A=uint8(A);
    %figure,imshow(new_A)
    new_Ar=ones(size(Ar)) - Ar;
    new_Ag=ones(size(Ag)) - Ag;
    new_Ab=ones(size(Ab)) - Ab;
    O = (Pic(:,:,1) - A(:,:,1))./T(:,:,1)+ 0.3;% + A(:,:,1).*new_Ar;
    P = (Pic(:,:,2) - A(:,:,2))./T(:,:,2)+ 0.3;% + A(:,:,2).*new_Ag;
    Q = (Pic(:,:,3) - A(:,:,3))./T(:,:,3)+ 0.3;% + A(:,:,3).*new_Ab;
    R=cat(3,O,P,Q);
    figure,imshow(R);
    % imwrite(R,'.\new_images\zenn.jpg')
    O = (O-min(O(:)))./(max(O(:))-min(O(:)));
    P = (P-min(P(:)))./(max(P(:))-min(P(:)));
    Q = (Q-min(Q(:)))./(max(Q(:))-min(Q(:)));
    R=cat(3,O,P,Q);
    R = uint8(R.*255);
    %figure,imshow(R);
    
    filename = [image_name, extension];
    imwrite(R,[infolder '\' filename]);
    k = k+1;
end