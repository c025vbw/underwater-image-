%red-chael法
% 一般的なヘイズ除去フィルタの関数
% Picは元画像(uint8)
% WidSizeはλ×λのマスク処理をするときのλの値(ダークチャネルや透過率の平滑化)
% の調整用Omegaはヘイズ除去パラメータ
% T0は透過率のゼロ割を防ぐ正定数(基本はT0=0.1)
% GuideWidSize:GuidedFilterに渡す引数 マスク処理のサイズ
% Epsilon:GuidedFilterの正定数(値が0に近いほど元の画像に近い)

clc;
clear all;
close all;
%画像が保存されているフォルダ
D = dir(['' '\*.png']);
%E_list{6} = dir(['C:\Users\Suzuki\Documents\MATLAB\result' '\*.png']);
imfolder = '';
rootname = 'file'; % ファイル名に使用する文字列
extension = '.png'; % 拡張子

%保存先のパス
infolder = '';
%% redチャネル画像生成
k=1;
for i = 1: length(D)
    %画像の読み込み
    img = imread([imfolder '\' D(i).name]);
    image_name = D(i).name(1, [1:end-4]);
    image_name_result = D(i).name(1, [1:end-4]);
    %figure,imshow(img);
    Pic = double(img)./255;
    [N,M,C] = size(img);
    
    %Rchanel逆転画像の作成
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

    %% 画像のサイズおよび色チャネルの取得
    [m,n,rgb]=size(dPic);

    %% 最小値画像の生成
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
    
    %% レッドチャネル画像の生成
    % DarkMat = ordfilt2(MinMat,1,ones(WidSize,WidSize),'symmetric');
    % SE = strel('square', WidSize);
    % DarkMat = imerode(MinMat,SE);
    % figure,imshow(DarkMat,[])
    % imwrite(DarkMat,'.\new_images\Dark.jpg')
    
    %%%% ダークチャネル画像から上位0.1%の輝度を探す %%%%

    %% satの計算
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
    %% ダークチャネル画像を行ベクトルにする
    vec = reshape(DarkMat,1,numel(DarkMat));

    %% 行ベクトルにしたダークチャネル画像をソートする
    vec = sort(vec,'descend');
    % vec = sort(vec);

    %% Ambientに画素値の上位10%を取り出し格納
    % Ambient = vec(ceil((n*m)*0.1)+1:n*m);
    Ambient = vec(1:ceil((n*m)*0.1)+1);
    %% 抜き出した画素値と等しい値を持つダークチャネル画像の位置情報から
    %% その位置だけ1で他は0になるような行列AmbientTableを作る
    AmbientTable = zeros(m,n);%初期化
    % AmbientTable(1:m,1:n)=0.0;%初期化
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
    %%検証用まずい処理
    % [row,col] = find(DarkMat==Ambient(1));
    % AmbientTable(row,col)=Ambient(1);
    %% 輝度画像生成 カラーからグレースケールにする
    LumiMat =mean(Pic,3);
    %  YCbCr = rgb2ycbcr(dPic);
    %  LumiMat =YCbCr(:,:,1);

    %% 原画像のレッドチャネルが上位10%になる画素だけ残す
    % AmbientMat = LumiMat.*AmbientTable;
    AmbientMat = Pic(:,:,1).*AmbientTable;
    AmbientMat(AmbientMat<=0) = 10;%下の行で０を選ばないようにする
    %% 残った輝度の中で一番高いものの位置を取り出し,元画像の色情報を環境光AmbientLumiに指定する
    [row,col]= find(AmbientMat == min(AmbientMat(:)),1);
    AmbientLumi=Pic(row,col,1:rgb);
    A = repmat(AmbientLumi,[m,n,1]);
    % A=uint8(A);
    %  figure,imshow(A)
    % imwrite(A,'.\new_images\A.jpg')
    %%%% 透過率行列の生成 %%%%
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
    %環境光
    Ar= double(A(:,:,1));
    Ag= double(A(:,:,2));
    Ab= double(A(:,:,3));
    new_Ar=ones(size(Ar)) - Ar;
    % 原画像
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

    %%%% 透過率の計算 %%%%
    % MinFltMinT = ordfilt2(MinT,1,ones(WidSize,WidSize),'symmetric');
    SE = strel('square', WidSize);
    MinFltMinT = imerode(MinT,SE);
    %tがマイナスにならないように正規化
    % MinFltMinT=MinFltMinT./max(MinFltMinT(:));
    t = ones(m,n) - MinFltMinT;
    %figure,imshow(t,[])
    % imwrite(t,'.\new_images\t.jpg')
    %% 透過率の平滑化処理
    T = GuidedFilter_Report(t,LumiMat,GuideWidSize,Epsilon);
    %figure,imshow(T)
    % imwrite(T,'.\new_images\At.jpg')
    %%%% 鮮明画像抽出 %%%%
    %%%小島君の処理(移植)%%%
     T = repmat( T , [ 1 , 1 , 3 ] ); % 奥行方向3成分に透過率を代入する
    % A = repmat(AmbientLumi,[m,n,1]);
    %% デヘイズ画像の生成

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