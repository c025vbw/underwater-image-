function A = amap(src_image)
% �摜�����ϊ�����֐�

    %���␳�@����1.2
    gannma=1.2;
    
    %�Ԑ����̒��o
    r_image=double(src_image(:, :, 1));
    newred_image=r_image.^1.2;
    
    onlyones=ones(size(newred_image));
    A=onlyones-newred_image;
end