function A = amap(src_image)
% ‰æ‘œ‚ğƒÁ•ÏŠ·‚·‚éŠÖ”

    %ƒÁ•â³@ƒÁ1.2
    gannma=1.2;
    
    %Ô¬•ª‚Ì’Šo
    r_image=double(src_image(:, :, 1));
    newred_image=r_image.^1.2;
    
    onlyones=ones(size(newred_image));
    A=onlyones-newred_image;
end