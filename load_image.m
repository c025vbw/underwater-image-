function img = load_image(num)
% num is the num of image with 

prefix = '..\data\images/';
suffix = '.png';
path = [prefix,num,suffix]

img = imread(path);