clear vars;
LF=zeros(512,512,9,9,3);


imgPath='hci_dataset/training/boxes/';
num=1;
for i=1:9
  for j=1:9  
      imgDir  = dir([imgPath '*.png']);
      LF(:,:,i,j,:)=imread([imgPath imgDir(num).name]);
      num=num+1;
  end
end  
LF_pinhole=reshape(permute(LF,[3 1 4 2 5]),[9*512 9*512 3]);
%LF_pinhole=reshape(permute(LF,[1 3 2 4 5]),[9*512 9*512 3]);

imwrite(LF_pinhole/255,'boxes/boxes_all.png');

