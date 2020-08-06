clear vars;
LF=zeros(512,512,9,9,3);
%for i=1:9
% lf_x(:,:,1,:)=imread('G:\LKY\epinet-master\hci_dataset\additional\town\input_Cam036.png');
% lf_x(:,:,2,:)=imread('G:\LKY\epinet-master\hci_dataset\additional\town\input_Cam037.png');
% lf_x(:,:,3,:)=imread('G:\LKY\epinet-master\hci_dataset\additional\town\input_Cam038.png');
% lf_x(:,:,4,:)=imread('G:\LKY\epinet-master\hci_dataset\additional\town\input_Cam039.png');
% lf_x(:,:,5,:)=imread('G:\LKY\epinet-master\hci_dataset\additional\town\input_Cam040.png');
% lf_x(:,:,6,:)=imread('G:\LKY\epinet-master\hci_dataset\additional\town\input_Cam041.png');
% lf_x(:,:,7,:)=imread('G:\LKY\epinet-master\hci_dataset\additional\town\input_Cam042.png');
% lf_x(:,:,8,:)=imread('G:\LKY\epinet-master\hci_dataset\additional\town\input_Cam043.png');
% lf_x(:,:,9,:)=imread('G:\LKY\epinet-master\hci_dataset\additional\town\input_Cam044.png');

% lf_x(:,:,1,:)=imread('G:\LKY\epinet-master\hci_dataset\additional\town\input_Cam036.png');
% lf_x(:,:,2,:)=imread('G:\LKY\epinet-master\hci_dataset\additional\town\input_Cam037.png');
% lf_x(:,:,3,:)=imread('G:\LKY\epinet-master\hci_dataset\additional\town\input_Cam038.png');
% lf_x(:,:,4,:)=imread('G:\LKY\epinet-master\hci_dataset\additional\town\input_Cam039.png');
% lf_x(:,:,5,:)=imread('G:\LKY\epinet-master\hci_dataset\additional\town\input_Cam040.png');
% lf_x(:,:,6,:)=imread('G:\LKY\epinet-master\hci_dataset\additional\town\input_Cam041.png');
% lf_x(:,:,7,:)=imread('G:\LKY\epinet-master\hci_dataset\additional\town\input_Cam042.png');
% lf_x(:,:,8,:)=imread('G:\LKY\epinet-master\hci_dataset\additional\town\input_Cam043.png');
% lf_x(:,:,9,:)=imread('G:\LKY\epinet-master\hci_dataset\additional\town\input_Cam044.png');

%imshow([lf_x(:,:,1,:)/255 lf_x(:,:,2,:)/255 lf_x(:,:,3,:)/255 lf_x(:,:,4,:)/255 lf_x(:,:,5,:)/255]);

imgPath='G:/LKY/epinet-master/hci_dataset/stratified/stripes/';
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

imwrite(LF_pinhole/255,'G:/LKY/专利/code_matlab/stratified/stripes/stripes_all.png');

%获取重聚焦后的EPI
% LF_all=imread('G:/LKY/专利/code_matlab/town/refocus_5/town_ref_-1.6_.png');
% LF=permute(reshape(LF_all,[9 512 9 512 3]),[2 4 1 3 5]);
% LF_EPI_x=reshape(permute(squeeze(LF(256,:,5,:,:)),[2 1 3]),[9 512 3]);
% LF_EPI_y=reshape(permute(squeeze(LF(:,256,:,5,:)),[2 1 3]),[9 512 3]);
% imwrite(LF_EPI_x,'G:/LKY/专利/code_matlab/town/refocus_5/LF_EPI_x.png');
% imwrite(LF_EPI_y,'G:/LKY/专利/code_matlab/town/refocus_5/LF_EPI_y.png');




%imshow(LF_pinhole/255);
%imwrite(LF_pinhole/255,'town.png');
%LF_EPI_x=reshape(permute(squeeze(LF(256,:,5,:,:)),[2 1 3]),[9 512 3]);
%imwrite(LF_EPI_x/255,'town_epi_x_256.png');
%LF_EPI_y=reshape(permute(squeeze(LF(:,256,:,5,:)),[2 1 3]),[9 512 3]);
%imwrite(LF_EPI_y/255,'town_epi_y_256.png');
%imshow(LF_EPI_x/255)
%imwrite(LF_EPI_x/255,'town_epi_x.png');
% LF_EPI_x_plus=reshape(permute(squeeze(LF(:,512,5,:,:)),[2 1 3]),[9 512 3]);
% imwrite(LF_EPI_x_plus/255,'town_epi_x_plus.png');


% LF_disp1=zeros(512,512,1,4,3);
% LF_disp1(:,:,1,1,:)=imread('G:\LKY\epinet-master\hci_dataset\additional\boardgames\input_Cam040.png');
% LF_disp1(:,:,1,2,:)=imread('G:\LKY\epinet-master\hci_dataset\additional\dishes\input_Cam040.png');
% LF_disp1(:,:,1,3,:)=imread('G:\LKY\epinet-master\hci_dataset\additional\pens\input_Cam040.png');
% LF_disp1(:,:,1,4,:)=imread('G:\LKY\epinet-master\hci_dataset\additional\pillows\input_Cam040.png');
% LF_disp_pinhole1=reshape(permute(LF_disp1,[1 3 2 4 5]),[512 4*512 3]);

% LF_disp2=zeros(512,512,1,4);
% LF_disp2(:,:,1,1)=imread('G:\LKY\专利\disp\boardgames.png');
% LF_disp2(:,:,1,2)=imread('G:\LKY\专利\disp\dishes.png');
% LF_disp2(:,:,1,3)=imread('G:\LKY\专利\disp\pens.png');
% LF_disp2(:,:,1,4)=imread('G:\LKY\专利\disp\pillows.png');
% LF_disp_pinhole2=reshape(permute(LF_disp2,[1 3 2 4]),[512 4*512]);

% LF_disp3=zeros(512,512,1,4,3);
% LF_disp3(:,:,1,1,:)=imread('G:\LKY\epinet-master\hci_dataset\additional\rosemary\input_Cam040.png');
% LF_disp3(:,:,1,2,:)=imread('G:\LKY\epinet-master\hci_dataset\additional\table\input_Cam040.png');
% LF_disp3(:,:,1,3,:)=imread('G:\LKY\epinet-master\hci_dataset\additional\tower\input_Cam040.png');
% LF_disp3(:,:,1,4,:)=imread('G:\LKY\epinet-master\hci_dataset\additional\town\input_Cam040.png');
% LF_disp_pinhole3=reshape(permute(LF_disp3,[1 3 2 4 5]),[512 4*512 3]);

% LF_disp4=zeros(512,512,1,4);
% LF_disp4(:,:,1,1)=imread('G:\LKY\专利\disp\rosemary.png');
% LF_disp4(:,:,1,2)=imread('G:\LKY\专利\disp\table.png');
% LF_disp4(:,:,1,3)=imread('G:\LKY\专利\disp\tower.png');
% LF_disp4(:,:,1,4)=imread('G:\LKY\专利\disp\town.png');
% LF_disp_pinhole4=reshape(permute(LF_disp4,[1 3 2 4]),[512 4*512]);

% imwrite(LF_disp_pinhole1/255,'LF_disp_pinhole1.png');
% imwrite(LF_disp_pinhole2/255,'LF_disp_pinhole2.png');
% imwrite(LF_disp_pinhole3/255,'LF_disp_pinhole3.png');
% imwrite(LF_disp_pinhole4/255,'LF_disp_pinhole4.png');