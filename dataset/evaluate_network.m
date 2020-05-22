% load the network 
% load('net75_iou.mat')

% read image from pixelImageLabelDatastore
testImages=readByIndex(test_plds,75:100);

for i=1:25
    subplot(5,5,i)
    im_pair = testImages(i,:);
    img = im_pair.inputImage{:};
    mask = im_pair.pixelLabelImage{:};
    img_mask = labeloverlay(img,mask);
    imshow(img_mask);
    
end
sgtitle('Ground Truth')

figure()
mask_preds = readimage(pxdsPred,75:100);
for j=1:25
    subplot(5,5,j)
    im_pair = testImages(j,:);
    img = im_pair.inputImage{:};
    mask = mask_preds{j};
    img_mask = labeloverlay(img,mask);
    imshow(img_mask);
    
end
sgtitle('Predicted Masks')
