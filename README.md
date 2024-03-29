# Medical-Image-Fusion
This is repository about medical image fusion.

## Introduction
The fusion of PET metabolic images and CT anatomical images can display metabolic activity and anatomical position at the same time, playing an indispensable role in the staging diagnosis. In order to improve the quality and information of PET/CT images, this paper proposes a fusion method of PET and CT based on non-subsampled steerable pyramid transform and Siamese Autoencoder. As the information contained in different frequency domains in medical images is different, the non-subsampled steerable pyramid transform is used to decompose the image into two image components firstly, low-pass sub-band and high-pass sub-band. For low-pass image components that contain most of the background information, PCA method is used to fuse them; for high-pass image components that contain detailed information such as image textures and edges, this article uses Siamese Autoencoder to fuse them. The inverted non-subsampled steerable pyramid transform is used to obtain the final PET/CT fusion image for the two components after fusion.

## Requirement
``PIL,
numpy,
matplotlib,
tensorflow,
keras,
matlab(optional)``


## Network Aritecture
![fig1](https://user-images.githubusercontent.com/30771950/127951735-b2c44319-4b0e-4863-a637-03b17f0bcef6.jpg)

## Demo

.....
