% 基于PCA变换的图像融合方法

up = imread('CT001.jpg');
low = imread('PET001.jpg');
subplot(2,2,1);imshow(up);title('高分辨率图像');
subplot(2,2,2);imshow(low);title('低分辨率图像');

[up_R] = double(up(:,:,1));
[up_G] = double(up(:,:,2));
[up_B] = double(up(:,:,3));

[low_R] = double(low(:,:,1));
[low_G] = double(low(:,:,2));
[low_B] = double(low(:,:,3));

[M,N,color] = size(up);

up_Mx = 0;
low_Mx = 0;
for i = 1:M
    for j = 1:N
        up_S = [up_R(i,j),up_G(i,j),up_B(i,j)];         % 生成由RGB组成的三维列向量
        up_Mx = up_Mx + up_S;                           % 叠加计算RGB各列向量的总和

        low_S = [low_R(i,j),low_G(i,j),low_B(i,j)];     
        low_Mx = low_Mx + low_S;
    end
end
up_Mx = up_Mx/(M * N);                                  % 计算三维列向量的平均值
low_Mx = low_Mx/(M * N);

up_Cx = 0;
low_Cx = 0;
for i = 1:M
    for j = 1:N
        up_S = [up_R(i,j),up_G(i,j),up_B(i,j)]';        % 矩阵转置
        up_Cx = up_Cx + up_S * up_S';                          

        low_S = [low_R(i,j),low_G(i,j),low_B(i,j)]';     
        low_Cx = low_Cx + low_S * low_S';
    end
end
up_Cx = up_Cx/(M * N) - up_Mx * up_Mx';                 % 计算协方差矩阵
low_Cx = low_Cx/(M * N) - low_Mx * low_Mx';

[up_A,up_latent] = eigs(up_Cx);                         % 协方差矩阵的特征向量组成的矩阵
[low_A,low_latent] = eigs(low_Cx);                      % 即PCA变换的系数矩阵，特征值

for i = 1 : M
    for j = 1 : N
       up_X = [up_R(i,j),up_G(i,j),up_G(i,j)]';        % 生成由R，G， B组成的三维列
       up_Y = up_A'*up_X;                              % 每个象素点进行PCA变换正变换
       up_Y = up_Y';
       up_R(i,j) = up_Y(1);                            % 高分辨率图片的第1主分量
       up_G(i,j) = up_Y(2);                            % 高分辨率图片的第2主分量
       up_B(i,j) = up_Y(3);                            % 高分辨率图片的第3主分量

       low_X = [low_R(i,j),low_G(i,j),low_G(i,j)]';
       low_Y = low_A'*low_X;
       low_Y = low_Y';
       low_R(i,j) = low_Y(1);                          % 低分辨率图片的第1主分量
       low_G(i,j) = low_Y(2);                          % 低分辨率图片的第2主分量
       low_B(i,j) = low_Y(3);                          % 低分辨率图片的第3主分量
   end
end

for i = 1 : M
    for j = 1 : N
       up_Y = [up_R(i,j),up_G(i,j),up_B(i,j)]';         % 生成由R，G， B组成的三维列向量 
       up_X = up_A*up_Y;                                % 每个象素点进行PCA变换反变换
       up_X = up_X';
       up_r(i,j) = up_X(1);
       up_g(i,j) = up_X(2);
       up_b(i,j) = up_X(3);

       low_Y = [up_R(i,j),low_G(i,j),low_B(i,j)]';
       low_X = low_A*low_Y;
       low_X = low_X';
       low_r(i,j) = low_X(1);
       low_g(i,j) = low_X(2);
       low_b(i,j) = low_X(3);
   end
end

RGB_up(:,:,1)=up_r;
RGB_up(:,:,2)=up_g;
RGB_up(:,:,3)=up_b;

RGB_low(:,:,1)=low_r;
RGB_low(:,:,2)=low_g;
RGB_low(:,:,3)=low_b;

subplot(2,2,3);imshow(uint8(RGB_up));title('高分辨率PCA变换图像');
subplot(2,2,4);imshow(uint8(RGB_low));title('低分辨率PCA变换图像');


