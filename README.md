<div align="center">   

# PyTorch Image Segmentation Project
</div>

### 环境配置
python version 3.8, torch 1.8.1, torchvision 0.9.1:
```
pip install torch==1.8.1 torchvision==0.9.1
```


### 数据准备
数据文件夹结构如下:
```
datasets/
  images/    # images
     train/
        img1.jpg
        img2.jpg
         .
         .
         .
     val/
        img1.jpg
        img2.jpg
         .
         .
         .
  labels/     # masks
     train/
        img1.png
        img2.png
         .
         .
         .
     val/
        img1.png
        img2.png
         .
         .
         .
```
### 训练
```
python train.py --input_size 224 224 --batch_size 32 --epochs 100 --nb_classes 2 --data_path ./datasets/ --output_dir ./output_dir 
```
### 评价模型
```
python eval.py --input_size 224 224 --batch_size 8 --weights ./output_dir/best.pth --data_path ./datasets/ --nb_classes 2
```
### 模型预测
```
python predict.py --input_size 224 224 --weights ./output_dir/best.pth --image_path ./1.jpg --nb_classes 2
```
### 导出onnx模型
```
python export_onnx.py --input_size 224 224 --weights ./output_dir/best.pth --nb_classes 2
python -m onnxsim best.onnx best_sim.onnx
```

### 结果可视化
#### 1. Pixel Accuracy曲线
![pix_acc.png](output_dir%2Fpix_acc.png)
#### 2. MIoU曲线
![miou.png](output_dir%2Fmiou.png)
#### 3. Loss曲线
![loss.png](output_dir%2Floss.png)
#### 4. 学习率曲线
![learning_rate.png](output_dir%2Flearning_rate.png)
#### 5. onnx模型结构(简化后)
![onnx.png](output_dir%2Fonnx.png)