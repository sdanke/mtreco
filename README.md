# Manga text recognization
本项目为漫画文本检测、识别、翻译系统的文本识别部分。

## Dependency
- Python3.6.10, PyTorch 1.7.1, CUDA 10.1等
- 安装: ```pip install -r requirements.txt```

## Corpus
- 下载[corpus.zip](https://drive.google.com/file/d/19FN_u11TUfrg8GjCUHyRQPaWC78SZU2r/view?usp=sharing)解压到data目录下

## Synthetic Image Generation
- 字体可用性检查：```tools/check_font.py```
- 图片生成：```tools/generate_line_images.py```
- 数据集分割：```tools/split_data.py```

## Train
- 编译[Deformable-ConvNets-V2](https://github.com/sdanke/deform_conv_v2), 替换```ocr/net/dcn/deform_conv_v2.dll```
- 训练：```python train.py --exp_name cafcn --train_data path/to/train/ids --valid_data path/to/val/ids --batch_size 24 --lr_scheduler_config config/cafcn_cyclic_config.json --log_interval 1  --num_workers 0 --num_epochs 50 --with_cuda```

## Pretrained Model
- [Google Drive](https://drive.google.com/file/d/1oGtpZbsAFAQc1OfP4-0qgkTW1KZHMyNK/view?usp=sharing)
accuracy: 0.96