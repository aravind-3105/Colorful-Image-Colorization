# Colorful-Image-Colorization

This is a PyTorch implementation of the paper [Colorful Image Colorization](https://arxiv.org/abs/1603.08511) by Richard Zhang, Phillip Isola, Alexei A. Efros.

## Dataset

- ImageNet - Location on ada `/share1/dataset`

## Requirements

- Python 3.6
- PyTorch 0.4.0
- Torchvision 0.2.1
- Numpy
- Matplotlib
- Pillow

## Usage

### Training

To train the model, run the following command:

```bash
python train.py --data_dir <data_dir> --batch_size <batch_size> --num_epochs <num_epochs> --learning_rate <learning_rate> --save_dir <save_dir>
```

- `data_dir`: Path to the directory containing the training images.

- `batch_size`: Batch size for training.

- `num_epochs`: Number of epochs to train for.

- `learning_rate`: Learning rate for training.

- `save_dir`: Path to the directory where the model will be saved.

### Testing

To test the model, run the following command:

```bash
python test.py --data_dir <data_dir> --model_path <model_path> --save_dir <save_dir>
```

- `data_dir`: Path to the directory containing the test images.

- `model_path`: Path to the model checkpoint.

- `save_dir`: Path to the directory where the results will be saved.

## Resources

Notion Link - [here](https://www.notion.so/aakashj2412/CV-Project-6f6e5725e29849ec8c9d9f06511b371f)
Original Source - [here](https://richzhang.github.io/colorization/)
Mid-eval Canva Link - [here](https://www.canva.com/design/DAFeeSZawPU/0RQG7aqjbvUVor1EC_NpxQ/edit?utm_content=DAFeeSZawPU&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)