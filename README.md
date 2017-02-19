Tensorflow MNIST Classifier
=============================
- Latest network designs
  * Convolutional layers
  * Batchnormalization layers
  * Residual networks
  * Leaky Relu activation
- Modularized Design
  * Easily add new network struture
  * Changable data IO
  * User friendly output
  * Easy expansion
  
With the default network configuration, this model can make a 99.5% correct prediction on MNIST data set.

Requirement:
- Tensorflow

Tutorial
---------------
`git clone https://github.com/YuhuaBillChen/tf-minist-playground.git`
`cd tf-minist-playground`
Training:
`python main.py`
Testing:
`python main.py --apply`

Usage
------------------------------
usage: MNISTProgram [-h] [--apply] [--gpu_id GPU_ID] [--epoch EPOCH]
                    [--blocks BLOCKS] [-n {0,1,2}] [--n_std N_STD]
                    [--n_percent N_PERCENT] [--filter_num FILTER_NUM]
                    [--fc_dim FC_DIM] [-c CHKDIR] [-o OUTDIR] [-b BATCH]

MNSIT image classification main program

optional arguments:
  -h, --help            show this help message and exit
  --apply               Turn on if you only needs to apply/test model without
                        training
  --gpu_id GPU_ID       Use the GPU with the given id, default is 0
  --epoch EPOCH         Number of epochs for training [25]
  --blocks BLOCKS       Number of blocks of two shortcut convolution layers in
                        the net[8]
  -n {0,1,2}, --noise {0,1,2}
                        Type of noise to add, 0: no noise, 1: noisy image, 2:
                        noisy label
  --n_std N_STD         Standard deviation of image noise[8]. Only affect when
                        noise is set to 1.
  --n_percent N_PERCENT
                        Percentage of mis-labels, default: [0.05]. Only affect
                        when noise is set to 2.
  --filter_num FILTER_NUM
                        Number of filter in convnet default:[32]
  --fc_dim FC_DIM       Dimension of first fully connected layer [256]
  -c CHKDIR, --chkdir CHKDIR
                        Tensorflow checkpoints folder, default:./chkpt/
  -o OUTDIR, --outdir OUTDIR
                        Output result in numpy format , default:./result/
  -b BATCH, --batch BATCH
                        batch_size of training sample, default:64
                        
Sample use case
-----------------------
Train with noise free data:
`python main.py --epoch 30 --blocks 5 --fc_dim 512 -c chkpt/noise_free -o result/noise_free`
Train with noisy images having a standard divatation of 8:
`python main.py --epoch 30 --blocks 5 --fc_dim 512  -noise 1 --n_std 8 -c chkpt/std_8 -o result/std_8`
Training with noisy labels having 5% error rate:
`python main.py --epoch 30 --blocks 5 --fc_dim 512  -noise 2 --n_percent 5 -c chkpt/percent_5 -o result/percent_5`
