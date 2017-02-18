#
# Noisy Image
#

#std 8
python main.py --epoch 30 --blocks 5 --fc_dim 512 --noise 1 --n_std 8 -c chkpt/blk5_fc512_b64/noisy_image/n_std_8 -o result/blk5_fc512_b64/noisy_image/n_std_8
python main.py --epoch 30 --blocks 5 --fc_dim 512 --noise 1 --n_std 8 -c chkpt/blk5_fc512_b64/noisy_image/n_std_8 -o result/blk5_fc512_b64/noisy_image/n_std_8 --apply

#std 32
python main.py --epoch 30 --blocks 5 --fc_dim 512 --noise 1 --n_std 32 -c chkpt/blk5_fc512_b64/noisy_image/n_std_32 -o result/blk5_fc512_b64/noisy_image/n_std_32
python main.py --epoch 30 --blocks 5 --fc_dim 512 --noise 1 --n_std 32 -c chkpt/blk5_fc512_b64/noisy_image/n_std_32 -o result/blk5_fc512_b64/noisy_image/n_std_32 --apply

#std 128
python main.py --epoch 30 --blocks 5 --fc_dim 512 --noise 1 --n_std 32 -c chkpt/blk5_fc512_b64/noisy_image/n_std_128 -o result/blk5_fc512_b64/noisy_image/n_std_128
python main.py --epoch 30 --blocks 5 --fc_dim 512 --noise 1 --n_std 32 -c chkpt/blk5_fc512_b64/noisy_image/n_std_128 -o result/blk5_fc512_b64/noisy_image/n_std_128 --apply

#
# Noisy Label
#

#percent 0.05
python main.py --epoch 30 --blocks 5 --fc_dim 512 --noise 2 --n_percent 0.05 -c chkpt/blk5_fc512_b64/noisy_label/n_percent_5 -o result/blk5_fc512_b64//noisy_label/n_percent_5
python main.py --epoch 30 --blocks 5 --fc_dim 512 --noise 2 --n_percent 0.05 -c chkpt/blk5_fc512_b64/noisy_label/n_percent_5 -o result/blk5_fc512_b64//noisy_label/n_percent_5 --apply

#percent 0.15
python main.py --epoch 30 --blocks 5 --fc_dim 512 --noise 2 --n_percent 0.15 -c chkpt/blk5_fc512_b64/noisy_label/n_percent_15 -o result/blk5_fc512_b64//noisy_label/n_percent_15
python main.py --epoch 30 --blocks 5 --fc_dim 512 --noise 2 --n_percent 0.15 -c chkpt/blk5_fc512_b64/noisy_label/n_percent_15 -o result/blk5_fc512_b64//noisy_label/n_percent_15 --apply

#percent 0.50
python main.py --epoch 30 --blocks 5 --fc_dim 512 --noise 2 --n_percent 0.50 -c chkpt/blk5_fc512_b64/noisy_label/n_percent_50 -o result/blk5_fc512_b64//noisy_label/n_percent_50
python main.py --epoch 30 --blocks 5 --fc_dim 512 --noise 2 --n_percent 0.50 -c chkpt/blk5_fc512_b64/noisy_label/n_percent_50 -o result/blk5_fc512_b64//noisy_label/n_percent_50 --apply
