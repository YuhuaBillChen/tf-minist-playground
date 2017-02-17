from tf_model import *
import argparse

DEFAULT_CHKDIR = "./chkpt/";
DEFAULT_OUTDIR = "./result/";
DEFAULT_BATCH_SIZE = 64;


class Main(object):
    def __init__(self):
        self.args = self.parameter_setting();
        self.model = MNISTModel.BaseModel(chkpt_dir=self.args.chkdir, batch_size=self.args.batch, noise=self.args.noise,
                                          n_std=self.args.n_std, n_percent=self.args.n_percent);
        self.main();
        return;

    def parameter_setting(self):
        parser = argparse.ArgumentParser(prog="MNISTProgram", description="MNSIT image classification main program");
        parser.add_argument("--apply", help="Turn on if you only needs to apply/test model without training",
                            action="store_true");
        parser.add_argument("--gpu_id", help="Use the GPU with the given id, default is 0", default=0, type=int);
        parser.add_argument("--epoch", help="Number of epochs for training [25]", default=25, type=int);
        parser.add_argument("--blocks", help="Number of blocks of two shortcut convolution layers in the net[8]",
                            default=8, type=int);
        parser.add_argument("-n", "--noise", help="Type of noise to add, 0: no noise, 1: noisy image, 2: noisy label",
                            default=0, choices=xrange(0, 3), type=int);
        parser.add_argument("--n_std", help="Standard deviation of image noise[8]. Only affect when noise is set to 1.",
                            default=8, type=int)
        parser.add_argument("--n_percent",
                            help="Standard deviation of image noise[8]. Only affect when noise is set to 1.",
                            default=8, type=int)
        parser.add_argument("--filter_num", help="Number of filter in convnet default:[32]", default=32, type=int);
        parser.add_argument("--fc_dim", help="Dimension of first fully connected layer [256]", default=256, type=int);
        parser.add_argument("-c", "--chkdir", help="Tensorflow checkpoints folder, default:%s" % DEFAULT_CHKDIR,
                            default=DEFAULT_CHKDIR, type=str);
        parser.add_argument("-o", "--outdir", help="Output result in numpy format , default:%s" % DEFAULT_OUTDIR,
                            default=DEFAULT_OUTDIR, type=str);
        parser.add_argument("-b", "--batch", help="batch_size of training sample, default:%d" % DEFAULT_BATCH_SIZE,
                            default=DEFAULT_BATCH_SIZE, type=int)
        args = parser.parse_args();
        return args;

    def main(self):
        print self.args;  # Print out the settings
        self.model.build(h_dim=self.args.filter_num, fc_dim=self.args.fc_dim, block_num=self.args.blocks);
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True;
        config.allow_soft_placement = True;
        print "Main process started"
        with tf.device("/gpu:%d" % self.args.gpu_id):
            with tf.Session(config=config) as sess:
                if not self.args.apply:
                    self.model.train(sess=sess, epoch_num=self.args.epoch);
                else:
                    if not os.path.exists(self.args.outdir):
                        os.makedirs(self.args.outdir);
                    [pred_labels, real_labels] = self.model.test(sess=sess);
                    np.save(os.path.join(self.args.outdir, "result"),
                            np.array([pred_labels, real_labels], dtype=np.float));
            sess.close();
        exit(0);


if __name__ == "__main__":
    m_main = Main();
    m_main.main();
