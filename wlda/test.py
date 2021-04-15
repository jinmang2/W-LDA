import argparse


parser = argparse.ArgumentParser(description='Training WAE in MXNet')
parser.add_argument('-dom', '--domain', type=str,
                    default='twenty_news', help='domain to run', required=False)
parser.add_argument('-data', '--data_path', type=str,
                    default='', help='file path for dataset', required=False)
parser.add_argument('-max_labels', '--max_labels', type=int, default=100,
                    help='max number of topics to specify as labels for a single training document', required=False)
parser.add_argument('-max_labeled_samples', '--max_labeled_samples', type=int,
                    default=10, help='max number of labeled samples per topic', required=False)
parser.add_argument('-label_seed', '--label_seed', type=lambda x: int(x) if x != 'None' else None,
                    default=None, help='random seed for subsampling the labeled dataset', required=False)
parser.add_argument('-mod', '--model', type=str,
                    default='dirichlet', help='model to use', required=False)
parser.add_argument('-desc', '--description', type=str, default='',
                    help='description for the experiment', required=False)
parser.add_argument('-alg', '--algorithm', type=str, default='standard',
                    help='algorithm to use for training: standard', required=False)
parser.add_argument('-bs', '--batch_size', type=int, default=256,
                    help='batch_size for training', required=False)
parser.add_argument('-opt', '--optim', type=str, default='Adam',
                    help='encoder training algorithm', required=False)
parser.add_argument('-lr', '--learning_rate', type=float,
                    default=1e-4, help='learning rate', required=False)
parser.add_argument('-l2', '--weight_decay', type=float,
                    default=0., help='weight decay', required=False)
parser.add_argument('-e_nh', '--enc_n_hidden', type=int, nargs='+', default=[
    128], help='# of hidden units for encoder or list of hiddens for each layer', required=False)
parser.add_argument('-e_nl', '--enc_n_layer', type=int, default=1,
                    help='# of hidden layers for encoder, set to -1 if passing list of n_hiddens', required=False)
parser.add_argument('-e_nonlin', '--enc_nonlinearity', type=str,
                    default='sigmoid', help='type of nonlinearity for encoder', required=False)
parser.add_argument('-e_weights', '--enc_weights', type=str,
                    default='', help='file path for encoder weights', required=False)
parser.add_argument('-e_freeze', '--enc_freeze', type=lambda x: (str(x).lower() == 'true'),
                    default=False, help='whether to freeze the encoder weights', required=False)
parser.add_argument('-lat_nonlin', '--latent_nonlinearity', type=str,
                    default='', help='type of to use prior to decoder', required=False)
parser.add_argument('-d_nh', '--dec_n_hidden', type=int, nargs='+', default=[
    128], help='# of hidden units for decoder or list of hiddens for each layer', required=False)
parser.add_argument('-d_nl', '--dec_n_layer', type=int, default=0,
                    help='# of hidden layers for decoder', required=False)
parser.add_argument('-d_nonlin', '--dec_nonlinearity', type=str,
                    default='', help='type of nonlinearity for decoder', required=False)
parser.add_argument('-d_weights', '--dec_weights', type=str,
                    default='', help='file path for decoder weights', required=False)
parser.add_argument('-d_freeze', '--dec_freeze', type=lambda x: (str(x).lower() == 'true'),
                    default=False, help='whether to freeze the decoder weights', required=False)
parser.add_argument('-d_word_dist', '--dec_word_dist', type=lambda x: (str(x).lower() == 'true'), default=False,
                    help='whether to init decoder weights with training set word distributions', required=False)
parser.add_argument('-dis_nh', '--dis_n_hidden', type=int, nargs='+', default=[
    128], help='# of hidden units for encoder or list of hiddens for each layer', required=False)
parser.add_argument('-dis_nl', '--dis_n_layer', type=int, default=1,
                    help='# of hidden layers for encoder, set to -1 if passing list of n_hiddens', required=False)
parser.add_argument('-dis_nonlin', '--dis_nonlinearity', type=str, default='sigmoid',
                    help='type of nonlinearity for discriminator', required=False)
parser.add_argument('-dis_y_weights', '--dis_y_weights', type=str, default='',
                    help='file path for discriminator_y weights', required=False)
parser.add_argument('-dis_z_weights', '--dis_z_weights', type=str, default='',
                    help='file path for discriminator_z weights', required=False)
parser.add_argument('-dis_freeze', '--dis_freeze', type=lambda x: (str(x).lower() == 'true'),
                    default=False, help='whether to freeze the encoder weights', required=False)
parser.add_argument('-include_w', '--include_weights', type=str, nargs='*', default=[],
                    help='weights to train on (default is all weights) -- all others are kept fixed; Ex: E.z_encoder D.decoder', required=False)
parser.add_argument('-eps', '--epsilon', type=float, default=1e-8,
                    help='epsilon param for Adam', required=False)
parser.add_argument('-mx_it', '--max_iter', type=int, default=50001,
                    help='max # of training iterations', required=False)
parser.add_argument('-train_stats_every', '--train_stats_every', type=int, default=100,
                    help='skip train_stats_every iterations between recording training stats', required=False)
parser.add_argument('-eval_stats_every', '--eval_stats_every', type=int, default=100,
                    help='skip eval_stats_every iterations between recording evaluation stats', required=False)
parser.add_argument('-ndim_y', '--ndim_y', type=int, default=256,
                    help='dimensionality of y - topic indicator', required=False)
parser.add_argument('-ndim_x', '--ndim_x', type=int, default=2,
                    help='dimensionality of p(x) - data distribution', required=False)
parser.add_argument('-saveto', '--saveto', type=str, default='',
                    help='path prefix for saving results', required=False)
parser.add_argument('-gpu', '--gpu', type=int, default=-2,
                    help='if/which gpu to use (-1: all, -2: None)', required=False)
parser.add_argument('-hybrid', '--hybridize', type=lambda x: (str(x).lower() == 'true'),
                    default=False, help='declaritive True (hybridize) or imperative False', required=False)
parser.add_argument('-full_npmi', '--full_npmi', type=lambda x: (str(x).lower() == 'true'),
                    default=False, help='whether to compute NPMI for full trajectory', required=False)
parser.add_argument('-eot', '--eval_on_test', type=lambda x: (str(x).lower() == 'true'), default=False,
                    help='whether to evaluate on the test set (True) or validation set (False)', required=False)
parser.add_argument('-verb', '--verbose', type=lambda x: (str(x).lower() == 'true'),
                    default=True, help='whether to print progress to stdout', required=False)
parser.add_argument('-dirich_alpha', '--dirich_alpha', type=float,
                    default=1e-1, help='param for Dirichlet prior', required=False)
parser.add_argument('-adverse', '--adverse', type=lambda x: (str(x).lower() == 'true'), default=True,
                    help='whether to turn on adverserial training (MMD or GAN). set to False if only train auto-encoder', required=False)
parser.add_argument('-update_enc', '--update_enc', type=lambda x: (str(x).lower() == 'true'),
                    default=True, help='whether to update encoder for unlabed_train_op()', required=False)
parser.add_argument('-labeled_loss_lambda', '--labeled_loss_lambda', type=float,
                    default=1.0, help='param for Dirichlet noise for label', required=False)
parser.add_argument('-train_mode', '--train_mode', type=str,
                    default='mmd', help="set to mmd or adv (for GAN)", required=False)
parser.add_argument('-kernel_alpha', '--kernel_alpha', type=float, default=1.0,
                    help='param for information diffusion kernel', required=False)
parser.add_argument('-recon_alpha', '--recon_alpha', type=float, default=-1.0,
                    help='multiplier of the reconstruction loss when combined with mmd loss', required=False)
parser.add_argument('-recon_alpha_adapt', '--recon_alpha_adapt', type=float, default=-1.0,
                    help='adaptively change recon_alpha so that [total loss = mmd + recon_alpha_adapt * recon loss], set to -1 if no adapt', required=False)
parser.add_argument('-dropout_p', '--dropout_p', type=float, default=-
                    1.0, help='dropout probability in encoder', required=False)
parser.add_argument('-l2_alpha', '--l2_alpha', type=float, default=-1.0,
                    help='alpha multipler for L2 regularization on latent vector', required=False)
parser.add_argument('-latent_noise', '--latent_noise', type=float, default=0.0,
                    help='proportion of dirichlet noise added to the latent vector after softmax', required=False)
parser.add_argument('-topic_decoder_weight', '--topic_decoder_weight', type=lambda x: (str(x).lower() == 'true'),
                    default=False, help='extract topic words based on decoder weights or decoder outputs', required=False)
parser.add_argument('-retrain_enc_only', '--retrain_enc_only', type=lambda x: (str(x).lower() == 'true'),
                    default=False, help='only retrain the encoder for reconstruction loss', required=False)
parser.add_argument('-l2_alpha_retrain', '--l2_alpha_retrain', type=float, default=0.1,
                    help='alpha multipler for L2 regularization on encoder output during retraining', required=False)
args = vars(parser.parse_args())

print(args["enc_n_hidden"])
print(args["dec_n_hidden"])
