import os
import pprint
import argparse
import torch
import pickle
import utils
import logging
import sys

from options import *
from model.ARWGAN import ARWGAN
from noise_layers.noiser import Noiser
from noise_argparser import NoiseArgParser

from train import train


def main():
    # 设置CUDA环境变量，指定只使用ID为0的GPU设备。
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # 优先使用GPU进行计算，如果没有可用的GPU，则降级使用CPU。
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # 创建一个根解析器，然后添加“new”和“continue”这两个子命令解析器。
    parent_parser = argparse.ArgumentParser(description='Training of ARWGAN nets')
    subparsers = parent_parser.add_subparsers(dest='command', help='Sub-parser for commands')

    new_run_parser = subparsers.add_parser('new', help='starts a new run')
    new_run_parser.add_argument('--data-dir', '-d', default='D:\\workspace\\watermark\\DataSet\\COCO\\data', type=str,
                                help='The directory where the data is stored.')
    new_run_parser.add_argument('--batch-size', '-b', default=12, type=int, help='The batch size.')
    new_run_parser.add_argument('--epochs', '-e', default=1, type=int, help='Number of epochs to run the simulation.')
    new_run_parser.add_argument('--name', required=True, type=str, help='The name of the experiment.')

    new_run_parser.add_argument('--size', '-s', default=128, type=int,
                                help='The size of the images (images are square so this is height and width).')
    new_run_parser.add_argument('--message', '-m', default=30, type=int, help='The length in bits of the watermark.')
    new_run_parser.add_argument('--continue-from-folder', '-c', default='', type=str,
                                help='The folder from where to continue a previous run. Leave blank if you are starting a new experiment.')
    # parser.add_argument('--tensorboard', dest='tensorboard', action='store_true',
    #                     help='If specified, use adds a Tensorboard log. On by default')
    new_run_parser.add_argument('--tensorboard', action='store_true',
                                help='Use to switch on Tensorboard logging.')
    new_run_parser.add_argument('--enable-fp16', dest='enable_fp16', action='store_true',
                                help='Enable mixed-precision training.')

    new_run_parser.add_argument('--noise', nargs='*', action=NoiseArgParser,
                                help="Noise layers configuration. Use quotes when specifying configuration, e.g. 'cropout((0.55, 0.6), (0.55, 0.6))'")

    new_run_parser.set_defaults(tensorboard=False)
    new_run_parser.set_defaults(enable_fp16=False)

    continue_parser = subparsers.add_parser('continue', help='Continue a previous run')
    continue_parser.add_argument('--folder', '-f', required=True, type=str,
                                 help='Continue from the last checkpoint in this folder.')
    continue_parser.add_argument('--data-dir', '-d', required=False, type=str,
                                 help='The directory where the data is stored. Specify a value only if you want to override the previous value.')
    continue_parser.add_argument('--epochs', '-e', required=False, type=int,
                                 help='Number of epochs to run the simulation. Specify a value only if you want to override the previous value.')
    # continue_parser.add_argument('--tensorboard', action='store_true',
    #                             help='Override the previous setting regarding tensorboard logging.')

    # 解析命令行输入的参数
    args = parent_parser.parse_args()
    checkpoint = None
    loaded_checkpoint_file_name = None

    if args.command == 'continue':
        this_run_folder = args.folder
        options_file = os.path.join(this_run_folder, 'options-and-config.pickle')
        train_options, net_config, noise_config = utils.load_options(options_file)
        checkpoint, loaded_checkpoint_file_name = utils.load_last_checkpoint(
            os.path.join(this_run_folder, 'checkpoints'))
        train_options.start_epoch = checkpoint['epoch'] + 1
        if args.data_dir is not None:
            train_options.train_folder = os.path.join(args.data_dir, 'train')
            train_options.validation_folder = os.path.join(args.data_dir, 'val')
        if args.epochs is not None:
            if train_options.start_epoch < args.epochs:
                train_options.number_of_epochs = args.epochs
            else:
                print(f'Command-line specifies of number of epochs = {args.epochs}, but folder={args.folder} '
                      f'already contains checkpoint for epoch = {train_options.start_epoch}.')
                exit(1)

    else:
        assert args.command == 'new'
        start_epoch = 1
        train_options = TrainingOptions(
            batch_size=args.batch_size,
            number_of_epochs=args.epochs,
            train_folder=os.path.join(args.data_dir, 'train'),
            validation_folder=os.path.join(args.data_dir, 'val'),
            runs_folder=os.path.join('.', 'runs'),
            start_epoch=start_epoch,
            experiment_name=args.name)

        noise_config = args.noise if args.noise is not None else []
        net_config = HiDDenConfiguration(H=args.size, W=args.size,
                                         message_length=args.message,
                                         encoder_blocks=4, encoder_channels=64,
                                         decoder_blocks=7, decoder_channels=64,
                                         use_discriminator=True,
                                         use_vgg=False,
                                         discriminator_blocks=3, discriminator_channels=64,
                                         decoder_loss=1,
                                         encoder_loss=0.7,
                                         adversarial_loss=1e-3,
                                         enable_fp16=args.enable_fp16
                                         )

        this_run_folder = utils.create_folder_for_run(train_options.runs_folder, args.name)
        # 打开options-and-config.pickle文件，将 train_options、noise_config 和 net_config 序列化为二进制形式保存到这个文件中
        with open(os.path.join(this_run_folder, 'options-and-config.pickle'), 'wb+') as f:
            pickle.dump(train_options, f)
            pickle.dump(noise_config, f)
            pickle.dump(net_config, f)

    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        handlers=[
                            logging.FileHandler(os.path.join(this_run_folder, f'{train_options.experiment_name}.log')),
                            logging.StreamHandler(sys.stdout)
                        ])
    if (args.command == 'new' and args.tensorboard) or \
            (args.command == 'continue' and os.path.isdir(os.path.join(this_run_folder, 'tb-logs'))):
        logging.info('Tensorboard is enabled. Creating logger.')
        from tensorboard_logger import TensorBoardLogger
        tb_logger = TensorBoardLogger(os.path.join(this_run_folder, 'tb-logs'))
    else:
        tb_logger = None

    noiser = Noiser(noise_config, device)
    model = ARWGAN(net_config, device, noiser, tb_logger)

    if args.command == 'continue':
        # if we are continuing, we have to load the model params
        assert checkpoint is not None
        logging.info(f'Loading checkpoint from file {loaded_checkpoint_file_name}')
        utils.model_from_checkpoint(model, checkpoint)

    logging.info('ARWGAN model: {}\n'.format(model.to_stirng()))
    logging.info('Model Configuration:\n')
    logging.info(pprint.pformat(vars(net_config)))
    logging.info('\nNoise configuration:\n')
    logging.info(pprint.pformat(str(noise_config)))
    logging.info('\nTraining train_options:\n')
    logging.info(pprint.pformat(vars(train_options)))

    train(model, device, net_config, train_options, this_run_folder, tb_logger)


if __name__ == '__main__':
    main()
