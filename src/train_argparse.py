import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Argument parser for your application')
    
    parser.add_argument('--output_dir', type=str, default='linear_probes/', 
                        help='Output directory')
    parser.add_argument('--probe_name', type=str, default='', 
                        help='Probe file name')
    parser.add_argument('--dataset', type=str, default='chess_data/lichess_train.pkl', 
                        help='Dataset path')
    parser.add_argument('--layer', type=int, default=6, 
                        help='Layer number')
    parser.add_argument('--batch_size', type=int, default=30, 
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, 
                        help='Learning rate')
    parser.add_argument('--wd', type=float, default=0.01, 
                        help='Weight decay')
    parser.add_argument('--betas', type=float, nargs=2, default=(0.9, 0.99), 
                        help='Betas for optimizers')
    parser.add_argument('--pos_start', type=int, default=5, 
                        help='Position start')
    parser.add_argument('--num_epochs', type=int, default=2, 
                        help='Number of epochs')
    parser.add_argument('--num_games', type=int, default=100000, 
                        help='Number of games')
    parser.add_argument('--resume', action='store_true', 
                        help='Flag to resume training')
    parser.add_argument('--log_frequency', type=int, default=100,
                        help='Number of batches between logs')
    parser.add_argument('--checkpoint_frequency', type=int, default=100,
                        help='Number of batches between logs')
    
    #TODO add mode/options arguments

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    # Add all argparse arguments to the global context
    for arg in vars(args):
        globals()[arg] = getattr(args, arg)
        
    for arg in vars(args):
        print(f'{arg}:{getattr(args,arg)}')
