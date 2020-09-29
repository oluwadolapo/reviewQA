import argparse

def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', dest='project_name', type=str, default='review-summarizer',
                        help='Name of project')
    parser.add_argument('--random_seed', dest='random_seed', type=int, default=42,
                        help='Random Seed')
    parser.add_argument('--rnn_type', dest='rnn_type', type=str, default='rnn',
                        help='The type of rnn model to be used')
    parser.add_argument('--with_attention', dest='with_attention', action='store_true',
                        default=False, help='Attention mechanism included or not?')
    parser.add_argument('--emb_dim', dest='emb_dim', type=int, default=100,
                        help='Embedding size')
    parser.add_argument('--dropout', dest='dropout', type=int, default=0.1,
                        help='Dropout')
    parser.add_argument('--hidden_size', dest='hidden_size', type=int, default=500,
                        help='LSTM hidden dimension')
    parser.add_argument('--encoder_n_layers', dest='encoder_n_layers', type=int, default=2,
                        help='Number of LSTM encoder layers')
    parser.add_argument('--decoder_n_layers', dest='decoder_n_layers', type=int, default=2,
                        help='Number of LSTM decoder layers')
    parser.add_argument('--model_path', dest='model_path', type=str, default=None,
                        help='Model Path if not training from scratch')
    parser.add_argument('--save_encoder_path', dest='save_encoder_path', type=str, default=None,
                        help='Directory for saving model encoder')
    parser.add_argument('--save_decoder_path', dest='save_decoder_path', type=str, default=None,
                        help='Directory for saving model decoder')
    parser.add_argument('--save_vocab_path', dest='save_vocab_path', type=str, default=None,
                        help='Directory for saving vocabulary path')
    parser.add_argument('--data_path', dest='data_path', type=str, default=None,
                        help='Dataset Path')           
    parser.add_argument('--local_test', dest='local_test', action='store_true',
                        default=False, help='Testing on local computer?')
    parser.add_argument('--from_scratch', dest='from_scratch', action='store_true',
                        default=False, help='Train from scratch or not?')
    parser.add_argument('--num_epoch', dest='num_epoch', type=int, default=1,
                        help='Number of training epochs')
    parser.add_argument('--max_length', dest='max_length', type=int, default=1000,
                        help='Maximum input length for encoder')
    parser.add_argument('--teacher_forcing_ratio', dest='teacher_forcing_ratio', type=int, default=0.5,
                        help='Teacher forcing ratio')
    parser.add_argument('--max_output_length', dest='max_output_length', type=int, default=512,
                        help='Maximum output length for decoder')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--predict_batch_size', dest='predict_batch_size', type=int, default=32,
                        help='Prediction batch size')
    parser.add_argument('--lr', dest='lr', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--warmup_proportion', dest='warmup_proportion', type=float, default=0.01,
                        help='Warmup proportion')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=0.0,
                        help='Weight decay')
    parser.add_argument('--adam_epsilon', dest='adam_epsilon', type=float, default=1e-8,
                        help='Adam Epsilon')
    parser.add_argument('--max_grad_norm', dest='max_grad_norm', type=float, default=1.0,
                        help='Max grad norm')
    parser.add_argument('--gradient_accumulation_steps', dest='gradient_accumulation_steps', type=int, default=1,
                        help='gradient accumulation steps')
    parser.add_argument('--num_train_epochs', dest='num_train_epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--warmup_steps', dest='warmup_steps', type=int, default=0,
                        help='Warmup steps')
    parser.add_argument('--wait_step', dest='wait_step', type=int, default=10,
                        help='Wait step')
    parser.add_argument('--num_beams', dest='num_beams', type=int, default=4,
                        help='Number of beams')

    args = parser.parse_args()
    return args