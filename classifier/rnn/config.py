import argparse

def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', dest='project_name', type=str, default=None,
                        help='Name of Project')
    parser.add_argument('--random_seed', dest='random_seed', type=int, default=1234,
                        help='Random Seed')
    parser.add_argument('--max_voc_size', dest='max_voc_size', type=int, default=25_000,
                        help='Maximum vocabulary size')
    parser.add_argument('--emb_dim', dest='emb_dim', type=int, default=100,
                        help='Embedding size')
    parser.add_argument('--dropout', dest='dropout', type=int, default=0.5,
                        help='Dropout')
    parser.add_argument('--lstm_hid_dim', dest='lstm_hid_dim', type=int, default=256,
                        help='LSTM hidden dimension')
    parser.add_argument('--n_lstm_layers', dest='n_lstm_layers', type=int, default=2,
                        help='Number of LSTM layers')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32,
                        help='Batch Size')
    parser.add_argument('--resume_training', dest='resume_training', action='store_true',
                        default=False, help='Training from scratch or not?')
    parser.add_argument('--n_epochs', dest='n_epochs', type=int, default=1,
                        help='Number of Epochs')
    parser.add_argument('--load_model_path', dest='load_model_path', type=str, default=None,
                        help='path for loading model')
    parser.add_argument('--load_emb_path', dest='load_emb_path', type=str, default=None,
                        help='path for loading embeddings')
    parser.add_argument('--load_voc_path', dest='load_voc_path', type=str, default=None,
                        help='path for loading vocabulary')
    parser.add_argument('--save_model_path', dest='save_model_path', type=str, default=None,
                        help='path for saving model')
    parser.add_argument('--save_emb_path', dest='save_emb_path', type=str, default=None,
                        help='path for saving embeddings')
    parser.add_argument('--save_voc_path', dest='save_voc_path', type=str, default=None,
                        help='path for saving vocabulary')
    parser.add_argument('--local_run', dest='local_run', action='store_true',
                        default=False, help='Platform used: local or colab')
    parser.add_argument('--data_path', dest='data_path', type=str, default=None,
                        help='data parent directory')
    parser.add_argument('--train_file_name', dest='train_file_name', type=str, default=None,
                        help='name of train data file')
    parser.add_argument('--val_file_name', dest='val_file_name', type=str, default=None,
                        help='name of validation data file')
    parser.add_argument('--test_file_name', dest='test_file_name', type=str, default=None,
                        help='name of test data file')
    args = parser.parse_args()
    return args
