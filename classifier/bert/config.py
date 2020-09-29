import argparse

def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', dest='project_name', type=str, default='question-answerability',
                        help='Name of Project')
    parser.add_argument('--random_seed', dest='random_seed', type=int, default=42,
                        help='Random Seed')
    parser.add_argument('--bert_type', dest='bert_type', type=str, default='bert-base-uncased',
                        help='The pre-trained bart model to be used')
    parser.add_argument('--model_path', dest='model_path', type=str, default=None,
                        help='Model Path if not training from scratch')
    parser.add_argument('--from_scratch', dest='from_scratch', action='store_true',
                        default=False, help='Train from scratch or not?')
    parser.add_argument('--data_path', dest='data_path', type=str, default=None,
                        help='Dataset Path')
    parser.add_argument('--output_dir', dest='output_dir', type=str, default=None,
                        help='Directory for saving model')
    parser.add_argument('--train_batch_size', dest='train_batch_size', type=int, default=8,
                        help='Training batch size')
    parser.add_argument('--test_batch_size', dest='test_batch_size', type=int, default=8,
                        help='Testing batch size')
    parser.add_argument('--max_len', dest='max_len', type=int, default=512,
                        help='Maximum input length for encoder')
    parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--adam_epsilon', dest='adam_epsilon', type=float, default=1e-8,
                        help='Adam Epsilon')
    parser.add_argument('--max_grad_norm', dest='max_grad_norm', type=float, default=1.0,
                        help='Max grad norm')
    parser.add_argument('--warmup_steps', dest='warmup_steps', type=int, default=0,
                        help='Warmup steps')
    parser.add_argument('--n_epochs', dest='n_epochs', type=int, default=1,
                        help='Number of training epochs')
    parser.add_argument('--local_test', dest='local_test', action='store_true',
                        default=False, help='Testing on local computer?')
    parser.add_argument('--model_eval', dest='model_eval', action='store_true',
                        default=False, help='Model evaluation on test data?')

    args = parser.parse_args()
    return args