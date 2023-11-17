import os
import argparse
import numpy as np
from collections import defaultdict
import torch

from predictors.helper import build_predictor

def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(description='Predict lables by ResNet with images input')

    parser.add_argument('--predictor_name', type=str, default='scene',
                        help='Name of the predictor used for analysis. (default: '
                            'scene)')
    parser.add_argument('--batch_size', type=int, default=2,
                    help='batch size of the prediction process')
    parser.add_argument('--image_path', type=str, default='samples',
                    help='batch size of the prediction process')
    parser.add_argument('--image_file', type=str, default='samples_100x256x256x3',
                    help='batch size of the prediction process')
    return parser.parse_args()

def get_batch_inputs(inputs, batch_size=None):
    """Gets inputs within mini-batch.

    This function yields at most `self.batch_size` inputs at a time.

    Args:
      inputs: Input data to form mini-batch.
      batch_size: Batch size. If not specified, `self.batch_size` will be used.
        (default: None)
    """
    total_num = inputs.shape[0]
    for i in range(0, total_num, batch_size):
        yield inputs[i:i + batch_size]

def main():
    args = parse_args()

    predictor = build_predictor(args.predictor_name)

    with np.load(os.path.join(args.image_path, args.image_file)) as data:
        images_numpy = data['arr_0']

    predictions = defaultdict(list)

    for batch_inputs in get_batch_inputs(images_numpy, args.batch_size):

        pred_outputs = predictor.easy_predict(batch_inputs)

        for pred_key, pred_val in pred_outputs.items():
            predictions[pred_key].append(pred_val)
    
    categories = np.concatenate(predictions['category'], axis=0)
    detailed_categories = {
        'score': categories,
        'name_to_idx': predictor.category_name_to_idx,
        'idx_to_name': predictor.category_idx_to_name,
    }
    np.save(os.path.join(args.image_path, 'labels', 'samples_100x256x256x3.npz'), detailed_categories)

if __name__ == '__main__':
    main()