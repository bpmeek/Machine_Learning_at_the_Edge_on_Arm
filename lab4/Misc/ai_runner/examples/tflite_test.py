###################################################################################
#   Copyright (c) 2021 STMicroelectronics.
#   All rights reserved.
#   This software is licensed under terms that can be found in the LICENSE file in
#   the root directory of this software component.
#   If no LICENSE file comes with this software, it is provided AS-IS.
###################################################################################

import sys
import os
import argparse
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

from stm_ai_runner import AiRunner

_DEFAULT = 'serial'


def dequantize(q, scale, zp):
    """Dequantize the array"""
    return (q.astype(np.float32) - np.float32(zp)) * np.float32(scale)


def stat_to_str(val):
    if val.dtype == np.float32:
        str = 'min={:.9f}, max={:.9f}, mean={:.1f}, std={:.1f}'.format(
            np.min(val), np.max(val), np.mean(val), np.std(val))
    else:
        str = 'min={}, max={}, mean={:.1f}, std={:.1f}'.format(
            np.min(val), np.max(val), np.mean(val), np.std(val))
    return str


def rmse(ref, pred):
    """Return Root Mean Squared Error (RMSE)."""
    return np.sqrt(((ref.astype(np.float32) - pred.astype(np.float32)) ** 2).mean())


def l2r(ref, pred):
  """Compute L2 relative error"""
  def magnitude(v):
    return np.sqrt(np.sum(np.square(v).flatten()))
  mag = magnitude(pred.astype(np.float32)) + np.finfo(np.float32).eps
  return magnitude((ref.astype(np.float32) - pred.astype(np.float32))) / mag


def generate_metrics(ref, pred, scale=1.0, zp=0, desc=None, plot=False):
    """Generate and display metrics"""

    assert ref.dtype == pred.dtype

    desc = '[{}]'.format(str(desc)) if desc else '[]'
    b_size = ref.shape[0]

    ref = ref.flatten()
    pred = pred.flatten()

    if ref.dtype != np.float32:
        # integer data type
        ref_f = dequantize(ref, scale, zp)
        pred_f = dequantize(pred, scale, zp)
        rmse_f = rmse(ref_f, pred_f)
        l2r_f = l2r(ref_f, pred_f)
        d = ref.astype(np.int32) - pred.astype(np.int32)

        # histogram
        err = []
        hist = {}
        for i in range(len(d)):
            if np.abs(d[i]) > 0:
                # err_ = np.abs(d[i])
                err_ = d[i]
                err.extend([err_])
                hist[err_] = hist.get(err_, 0) + 1
    else:
        # float32 data type
        d = ref - pred
        rmse_f = rmse(ref, pred)
        l2r_f = l2r(ref, pred)       

    print('')
    print(f'{desc} - {b_size} samples, {ref.dtype}, {ref.size} items')
    print('-' * 80)
    if ref.dtype != np.float32:
        print(f' diff     : {len(err)}/{len(ref)}, hist={hist}')
    print(' ref      : {}'.format(stat_to_str(ref)))
    print(' pred     : {}'.format(stat_to_str(pred)))
    print(' diff     : {}'.format(stat_to_str(d)))
    if ref.dtype != np.float32:
        print(' metrics  : rmse={:.9f} l2r={:.9f} (quant.)'.format(rmse(ref, pred), l2r(ref, pred)))
        print(' metrics  : rmse={:.9f} l2r={:.9f} (dequant.)'.format(rmse_f, l2r_f), flush=True)
    else:
        print(' metrics  : rmse={:.9f} l2r={:.9f}'.format(rmse_f, l2r_f), flush=True)

    if ref.dtype == np.float32:
        return

    if not len(err) or not plot:
        return

    import matplotlib.pyplot as plt

    # display histogram
    bins = [b for b in range(min(err), max(err)+2)]
    plt.style.use('ggplot')
    plt.hist(np.array(err), bins=bins, density=False, align='left', rwidth=0.8)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(f'Diff. histogram: {len(err)}/{len(d)} items')
    plt.show()

    return


def tf_create(f_path, verbosity):

    if not f_path:
        return None

    print('\nCreate the interpreter for the "{}" model'.format(f_path))

    tf_interpreter = tf.lite.Interpreter(model_path=f_path)
    tf_interpreter.allocate_tensors()

    input_details = tf_interpreter.get_input_details()
    output_details = tf_interpreter.get_output_details()

    if verbosity:
        def _tens_to_str(val):
            ext_ = ''
            if val.get('quantization', None):
                ext_ = ', scale={}, zp={}'.format(val['quantization'][0], val['quantization'][1])
            return "{}, {}{}".format(tuple(val['shape']), val['dtype'], ext_)

        print('-' * 80)
        print("{:20s} : {}/{}".format('inputs/outputs', len(input_details), len(output_details)))
        for tens in input_details:
            print("{:20s} : {}".format(tens['name'], _tens_to_str(tens)))
        for tens in output_details:
            print("{:20s} : {}".format(tens['name'], _tens_to_str(tens)))
        print('-' * 80)

    return tf_interpreter


def tf_run(tf_interpreter, inputs):

    assert len(tf_interpreter.get_input_details()) == 1

    input_details = tf_interpreter.get_input_details()
    output_details = tf_interpreter.get_output_details()
    batch_size = inputs[0].shape[0]

    # align the shape of the inputs (c-model is always - (b, h, w, c))
    inputs[0] = inputs[0].reshape((-1,) +\
        tuple(input_details[0]['shape'][1:]))

    for batch in range(batch_size):
        input = np.expand_dims(inputs[0][batch], axis=0)
        tf_interpreter.set_tensor(input_details[0]['index'], input)
        tf_interpreter.invoke()
        output = tf_interpreter.get_tensor(output_details[0]['index'])
        if batch == 0:
            outputs = [output]
        else:
            outputs[0] = np.append(outputs[0], output, axis=0)
    return outputs


def load_data(f_npz):

    val_data = np.load(f_npz)
    return [val_data['m_inputs_1']], [val_data['c_outputs_1']]


def test(args):

    # create the interpreters
    tf_interpreter = tf_create(args.model, args.verbosity)

    ai_runner = AiRunner(debug=args.debug)
    if not ai_runner.connect(args.desc):
        return 1

    ai_runner.summary(level=args.verbosity)

    # check input/output signatures
    assert len(tf_interpreter.get_input_details()) == 1
    assert tf_interpreter.get_input_details()[0]['dtype'] == ai_runner.get_input_infos()[0]['type']
    # assert tuple(tf_interpreter.get_input_details()[0]['shape']) == ai_runner.get_input_infos()[0]['shape']

    # generate input data
    if args.npz:
        inputs, _ = load_data(args.npz)
        inputs[0] = inputs[0].reshape((-1,) + ai_runner.get_input_infos()[0]['shape'][1:])
    else:
        inputs = ai_runner.generate_rnd_inputs(batch_size=args.batch)

    # run the interpreters
    print('\nRunning the interpreters...', flush=True)

    outputs, _ = ai_runner.invoke(inputs)
    tf_outputs = tf_run(tf_interpreter, inputs)

    # Compare results
    print('\nComparing results...', flush=True)

    for i, (tf_res, res) in enumerate(zip(tf_outputs, outputs)):
        tens = ai_runner.get_output_infos()[i]
        if tf_res.dtype == np.float32:
            desc = '{}'.format(tens['name'])
            generate_metrics(tf_res, res, desc=desc)
            np.testing.assert_almost_equal(tf_res, res.reshape(tf_res.shape), decimal=6)
        else:
            desc = '{}, scale={}, zp={}'.format(tens['name'], tens['scale'], tens['zero_point'])
            generate_metrics(tf_res, res, scale=tens['scale'], zp=tens['zero_point'], desc=desc, plot=args.plot)


def main():
    """ script entry point """

    parser = argparse.ArgumentParser(description='Utility to test a generated TFlite model')
    parser.add_argument('--model', '-m', metavar='STR', type=str, help='TFLite model file',
                        default=None)
    parser.add_argument('--desc', '-d', metavar='STR', type=str, help='description for STM AI connection',
                        default=_DEFAULT)
    parser.add_argument('--batch', '-b', metavar='INT', type=int, help='batch_size', default=2)
    parser.add_argument('--verbosity', '-v',
                        nargs='?', const=1,
                        type=int, choices=range(0, 3),
                        help="set verbosity level",
                        default=0)

    parser.add_argument('--npz', metavar='STR', type=str, help='NPZ file', default=None)

    parser.add_argument('--debug', action='store_true', help="debug option")
    parser.add_argument('--plot', action='store_true', help="plot option")
    args = parser.parse_args()

    return test(args)


if __name__ == '__main__':
    sys.exit(main())

