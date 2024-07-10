###################################################################################
#   Copyright (c) 2021 STMicroelectronics.
#   All rights reserved.
#   This software is licensed under terms that can be found in the LICENSE file in
#   the root directory of this software component.
#   If no LICENSE file comes with this software, it is provided AS-IS.
###################################################################################
"""
Typical ai_runner example
"""
import sys
import argparse
import numpy as np

from stm_ai_runner import AiRunner

_DEFAULT = 'stm32ai_ws/'


def run(args):
    """Processing function"""  # noqa: DAR101,DAR201,DAR401
    mode = AiRunner.Mode.IO_ONLY

    if args.mode == 'per_layer':
        mode = AiRunner.Mode.PER_LAYER
    elif args.mode == 'per_layer_with_data':
        mode = AiRunner.Mode.PER_LAYER_WITH_DATA

    print('Opening stm.ai runtime "{}"..'.format(args.desc), flush=True)

    runner = AiRunner(debug=args.debug)
    runner.connect(args.desc, reload=False)

    if not runner.is_connected:
        print('ERR: connection to stm.ai runtime has failed..')
        print(' {}'.format(runner.get_error()))
        return 1

    print(runner, flush=True)
    if args.verbosity:
        print(runner.get_info())

    c_name = runner.names[0] if not args.name else args.name

    if c_name not in runner.names:
        print('ERR: c-model "{}" is not available'.format(c_name))
        return 1

    session = runner.session(c_name)
    if args.verbosity:
        print(session.get_input_infos())
        print(session.get_output_infos())
        print('session = ', session, flush=True)

    session.summary()

    print('Running c-model "{}" with random data (b={})..'.format(c_name, args.batch), flush=True)

    inputs = runner.generate_rnd_inputs(session.name, batch_size=args.batch)

    outputs, profiler = session.invoke(inputs, mode=mode)

    if args.debug:
        print(profiler)

    print('\nReport summary - total execution time : {:.02f}ms'.format(profiler['debug']['host_duration']))
    print(' nb samples            : {}'.format(len(profiler['c_durations'])))
    print(' duration by sample    : {:.03f}ms'.format(np.array(profiler['c_durations']).mean()))
    print(' rt duration by sample : {:.03f}ms'.format(np.array(profiler['debug']['exec_times']).mean()))

    # display profiling info by c-node if available
    if profiler['c_nodes']:
        sum_ = 0.0
        print(' nb nodes              : {}'.format(len(profiler['c_nodes'])))
        print(' duration per node/sample :')
        print(' ' + '-' * 80)
        print(' {:<4s} {:<4s} {:<30s} {:>8s}  {}'.format('c_id', 'm_id', 'type', 'dur(ms)', 'outputs'))
        print(' ' + '-' * 80)
        for i, c_node in enumerate(profiler['c_nodes']):
            dur_ = np.array(c_node['c_durations']).mean()
            ext_ = ''
            if c_node['data'] is not None:
                for d_idx, data in enumerate(c_node['data']):
                    szp = ''
                    if c_node['scale'] and c_node['scale'][d_idx]:
                        szp = 's={:06f} zp={}'.format(c_node['scale'][d_idx], c_node['zero_point'][d_idx])
                    ext_ += '{} {} {}'.format(data.shape, data.dtype, szp)
                    if d_idx != len(c_node['data']) - 1:
                        ext_ += '\n' + ' ' * 52
            print(' {:<4d} {:<4d} {:<30s} {:>8.03f}  {}'.format(
                i,
                c_node['m_id'],
                c_node['layer_desc'],
                dur_, ext_))
            sum_ += dur_
        print(' ' + '-' * 80)
        print(' {:>40s} {:>8.03f}ms'.format('total :', sum_))

    # display the i/o data (including the internal features if available)
    if args.data:
        print('')
        for idx in range(inputs[0].shape[0]):
            print('- sample {} {}'.format(idx, '-' * 40))
            for j, data in enumerate(inputs):
                print(' i[{}]{}: {}'.format(j, data[idx].shape, data[idx].flatten()))
            for k, c_node in enumerate(profiler['c_nodes']):
                if c_node['data'] is not None:
                    for data_idx, data in enumerate(c_node['data']):
                        if data.size:
                            print('  n[{}.{}]{}: {}'.format(k, data_idx, data[idx].shape, data[idx].flatten()))

            for j, data in enumerate(outputs):
                print(' o[{}]{}: {}'.format(j, data[idx].shape, data[idx].flatten()))

    runner.disconnect()

    return 0


def main():
    """Main function to parse the arguments"""  # noqa: DAR101,DAR201,DAR401
    parser = argparse.ArgumentParser(description='AI runner')
    parser.add_argument('--desc', '-d', metavar='STR', type=str, help='description', default=_DEFAULT)
    parser.add_argument('--batch', '-b', metavar='INT', type=int, help='batch_size', default=2)
    parser.add_argument('--name', '-n', metavar='STR', type=str, help='c-model name', default=None)
    parser.add_argument('--mode', '-m', type=str, help='mode', default='io_only',
                        choices=['io_only', 'per_layer', 'per_layer_with_data'])
    parser.add_argument('--verbosity', '-v',
                        nargs='?', const=1,
                        type=int, choices=range(0, 3),
                        help="set verbosity level",
                        default=0)
    parser.add_argument('--debug', action='store_true', help="debug option")
    parser.add_argument('--data', action='store_true', help="show the data")
    args = parser.parse_args()

    return run(args)


if __name__ == '__main__':
    sys.exit(main())
