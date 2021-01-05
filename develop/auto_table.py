def generate(data, col1='**Member name**', col2='**Initial Value**', col3='**Explanation**'):
    key_len = len(col1)
    val_len = len(col2)
    desc_len = len(col3)

    # get length
    for key, val, desc in data:
        if len(key) > key_len:
            key_len = len(key)
        if len(str(val)) > val_len:
            val_len = len(str(val))
        if len(desc) > desc_len:
            desc_len = len(desc)

    # generate header
    lines = []
    lines.append(f"{'=' * key_len} {'=' * val_len} {'=' * desc_len}")
    line = f"{col1}{' ' * (key_len - 15)} " + \
           f"{col2}{' ' * (val_len - 17)} " + \
           f'{col3}'
    lines.append(line)
    lines.append(f"{'-' * key_len} {'-' * val_len} {'-' * desc_len}")

    # generate
    for i, (key, val, desc) in enumerate(data):
        line = f"{key}{' ' * (key_len - len(key))} " + \
               f"{str(val)}{' ' * (val_len - len(str(val)))} " + \
               f'{desc}'
        lines.append(line)
        if i + 1 != len(data):
            lines.append('')

    # generate ender
    lines.append(f"{'=' * key_len} {'=' * val_len} {'=' * desc_len}")

    print('\n'.join(lines))


if __name__ == '__main__':
    generate([('Vddddddddddddddddddddddddd', 0., 'Membrane potential.'),
              ('input', 0. ,'External and synaptic input current.')])

