def generate(data,
             col1='**Parameter**',
             col2='**Value**',
             col3='**Unit**',
             col4='**Explanation**'):
    key_len = len(col1)
    val_len = len(col2)
    unit_len = len(col3)
    desc_len = len(col4)

    fdata = []
    lines = data.strip().split('\n')
    for line in lines:
        if line.strip():
            ss = line.strip().split()
            fdata.append((ss[0], ss[1], ss[2], ' '.join(ss[3:])))

    # get length
    for key, val, unit, desc in fdata:
        if len(key) > key_len:
            key_len = len(key)
        if len(val) > val_len:
            val_len = len(val)
        if len(unit) > unit_len:
            unit_len = len(unit)
        if len(desc) > desc_len:
            desc_len = len(desc)

    # generate header
    lines = []
    lines.append(f"{'=' * key_len} {'=' * val_len} {'=' * unit_len} {'=' * desc_len}")
    line = f"{col1}{' ' * (key_len - len(col1))} " + \
           f"{col2}{' ' * (val_len - len(col2))} " + \
           f"{col3}{' ' * (unit_len - len(col3))} " + \
           f'{col4}'
    lines.append(line)
    lines.append(f"{'-' * key_len} {'-' * val_len} {'-' * unit_len} {'-' * desc_len}")

    # generate
    for i, (key, val, unit, desc) in enumerate(fdata):
        line = f"{key}{' ' * (key_len - len(key))} " + \
               f"{val}{' ' * (val_len - len(val))} " + \
               f"{unit}{' ' * (unit_len - len(unit))} " + \
               f'{desc}'
        lines.append(line)
        if i + 1 != len(fdata):
            lines.append('')

    # generate ender
    lines.append(f"{'=' * key_len} {'=' * val_len} {'=' * unit_len} {'=' * desc_len}")

    print('\n'.join(lines))


if __name__ == '__main__':
    generate('''
    Vddddddddddddddddddddddddd 0. / Membrane potential.
    input 0. / External and synaptic input current.
    ''')

