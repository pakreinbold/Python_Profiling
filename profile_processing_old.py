import pandas as pd


def memprof_to_df(file_name):
    '''
    Converts a text file containing the output of a memory profiler run into a
    pandas DataFrame. To get the text file, run something like

    python -m memory_profiler your_script.py > memory_profile.txt

    Args:
        file_name (str): path to the txt file containing the memory profile

    Returns:
        pandas DataFrame: indexed by line number, the columns contain the
            cumulative memory usage, the incremental memory usage, and the
            number of occurences.
    '''

    # Open the stored output
    with open(file_name) as text_file:
        text = text_file.read()

    # Get rid of nonsense
    text_ = text.replace('\x00', '').split()

    # Use every other instance of 'MiB' to identify rows
    MiB_locations = [x for x, y in enumerate(text_) if y == 'MiB'][1::2]

    # Fill the rows
    rows = [[text_[ind-4], text_[ind-3] + ' MiB',
             text_[ind-1] + ' MiB', text_[ind+1]]
            for ind in MiB_locations]

    # Put it into a DataFrame
    headers = ['line', 'total_memory', 'increment', 'occurences']
    df = pd.DataFrame(rows, columns=headers).set_index('line')

    return df


def memprof_to_df_content(file_name):
    '''
    Converts a text file containing the output of a memory profiler run into a
    pandas DataFrame. To get the text file, run something like

    python -m memory_profiler your_script.py > memory_profile.txt

    Args:
        file_name (str): path to the txt file containing the memory profile

    Returns:
        pandas DataFrame: indexed by line number, the columns contain the
            cumulative memory usage, the incremental memory usage, and the
            number of occurences.
    '''

    # Open the stored output
    with open(file_name) as text_file:
        text = text_file.read()

    # Get rid of nonsense
    text_ = text.replace('\x00', '').split()

    # Use every other instance of 'MiB' to identify rows
    MiB_locations = [x for x, y in enumerate(text_) if y == 'MiB'][1::2]

    # Fill the rows
    rows = []
    for n in range(len(MiB_locations)):
        ind = MiB_locations[n]
        row = [text_[ind-4], text_[ind-3] + ' MiB',
               text_[ind-1] + ' MiB', text_[ind+1]]
        if n == len(MiB_locations) - 1:
            content = ' '.join(text_[ind+2:])
        else:
            next_ind = MiB_locations[n+1]
            content_parts = text_[ind+2:next_ind-4]
            last_part = content_parts[-1]
            try:
                int(last_part)
                if int(last_part) == int(row[0]) + 1:
                    content_parts.pop()
            except Exception:
                pass
            content = ' '.join(content_parts)
        row.append(content)
        rows.append(row)

    # Put it into a DataFrame
    headers = ['line', 'total_memory', 'increment', 'occurences', 'content']
    df = pd.DataFrame(rows, columns=headers).set_index('line')

    return df
