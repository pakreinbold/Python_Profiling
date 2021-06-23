import pandas as pd
import re
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import argparse
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ProfileProcessor:
    '''
    Use to convert memory/line profiles from .txt documents into DataFrames
    and plot the results by line #.

    Attributes
    ----------
    memory_path, line_path : str
        path to .txt file containing memory/line profile
    memory_text, line_text : list of str
        characters in the memory/line profile
    memory_df : pd DataFrame
        contains the memory profile information
        columns: line, total memory (MiB), memory delta (MiB),
                occurences, @profile
    line_df : pd DataFrame
        contains the line profile information
        columns: line, hits, time (s), per_hit (s), perc_time (%)
    fun_locs : dict
        keys are the line numbers for the start of functions, values are the
        function names

    Methods
    -------
    __init__:
        constructs attributes
    is_float:
        checks if string can be converted to float
    is_int:
        checks if string can be converted to int
    get_dtype:
        assigns type to string in order of int, float, MiB, str
    find_pattern_matches:
        finds occurences of patterns in text objects to find locations of
        DataFrame rows
    get_mem_rows, get_line_rows:
        constructs the rows of the DataFrame based on text file structure
    get_profile_df:
        combines methods to construct DataFrame with profile info
    plot_both:
        combines DataFrames from both line & memory into one for double
        y-axis plot
    '''
    def __init__(self, memory_path=None, line_path=None):
        self.memory_path = memory_path
        self.line_path = line_path
        self.memory_text = None
        self.line_text = None
        self.memory_df = None
        self.line_df = None
        self.fun_locs = {}

        if self.memory_path:
            with open(self.memory_path) as text_file:
                text = text_file.read()
            self.memory_text = text.replace('\x00', '').split()
            self.memory_df = self.get_profile_df(self.memory_text, 'memory')

        if self.line_path:
            with open(self.line_path) as text_file:
                text = text_file.read()
            self.line_text = text.replace('\x00', '').split()
            self.line_df = self.get_profile_df(self.line_text, 'line')

    def is_float(self, s):
        try:
            float(s)
        except Exception:
            return False
        return True

    def is_int(self, s):
        try:
            int(s)
        except Exception:
            return False
        return True

    def get_dtype(self, s):
        if self.is_int(s):
            return 'i'
        elif self.is_float(s):
            return 'f'
        elif s == 'MiB':
            return 'm'
        else:
            return 's'

    def find_pattern_matches(self, string, pattern):
        matches = []
        for n in range(len(string) - len(pattern) + 1):
            s = ''.join(string[n:n+len(pattern)])
            if s == pattern:
                matches.append(n)
        return matches

    def get_mem_rows(self, string, matches):
        rows = []
        for n in matches:
            row = [int(string[n]), float(string[n+1]),
                   float(string[n+3]), int(string[n+5]),
                   string[n+6] == '@profile']
            if string[n+6] == '@profile':
                self.fun_locs[row[0]] = string[n+9].split('(')[0]
            rows.append(row)
        return rows

    def get_line_rows(self, string, matches):
        rows = [[int(string[n]), int(string[n+1]),
                 float(string[n+2]), float(string[n+3]), float(string[n+4])]
                for n in matches]
        return rows

    def get_profile_df(self, text, kind):

        # Classify each of the strings in the text
        types = [self.get_dtype(s) for s in text]

        # Filter the kind of profile
        if kind == 'memory':
            pattern = 'ifmfmi'
            get_rows = self.get_mem_rows
            cols = ['line', 'total memory (MiB)', 'memory delta (MiB)',
                    'occurences', '@profile']
        elif kind == 'line':
            pattern = 'iifff'
            get_rows = self.get_line_rows
            cols = ['line', 'hits', 'time (s)', 'per_hit (s)', 'perc_time (%)']
        else:
            raise Exception('kind must be "memory" or "line"')

        # Get indices where pattern matches start
        matches = self.find_pattern_matches(types, pattern)

        # Get the rows from the pattern indices
        rows = get_rows(text, matches)

        # Construct DataFrame from the rows
        df = pd.DataFrame(rows, columns=cols)

        # Convert units to s
        if kind == 'line':
            timer_unit = float(
                re.findall('Timer unit: .* s', ' '.join(text))
                [0].split()[2])
            cols = ['time (s)', 'per_hit (s)']
            df[cols] = timer_unit * df[cols]
        elif kind == 'memory':
            df['memory delta (MiB)'] = df['total memory (MiB)'].diff()
            df = df[~df['@profile']]

        return df

    def plot_both(self, xtick_style=None, font='Georgia'):
        fs = 16
        if not (self.memory_text and self.line_text):
            raise Exception('Need both memory and line to plot both.')

        df = self.memory_df.merge(self.line_df, on='line', how='outer')\
            .set_index('line')

        # create figure and axis objects with subplots()
        fig_x = 18
        _, ax = plt.subplots(figsize=(fig_x, 6))

        # Make first Plot
        ax.plot(df.index, df['time (s)'], '_', color="red", marker="^")
        ax.set_xlabel("Line #", fontsize=fs, fontname=font)
        ax.set_ylabel("Time (s)",
                      color="red", fontsize=fs, fontname=font)

        # twin object for two different y-axis on the sample plot
        ax2 = ax.twinx()

        # Make second plot
        ax2.plot(df.index, df['memory delta (MiB)'], '_',
                 color="blue", marker="v")
        mn = df['memory delta (MiB)'].min()
        mx = df['memory delta (MiB)'].max()
        ax2.set_ylabel("Memory Change (MiB)",
                       color="blue", fontsize=fs, fontname=font)

        # Add function labels
        for fun_start, fun_name in self.fun_locs.items():
            ax2.plot([fun_start, fun_start], [mn, mx], 'k--')
            ax2.text(fun_start, (mx + mn)/2, fun_name, rotation='vertical',
                     fontname=font, fontsize=fs-2,
                     bbox={'facecolor': 'white', 'edgecolor': 'black'},
                     ha='center', va='center')

        if xtick_style == 'dense':
            ax.xaxis.set_major_locator(MultipleLocator(5))
            ax.xaxis.set_major_formatter('{x:.0f}')

            # For the minor ticks, use no labels; default NullFormatter.
            ax.xaxis.set_minor_locator(MultipleLocator(1))

            ax.tick_params(direction="in", which='both')
            ax2.tick_params(direction="in")

        plt.show()

    def plotly_both(self):
        # Make the merged DataFrame
        if not (self.memory_text and self.line_text):
            raise Exception('Need both memory and line to plot both.')

        df = self.memory_df.merge(self.line_df, on='line', how='outer')\
            .set_index('line')

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Make first scatter
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['time (s)'],
                name='Time Profiling', mode='markers',
                marker_symbol='triangle-up'
                ),
            secondary_y=False
        )

        # Make the second scatter
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['memory delta (MiB)'],
                name='Memory Profiling', mode='markers',
                marker_symbol='triangle-down'
                ),
            secondary_y=True
        )

        # Set y-axes titles
        fig.update_yaxes(title_text='Time (s)', secondary_y=False)
        fig.update_yaxes(title_text="Memory Change (MiB)", secondary_y=True)

        fig.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mpath', type=str, help='Path to the memory profile')
    parser.add_argument('lpath', type=str, help='Path to the line profile')
    args = parser.parse_args()

    pp = ProfileProcessor(memory_path=args.mpath,
                          line_path=args.lpath)
    pp.plot_both(xtick_style='dense')
