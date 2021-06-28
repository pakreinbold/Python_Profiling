import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import argparse
import plotly.express as px


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
        self.all_df = None
        self.stats_df = None

        # Get the memory df if a path is provided to the txt file
        if self.memory_path:
            dfs = []
            trial_no = 0
            for path in self.memory_path:
                trial_no += 1
                with open(path) as text_file:
                    text = text_file.read()
                self.memory_text = [s.split() for s in text.replace('\x00', '')
                                    .split('\n') if s != '']
                df = self.get_profile_df('memory')
                df['Trial'] = trial_no
                dfs.append(df)
            self.memory_df = pd.concat(dfs)

        # Get the line df if a path is provided to the txt file
        if self.line_path:
            dfs = []
            trial_no = 0
            for path in self.line_path:
                trial_no += 1
                with open(path) as text_file:
                    text = text_file.read()
                self.line_text = [s.split() for s in text.replace('\x00', '')
                                  .split('\n') if s != '']
                df = self.get_profile_df('line')
                df['Trial'] = trial_no
                dfs.append(df)
            self.line_df = pd.concat(dfs)

        # If both txt file paths are provided make one big df
        if self.memory_path and self.line_path:
            self.all_df = self.memory_df.merge(
                self.line_df, on=['Line', 'Text', 'Trial'], how='outer'
            ).drop_duplicates()
            st = set(self.all_df.columns) \
                - {'File', 'Function', 'Line', 'Text', 'Trial'}
            ordered_columns = ['File', 'Function', 'Line'] \
                + list(st) + ['Text', 'Trial']
            self.all_df = self.all_df[ordered_columns]\
                .sort_values(['File', 'Line'])

            # Condense into aggregates
            num_cols = ['Memory Delta (MiB)', 'Time (s)']
            means = self.all_df\
                .groupby(['File', 'Function', 'Line', 'Text'])\
                [num_cols].mean().reset_index()
            means.rename(
                columns={col: 'Mean ' + col for col in num_cols},
                inplace=True)
            stds = self.all_df\
                .groupby(['File', 'Function', 'Line', 'Text'])\
                [num_cols].std().reset_index()
            stds.rename(
                columns={col: 'Std ' + col for col in num_cols},
                inplace=True)
            self.stats_df = means.merge(stds, on=['File', 'Line', 'Function', 'Text'], how='outer')

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

    def get_mem_rows(self, pattern):
        rows = []
        for line in self.memory_text:
            types = ''.join([self.get_dtype(s) for s in line])
            if types[0] == 'i':
                if types[:len(pattern)] == pattern:
                    rows.append([int(line[0]), float(line[1]), float(line[3]),
                                 int(line[5]), ' '.join(line[6:])])
                else:
                    rows.append([int(line[0]), np.nan, np.nan, np.nan,
                                 ' '.join(line[1:])])
        return rows

    def get_line_rows(self, pattern):
        rows = []
        function_name = None
        file_name = None
        for line in self.line_text:
            types = ''.join([self.get_dtype(s) for s in line])
            if line[0] == 'File:':
                file_name = line[1]
            elif line[0] == 'Function:':
                function_name = line[1]
            elif types[0] == 'i':
                if types[:len(pattern)] == pattern:
                    rows.append([int(line[0]), int(line[1]), float(line[2]),
                                 float(line[3]), float(line[4]),
                                 ' '.join(line[5:]), file_name, function_name])
                else:
                    rows.append([int(line[0]), 0, 0, 0, 0, ' '.join(line[1:]),
                                file_name, function_name])
        return rows

    def find_next_gen(self, row, function_list):
        parent = row['Function']
        text = row['Text']
        try:
            child = text.split('(')[0].split()[-1]
            if child in function_list:
                return child
            else:
                return parent
        except Exception:
            return parent

    def clean_filenames(self, s, mode='osx'):
        if mode == 'osx':
            return s.split('/')[-1]
        elif mode == 'win':
            return s.split('\\')[-1]

    def get_profile_df(self, kind):

        # Filter the kind of profile
        if kind == 'memory':
            pattern = 'ifmfmi'
            get_rows = self.get_mem_rows
            cols = ['Line', 'Total Memory (MiB)', 'Memory Delta (MiB)',
                    'Occurences', 'Text']
        elif kind == 'line':
            pattern = 'iifff'
            get_rows = self.get_line_rows
            cols = ['Line', 'Hits', 'Time (s)', 'Per Hit (s)', 'Perc Time (%)',
                    'Text', 'File', 'Function']
        else:
            raise Exception('kind must be "memory" or "line"')

        # Get the rows from the pattern indices
        rows = get_rows(pattern)

        # Construct DataFrame from the rows
        df = pd.DataFrame(rows, columns=cols)

        # Convert units to s
        if kind == 'line':
            timer_unit = float(self.line_text[0][-2])
            cols = ['Time (s)', 'Per Hit (s)']
            df[cols] = timer_unit * df[cols]
            df['File'] = df['File'].apply(
                lambda s: self.clean_filenames(s, mode='win'))
        elif kind == 'memory':
            # df['Memory Delta (MiB)'] = df['Total Memory (MiB)'].diff()
            df.loc[df['Text'] == '@profile', 'Memory Delta (MiB)'] = 0.0
            df.fillna(0.0, inplace=True)

        return df

    def plot_byline(self, files=[], xtick_style=None, font='Georgia'):
        fs = 16
        if not (self.memory_text and self.line_text):
            raise Exception('Need both memory and line to plot both.')

        # If no files specified, show them all
        if len(files) == 0:
            files = self.stats_df['File'].unique()

        for fl in files:
            df = self.stats_df[self.stats_df['File'] == fl]

            # create figure and axis objects with subplots()
            fig_x = 18
            _, ax = plt.subplots(figsize=(fig_x, 6))

            # Make first Plot
            ax.bar(x=df['Line'], height=df['Mean Time (s)'],
                   yerr=df['Std Time (s)'],
                   fc=[1, 0, 0, 0.3], edgecolor=[1, 0, 0, 1])
            ax.set_xlabel("Line #", fontsize=fs, fontname=font)
            ax.set_ylabel("Time (s)",
                          color="red", fontsize=fs, fontname=font)
            ax.set_ylim(bottom=0.0)

            # twin object for two different y-axis on the sample plot
            ax2 = ax.twinx()

            # Make second plot
            ax2.bar(df['Line'], df['Mean Memory Delta (MiB)'],
                    yerr=df['Std Memory Delta (MiB)'],
                    color=[0, 0, 1, 0.3], edgecolor=[0, 0, 1, 1])
            mn = df['Mean Memory Delta (MiB)'].min()
            mx = df['Mean Memory Delta (MiB)'].max()
            ax2.set_ylabel("Memory Change (MiB)",
                           color="blue", fontsize=fs, fontname=font)

            # Add function labels
            fun_locs = df.groupby('Function')['Line'].min().to_dict()
            for fun_name, fun_start in fun_locs.items():
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

    def plot_ranks(self, mode, num_ranks=10):
        if mode == 'line':
            y_lbl = 'Time (s)'
            df = self.line_df.sort_values(y_lbl, ascending=False)\
                .head(num_ranks).reset_index()
            hover = {'Line': True, 'Time (s)': ':.2f', 'Hits': True,
                     'Text': True}
            tit = 'Slowest Lines'
        elif mode == 'memory':
            y_lbl = 'Memory Delta (MiB)'
            df = self.memory_df.sort_values(y_lbl, ascending=False)\
                .head(num_ranks).reset_index()
            hover = {'Memory Delta (MiB)': ':.2f', 'Line': True,
                     'Occurences': True, 'Total Memory (MiB)': ':.2f',
                     'Text': True}
            tit = 'Most Memory Intensive Lines'
        else:
            print('Input mode must be "line" or "memory". ' +
                  f'Cannot show ranks for "{mode}"')
            return

        # Get the top ranks
        df.index.name = 'Rank'
        df.index += 1

        # Make bar chart
        fig = px.bar(df, x=y_lbl, color=y_lbl, orientation='h',
                     hover_data=hover, text='Line',
                     color_continuous_scale='sunsetdark')
        fig.update_traces(textposition='outside')
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(
            title=tit,
            yaxis=dict(
                tickmode='linear',
                tick0=1,
                dtick=1
            ),
            width=1200, height=600
        )
        fig.show()

    def plot_byfn(self, mode, ag='Mean'):
        if mode == 'memory':
            ylbl = ag + ' Memory Delta (MiB)'
        elif mode == 'line':
            ylbl = ag + ' Time (s)'

        if 'Branch-Function' not in self.stats_df.columns:
            self.stats_df['Branch-Function'] = self.stats_df.apply(
                lambda row: self.find_next_gen(
                    row, self.stats_df['Function'].unique()),
                axis=1)

        hover = {'Line': True, 'Text': True}

        fig = px.bar(
            self.stats_df, x='Function', y=ylbl,
            color='Branch-Function', hover_data=hover
        )
        fig.update_layout(width=1000, height=500)
        fig.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mpath', type=str, help='Path to the memory profile')
    parser.add_argument('lpath', type=str, help='Path to the line profile')
    args = parser.parse_args()

    pp = ProfileProcessor(memory_path=args.mpath,
                          line_path=args.lpath)
    pp.plot_both(xtick_style='dense')
