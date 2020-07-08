import pandas as pd
import numpy as np
from pandas.io.common import ParserError,EmptyDataError


class Processor:
    def __init__(self, filepath, method=None):
        self.filepath = filepath
        self.dataset, self.feature_type, self.old_df = self.prepate_dataset()
        self.normalized = self.normalize(method)
        self.npy = self.count_other_stat()
        self.headers = self.make_headers()

    def prepate_dataset(self):
        '''Splitting features column into new columns and cleaning'''
        try:
            df = pd.read_csv(self.filepath, sep='\t',header=0)
        except ParserError:
            print('Had problems parsing your file')
        except EmptyDataError:
            print("Your file seems to be empty, try another one")

        ds = df.features.str.split(',', expand=True)
        feature = ds.pop(ds.columns[0])[0]  # assuming there would be features of a single type in a file
        ns = np.array(ds)
        df.drop(df.columns[1:], axis=1, inplace=True)
        return ns.astype(np.float), feature, df

    @staticmethod
    def zscore(table):
        """Implementation of z-score standartization"""
        new_list = []
        for idx, col in enumerate(table.T, 1):
            zcol = []
            for v in col:
                cell_vall = ((v - np.mean(col)) / np.std(col))
                zcol.append(cell_vall)
            new_list.append(zcol)
        return np.array(new_list).T

    def normalize(self, method):
        if method.lower() == 'z':
            return self.zscore(self.dataset)
        else:
            raise ValueError("Sorry, for now only zscore is implemented. You can call it by normalize(method='z') ")

    def count_other_stat(self):
        """Counting max index for each row
         and also deviation of each max feature from mean(max_i)
         than adding them to the existing dataset """
        local_maxs = np.array([row.argmax() for row in self.normalized])
        data_with_maxs = np.hstack((self.normalized, local_maxs[:, None]))

        abs_mean_diff = []
        with np.nditer(data_with_maxs[:, -1], op_flags=['readwrite']) as it:
            for mx_idx in it:
                diff = mx_idx - np.mean(data_with_maxs[:, -1], axis=0)
                abs_mean_diff.append([diff])

        abs_mean_diff = np.array(abs_mean_diff)
        final_ar = np.hstack((data_with_maxs, abs_mean_diff))
        return final_ar

    def make_headers(self):
        """Creating header for the resulting dataset"""
        headers = []
        for i in range(1, len(self.npy.T) - 1):
            headers.append(f'feature_{self.feature_type}_stand_{i}')
        headers.append(f'max_feature_{self.feature_type}_index')
        headers.append(f'max_feature_{self.feature_type}_abs_mean_diff')
        return headers

    def create_df(self):
        fd = pd.DataFrame(self.npy, columns=self.headers)
        r = pd.concat([self.old_df, fd], axis=1)
        return r

    def save_to_file(self, filename):
        dfs = self.create_df()
        with np.printoptions(precision=3, suppress=True):
            print(dfs)
            dfs.to_csv(filename, sep='\t', index=False)


def main():
    p = Processor('data/test.tsv', method='z')
    p.save_to_file('test_proc.tsv')


if __name__ == "__main__":
    main()
