import pandas as pd
import tables
import numpy as np


def load_data(file_name):
    cols = ['<DATE>', '<TIME>', '<HIGH>', '<LOW>']
    data = pd.read_csv(file_name, usecols=cols)
    data.rename(columns={'<DATE>': 'DATE', '<TIME>': 'TIME', '<HIGH>': 'HIGH', '<LOW>': 'LOW'}, inplace=True)
    data.dropna(inplace=True)
    data = data[(data['DATE'].str.isdigit()) & (data['TIME'].str.isdigit())].reset_index(drop=True)
    data = data.astype(float).astype(int)

    return data


class Solver:
    def __init__(self, file_name):
        self.data = load_data(file_name)
        self.M = 10
        self.N = 13
        self.P = 60
        self.time_step = self.P / self.N
        self.price_step = 1 / self.M

    def run(self, start_time, end_time):
        output_filename = str(self.data['DATE'][start_time]) + '.h5'
        file, datetime_array, data_array = self.create_output_storage(output_filename)

        self.process_data(start_time, end_time, datetime_array, data_array)

        file.close()

    def create_output_storage(self, filename):
        file = tables.open_file(filename, mode='w')
        datetime_atom = tables.Int64Atom()
        datetime_array = file.create_earray(file.root, 'datetime', datetime_atom, (0,))

        data_atom = tables.Float64Atom()
        data_array = file.create_earray(file.root, 'data', data_atom, (0, 250))

        return file, datetime_array, data_array

    def price_counter(self, price_min, price_max, low, high):
        if low >= price_min and high < price_max:
            return 1
        elif low >= price_max or high <= price_min:
            return 0
        elif low >= price_min:
            return (price_max - low) / (high - low)
        elif high <= price_max:
            return (high - price_min) / (high - low)
        else:
            return (price_max - price_min) / (high - low)

    def process_data(self, start_time, end_time, datetime_array, data_array):
        for i in range(self.P + start_time, end_time):
            max_price = self.data['HIGH'][i - self.P: i].max()
            min_price = self.data['LOW'][i - self.P: i].min()
            low_data = self.data['LOW'][i - self.P: i].apply(
                lambda x: (x - min_price) / (max_price - min_price)).reset_index(
                drop=True)
            high_data = self.data['HIGH'][i - self.P: i].apply(
                lambda x: (x - min_price) / (max_price - min_price)).reset_index(
                drop=True)
            table = []
            for n in range(0, self.N):
                n_max = self.time_step * (n + 1)
                n_min = self.time_step * n
                n_maxi = int(n_max)
                for m in range(0, self.M):
                    n_mini = int(n_min)
                    result = (1 - (n_min % 1)) * self.price_counter(self.price_step * m, self.price_step * (m + 1),
                                                                    low_data[n_mini],
                                                                    high_data[n_mini])
                    n_mini += 1
                    while n_mini < n_maxi:
                        result += 1 * self.price_counter(self.price_step * m, self.price_step * (m + 1),
                                                         low_data[n_mini],
                                                         high_data[n_mini])
                        n_mini += 1
                    result += (n_max % 1) * self.price_counter(self.price_step * m, self.price_step * (m + 1),
                                                               low_data[n_mini],
                                                               high_data[n_mini])
                    table.append(result)
            # print(round(np.array(table).sum(), 2))
            datetime_int = self.data['TIME'][i] // 100
            datetime_array.append([datetime_int])
            to_save = list(low_data) + list(high_data) + table
            to_save = np.array([to_save])
            to_save_shaped = to_save.reshape(1, 250)
            data_array.append(to_save_shaped)


solver = Solver('./2_5219905305105663004')
solver.run(0, 797)
