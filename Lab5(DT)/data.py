from pandas import read_csv


def read_dataset(file_name):
    return read_csv(file_name, usecols=lambda x: x != 'y').values.tolist(), read_csv(file_name)['y'].values.tolist()


def read_train(i):
    return read_dataset('DT_csv/' + str(i).zfill(2) + '_train.csv')


def read_test(i):
    return read_dataset('DT_csv/' + str(i).zfill(2) + '_test.csv')


train_data = [read_train(i) for i in range(1, 22)]
test_data = [read_test(i) for i in range(1, 22)]
train_X_list, train_y_list = [x[0] for x in train_data], [[y - 1 for y in x[1]] for x in train_data]
test_X_list, test_y_list = [x[0] for x in test_data], [[y - 1 for y in x[1]] for x in test_data]
