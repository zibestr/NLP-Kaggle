from src.ml.make_dataset import save_dataset

if __name__ == '__main__':
    save_dataset(True, 'train')
    print('Save training dataset')
    save_dataset(False, 'test')
    print('Save testing dataset')
