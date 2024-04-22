from src.dl.make_dataset import save_dataset

if __name__ == '__main__':
    save_dataset(True, 'mps', 'train')
    print('Save training dataset')
    save_dataset(False, 'mps', 'test')
    print('Save testing dataset')
