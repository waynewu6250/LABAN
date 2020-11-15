import torch as t
from torch.utils.data import Dataset, DataLoader
import pickle
from config import opt
from sklearn.model_selection import train_test_split
import numpy as np

class CoreDataset(Dataset):

    def __init__(self, data, labels, masks, num_labels, opt, segs=None, X_lengths=None):
        self.data = data
        self.labels = labels
        self.masks = masks
        self.num_data = len(self.data)
        self.maxlen = opt.maxlen
        self.num_labels = num_labels
        self.mode = opt.data_mode
        self.sentence_mode = opt.sentence_mode
        self.segs = segs
        self.X_lengths = X_lengths
        self.opt = opt

    def __getitem__(self, index):

        # caps
        caps = self.data[index]
        caps = t.tensor(caps)

        # labels
        label = self.labels[index]
        if self.mode == 'single':
            labels = t.from_numpy(np.array(label))
        else:
            label = t.LongTensor(np.array(label))
            labels = t.zeros(self.num_labels).scatter_(0, label, 1)

        # masks
        masks = t.tensor(self.masks[index])

        if self.opt.dialog_data_mode:
            X_lengths = t.tensor(self.X_lengths[index])
            return caps, labels, masks, X_lengths

        if self.sentence_mode == 'one':
            return caps, labels, masks
        else:
            # segments
            segs = t.tensor(self.segs[index]) 
            return caps, labels, masks, segs
        

    def __len__(self):
        return len(self.data)

def get_dataloader(data, labels, masks, num_labels, opt, segs=None, X_lengths=None):
    dataset = CoreDataset(data, labels, masks, num_labels, opt, segs, X_lengths)
    batch_size = opt.dialog_batch_size if opt.dialog_data_mode else opt.batch_size
    return DataLoader(dataset, 
                      batch_size=batch_size, 
                      shuffle=False)

######################################################################

if __name__ == '__main__':
    
    with open(opt.data_path, 'rb') as f:
        data = pickle.load(f)
    X, y, entities = zip(*data)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    
    