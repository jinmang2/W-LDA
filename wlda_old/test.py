class Data(object):
    '''
    Data Generator object. Main functionality is contained in ``minibatch'' method
    and ``subsampled_labeled_data'' if training in a semi-supervised fashion.
    Introducing new datasets requires implementing ``load'' and possibly overwriting
    portions of ``__init__''.
    '''
    def __init__(self, batch_size=1, data_path='', ctx=mx.cpu(0)):
        '''
        Constructor for Data.
        Args
        ----
        batch_size: int, default 1
          An integer specifying the batch size - required for precompiling the graph.
        data_path: string, default ''
          This is primarily used by Mulan to specify which dataset to load from Mulan,
          e.g., data_path='bibtex'.
        ctx: mxnet device context, default mx.cpu(0)
          Which device to store/run the data and model on.
        Returns
        -------
        Data object
        '''
        self.batch_size = batch_size
        if data_path == '':
            data, labels, maps = self.load()
        else:
            data, labels, maps = self.load(data_path)
        self.ctx = ctx
        # # normalize the data:
        # def softmax(x):
        #     """Compute softmax values for each sets of scores in x."""
        #     e_x = np.exp(x - np.max(x, axis=1).reshape((-1,1)))
        #     return e_x / np.sum(e_x, axis=1).reshape((-1,1))
        # for i in range(len(data)):
        #     data[i] = softmax(data[i])

        data_names = ['train','valid','test','train_with_labels','valid_with_labels','test_with_labels']
        label_names = ['train_label', 'valid_label', 'test_label']

        self.data = dict(zip(data_names, data))
        self.labels = dict(zip(label_names, labels))

        # repeat data to at least match batch_size
        for k, v in self.data.items():
            if v is not None and v.shape[0] < self.batch_size:
                print('NOTE: Number of samples for {0} is smaller than batch_size ({1}<{2}). Duplicating samples to exceed batch_size.'.format(k,v.shape[0],self.batch_size))
                if type(v) is np.ndarray:
                    self.data[k] = np.tile(v, (self.batch_size // v.shape[0] + 1, 1))
                else:
                    self.data[k] = mx.nd.tile(v, (self.batch_size // v.shape[0] + 1, 1))

        for k, v in self.labels.items():
            if v is not None and v.shape[0] < self.batch_size:
                print('NOTE: Number of samples for {0} is smaller than batch_size ({1}<{2}). Duplicating samples to exceed batch_size.'.format(k,v.shape[0],self.batch_size))
                self.labels[k] = np.tile(v, (self.batch_size // v.shape[0] + 1, ))

        map_names = ['vocab2dim','dim2vocab','topic2dim','dim2topic']
        self.maps = dict(zip(map_names, maps))
        dls = [self.dataloader(d, batch_size) for d in data]
        dis = [iter(dl) if dl is not None else None for dl in dls]
        self.dataloaders = dict(zip(data_names, dls))
        self.dataiters = dict(zip(data_names, dis))
        self.wasreset = dict(zip(data_names, np.ones(len(data_names), dtype='bool')))

        self.data_dim = self.data['train'].shape[1]
        if self.data['train_with_labels'] is not None:
            self.label_dim = self.data['train_with_labels'].shape[1] - self.data['train'].shape[1]


    def dataloader(self, data, batch_size, shuffle=True):
        '''
        Constructs a data loader for generating minibatches of data.
        Args
        ----
        data: numpy array, no default
          The data from which to load minibatches.
        batch_size: integer, no default
          The # of samples returned in each minibatch.
        shuffle: boolean, default True
          Whether or not to shuffle the data prior to returning the data loader.
        Returns
        -------
        DataLoader: A gluon DataLoader iterator
        '''
        if data is None:
            return None
        else:
            # inds = np.arange(data.shape[0])
            # if shuffle:
            #     np.random.shuffle(inds)
            # ordered = data[inds]
            # N, r = divmod(data.shape[0], batch_size)
            # if r > 0:
            #     ordered = np.vstack([ordered, ordered[:r]])
            if type(data) is np.ndarray:
                return gluon.data.DataLoader(data, batch_size, last_batch='discard', shuffle=shuffle)
            else:
                return io.NDArrayIter(data={'data': data}, batch_size=batch_size, shuffle=shuffle, last_batch_handle='discard')

    def force_reset_data(self, key, shuffle=True):
        '''
        Resets minibatch index to zero to restart an epoch.
        Args
        ----
        key: string, no default
          Required to select appropriate data in ``data'' object,
          e.g., 'train', 'test', 'train_with_labels', 'test_with_labels'.
        shuffle: boolean, default True
          Whether or not to shuffle the data prior to returning the data loader.
        Returns
        -------
        Nothing.
        '''
        if self.data[key] is not None:
            if type(self.data[key]) is np.ndarray:
                self.dataloaders[key] = self.dataloader(self.data[key], self.batch_size, shuffle)
                self.dataiters[key] = iter(self.dataloaders[key])
            else:
                self.dataiters[key].hard_reset()
            self.wasreset[key] = True

    def minibatch(self, key, pad_width=0):
        '''
        Returns a minibatch of data (stored on device self.ctx).
        Args
        ----
        key: string, no default
          Required to select appropriate data in ``data'' object,
          e.g., 'train', 'test', 'train_with_labels', 'test_with_labels'.
        pad_width: integer, default 0
          The amount to zero-pad the labels to match the dimensionality of z.
        Returns
        -------
        minibatch: NDArray on device self.ctx
          An NDArray of size batch_size x # of features.
        '''
        if self.dataiters[key] is None:
            return None
        else:
            if type(self.data[key]) is np.ndarray:
                try:
                    mb = self.dataiters[key].__next__().reshape((self.batch_size, -1))
                    if pad_width > 0:
                        mb = mx.nd.concat(mb, mx.nd.zeros((self.batch_size, pad_width)))
                    return mb.copyto(self.ctx)
                except:
                    self.force_reset_data(key)
                    mb = self.dataiters[key].__next__().reshape((self.batch_size, -1))
                    if pad_width > 0:
                        mb = mx.nd.concat(mb, mx.nd.zeros((self.batch_size, pad_width)))
                    return mb.copyto(self.ctx)
            else:
                try:
                    mb = self.dataiters[key].__next__().data[0].as_in_context(self.ctx)
                    return mb
                except:
                    self.dataiters[key].hard_reset()
                    mb = self.dataiters[key].__next__().data[0].as_in_context(self.ctx)
                    return mb

    def get_documents(self, key, split_on=None):
        '''
        Retrieves a minibatch of documents via ``data'' object parameter.
        Args
        ----
        key: string, no default
          Required to select appropriate data in ``data'' object,
          e.g., 'train', 'test', 'train_with_labels', 'test_with_labels'.
        split_on: integer, default None
          Useful if self.data[key] contains both data and labels in one
          matrix and want to split them, e.g., split_on = data_dim.
        Returns
        -------
        minibatch: NDArray if split_on is None, else [NDarray, NDArray]
        '''
        if 'labels' in key:
            batch = self.minibatch(key, pad_width=self.label_pad_width)
        else:
            batch = self.minibatch(key)
        if split_on is not None:
            batch, labels = batch[:,:split_on], batch[:,split_on:]
            return batch, labels
        else:
            return batch

    @staticmethod
    def visualize_series(y, ylabel, file, args, iteration, total_samples, labels=None):
        '''
        Plots and saves a figure of y vs iterations and epochs to file.
        Args
        ----
        y: a list (of lists) or numpy array, no default
          A list (of possibly another list) of numbers to plot.
        ylabel: string, no default
          The label for the y-axis.
        file: string, no default
          A path with filename to save the figure to.
        args: dictionary, no default
          A dictionary of model, training, and evaluation specifications.
        iteration: integer, no default
          The current iteration in training.
        total_samples: integer, no default
          The total number of samples in the dataset - used along with batch_size
          to convert iterations to epochs.
        labels: list of strings, default None
          If y is a list of lists, the labels contains names for each element
          in the nested list. This is used to create an appropriate legend
          for the plot.
        Returns
        -------
        Nothing.
        '''
        if len(y) > 0:
            fig = plt.figure()
            ax = plt.subplot(111)
            x = np.linspace(0, iteration, num=len(y)) * args['batch_size'] / total_samples
            y = np.array(y)
            if len(y.shape) > 1:
                for i in range(y.shape[1]):
                    if labels is None:
                        plt.plot(x,y[:,i])
                    else:
                        plt.plot(x,y[:,i], label=labels[i])
            else:
                plt.plot(x,y)
            ax.set_ylabel(ylabel)
            ax.set_xlabel('Epochs')
            plt.grid(True)

            ax2 = ax.twiny()

            # https://pythonmatplotlibtips.blogspot.com/2018/01/add-second-x-axis-below-first-x-axis-python-matplotlib-pyplot.html
            # Decide the ticklabel position in the new x-axis,
            # then convert them to the position in the old x-axis
            # xticks list seems to be padded with extra lower and upper ticks --> subtract 2 from length
            newlabel = np.around(np.linspace(0, iteration, num=len(ax.get_xticks())-2)).astype('int') # labels of the xticklabels: the position in the new x-axis
            # ax2.set_xticks(ax.get_xticks())
            ax2.set_xticks(newlabel * args['batch_size'] / total_samples)
            ax2.set_xticklabels(newlabel//1000)

            ax2.xaxis.set_ticks_position('bottom') # set the position of the second x-axis to bottom
            ax2.xaxis.set_label_position('bottom') # set the position of the second x-axis to bottom
            ax2.spines['bottom'].set_position(('outward', 36))
            ax2.set_xlabel('Thousand Iterations')
            ax2.set_xlim(ax.get_xlim())

            if labels is not None:
                lgd = ax.legend(loc='center left', bbox_to_anchor=(1.05, 1))
                fig.savefig(args['saveto']+file, additional_artists=[lgd], bbox_inches='tight')
            else:
                fig.tight_layout()
                fig.savefig(args['saveto']+file)
            plt.close()

    def load(self, path=''):
        '''
        Loads data and maps from path.
        Args
        ----
        path: string, default ''
          A path to the data file.
        Returns
        -------
        data: list of numpy arrays
          A list of the different subsets of data, e.g.,
          `train', `test', 'train_with_labels', 'test_with_labels'.
        maps: list of dictionaries
          A list of dictionaries for mapping between dimensions and strings,
          e.g., 'vocab2dim', 'dim2vocab', 'topic2dim', 'dim2topic'.
        '''
        data = [np.empty((1,1)) for data in ['train','valid','test','train_with_labels','valid_with_labels','test_with_labels']]
        maps = [{'a':0}, {0:'a'}, {'Letters':0}, {0:'Letters'}]
        self.data_path = path + '***.npz'
        return data, maps