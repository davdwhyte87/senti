from string import punctuation
from collections import Counter
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

# import datasets
with open('data/reviews.txt', 'r') as f:
    reviews = f.read()

with open('data/labels.txt', 'r') as f:
    labels = f.read()

# remove punctuatios
reviews = reviews.lower()
# all_text = ''.join([c for c in reviews if c not in punctuation])
all_text = ''
for c in reviews:
    if c not in punctuation:
        all_text = all_text+c
print(type(all_text))
reviews_split = all_text.split('\n')
print(len(reviews_split))

# change list without punctuation to text
all_text = ' '.join(reviews_split)
# split text into words
words = all_text.split()

# build a dictionary that maps words to integers
counter = Counter(words)
vocab = sorted(counter, key=counter.get, reverse=True)
vocab_to_int = []

vocab_to_int = { word: ii for ii, word in enumerate(vocab, 1)}

reviews_int = []
# get just the int from the vocab_to_int dictionary
for review in reviews_split:
    reviews_int.append([vocab_to_int[word] for word in review.rsplit()])


# # get indices of any reviews with length 0
# non_zero_idx = [ii for ii, review in enumerate(reviews_ints) if len(review) != 0]

# # remove 0-length reviews and their labels
# reviews_ints = [reviews_ints[ii] for ii in non_zero_idx]

labels_split = labels.split('\n')
encoded_labels = np.array([1 if label == 'positive' else 0 for label in labels_split])

print('Number of reviews before removing outliers: ', len(reviews_int))

## remove any reviews/labels with zero length from the reviews_ints list.

# get indices of any reviews with length 0
non_zero_idx = [ii for ii, review in enumerate(reviews_int) if len(review) != 0]

# remove 0-length reviews and their labels
reviews_int = [reviews_int[ii] for ii in non_zero_idx]
encoded_labels = np.array([encoded_labels[ii] for ii in non_zero_idx])

print('Number of reviews after removing outliers: ', len(reviews_int))

def pad_features(reviews_int, seq_length):
    ''' Return features of review_ints, where each review is padded with 0's
        or truncated to the input seq_length.
    '''

    # getting the correct rows x cols shape
    features = np.zeros((len(reviews_int), seq_length), dtype=int)

    # for each review, I grab that review and
    for i, row in enumerate(reviews_int):
        features[i, -len(row):] = np.array(row)[:seq_length]

    return features


seq_length = 200
features = pad_features(reviews_int, seq_length=seq_length)
print(features)
# assert len(features)==len(reviews_int), "Your features should have as many rows as reviews."
# assert len(features[0])==seq_length, "Each feature row should contain seq_length values."


split_frac = 0.8
split_idx = int(len(features)*split_frac)
print(len(features))
train_x, remaining_x = features[:split_idx], features[split_idx:]
train_y, remaining_y = encoded_labels[:split_idx], encoded_labels[split_idx:]

test_idx = int(len(remaining_x)*0.5)
val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]

## print out the shapes of your resultant feature data
print("\t\t\tFeature Shapes:")
print("Train set: \t\t{}".format(train_x.shape),
      "\nValidation set: \t{}".format(val_x.shape),
      "\nTest set: \t\t{}".format(test_x.shape))


train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
batch_size = 50
train_loader = DataLoader(train_data, batch_size, True)
test_loader = DataLoader(test_data, batch_size, True)
val_loader = DataLoader(valid_data, batch_size, True)

dataiter = iter(train_loader)
sample_x, sample_y = dataiter.next()

# First checking if GPU is available
train_on_gpu=torch.cuda.is_available()
if(train_on_gpu):
    print('Training on GPU.')
else:
    print('No GPU available, training on CPU.')

class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout_prob=0.5):
        super(SentimentRNN, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout_prob, batch_first=True)

        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)
        x = x.long()
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        out = self.dropout(lstm_out)
        out = self.fc(out)
        # sigmoid function
        sig_out = self.sig(out)

        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1] # get last batch of labels

        # return last sigmoid output and hidden state  Modu
        return sig_out, hidden

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden

# Instantiate the model w/ hyperparams
vocab_size = len(vocab_to_int)+1 # +1 for the 0 padding + our word tokens
output_size = 1
embedding_dim = 400
hidden_dim = 256
n_layers = 2

net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
# print(net)
lr = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(params=net.parameters(), lr=lr)

epochs = 1
print_every = 100
clip = 5
# move model to GPU, if available
if(train_on_gpu):
    net.cuda()

net.train()

# train for some number of epochs
for e in range(epochs):
    # initialize hidden state
    h = net.init_hidden(batch_size)

    # batch loop
    for inputs, labels in train_loader:
        # counter += 1
        if(train_on_gpu):
            inputs, labels = inputs.cuda(), labels.cuda()

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])

        # zero accumulated gradients
        net.zero_grad()

        # get the output from the model
        output, h = net(inputs, h)
        print(output)
        # calculate the loss and perform backprop
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()

        # loss stats
        # if counter % print_every == 0:
# Get validation loss
# val_h = net.init_hidden(batch_size)
# val_losses = []
# net.eval()
# for inputs, labels in val_loader:
#     # Creating new variables for the hidden state, otherwise
#     # we'd backprop through the entire training history
#     val_h = tuple([each.data for each in val_h])
#
#     if(train_on_gpu):
#         inputs, labels = inputs.cuda(), labels.cuda()
#
#     output, val_h = net(inputs, val_h)
#     val_loss = criterion(output.squeeze(), labels.float())
#
#     val_losses.append(val_loss.item())

net.train()
print("Epoch: {}/{}...".format(e+1, epochs),
      "Step: {}...".format(counter),
      "Loss: {:.6f}...".format(loss.item()))
      # "Val Loss: {:.6f}".format(np.mean(val_losses)))