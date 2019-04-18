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
tran_loader = DataLoader(train_data, batch_size, True)
test_loader = DataLoader(test_data, batch_size, True)
val_loader = DataLoader(valid_data, batch_size, True)

dataiter = iter(train_loader)
sample_x, sample_y = dataiter.next()


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
    
