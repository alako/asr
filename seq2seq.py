import os
import pickle
import torch
import numpy as np
from tqdm import tqdm
import wandb

import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from nltk.metrics.distance import edit_distance
from collections import defaultdict
from torch.utils.data import DataLoader
import argparse

### Dataset ###
class TIMITDataset(Dataset):
    """TIMIT speech recognition dataset."""

    def __init__(self, root_dir, char_to_ix, ix_to_char):

        self.root_dir = root_dir
        file_list = os.listdir(self.root_dir)
        self.files = []
        for file in file_list:
            if file.endswith(".npy"):
                self.files.append(file)
        self.char_to_ix = char_to_ix
        self.ix_to_char = ix_to_char

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        
        prefix = fname.split(".")[0]
        txt_path = prefix + ".txt"
        txt_file = open(self.root_dir + "/" + txt_path)
        txt = txt_file.read().replace(" ", "")
        txt =[char for char in txt]
        txt.append("<EOS>")
        txt.insert(0, "<SOS>")
        txt_file.close()
        
        ix_array = np.zeros(len(txt), dtype=np.int)
        for i, char in enumerate(txt):
            ix = self.char_to_ix[char]
            if ix is None:
                ix = self.char_to_ix["<UNK>"]
            ix_array[i] = ix  
        
        features = np.load(self.root_dir + "/" + fname)
        
        sample = (torch.tensor(features), torch.tensor(ix_array))
       
        return sample

def pad_batch(batch):
    (xx, yy) = zip(*batch)
    x_lens = [x.shape[-1] for x in xx]
    y_lens = [len(y) for y in yy]
    
    max_x = max(x_lens)
    
    xx_pad = torch.zeros((len(x_lens), 20, max_x))
    for i, x in enumerate(xx):
        xx_pad[i, :, 0:x_lens[i]] = x

    yy_pad = pad_sequence(yy, batch_first=True, padding_value=3)

    return xx_pad, yy_pad, x_lens, y_lens

### Encoder and Decoder Networks ###
class Encoder(nn.Module):
    '''
    A unidirectional LSTM encoder
    Args:
        input_dim: The number of classes for the input 
        hid_dim: The dimension of each token's embedding
        n_layers: The number of LSTM layers
    '''
    def __init__(self, input_dim, hid_dim, n_layers):
        super().__init__()
        
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.rnn = torch.nn.LSTM(input_dim, hid_dim, n_layers, batch_first=True)
        
    def forward(self, audio):
        '''
        Inputs:
            audio - shape: (B, C, L), dtype torch.float32
        where:
            audio is a pytorch tensor containing a batch of preprocessed audio samples
            B is the batch dimension
            C is the number of features for the audio
            L is the number of timesteps sampled
        
        Outputs :
            hidden - shape: (n_layers, B, hid_dim), dtype torch.float32
            cell - shape: (n_layers, B, hid_dim), dtype torch.float32
        '''

        # audio should have (B, L, C) shape
        audio = audio.transpose(1, 2)
        output, (hidden, cell) = self.rnn(audio)
        return hidden, cell

class Decoder(nn.Module):
    '''
    A unidirectional LSTM decoder
    Args:
        output_dim - The number of classes for the output
        emb_dim - The dimension of each input token's embedding
        hid_dim - The dimension of the hidden state
        n_layers - The number of LSTM layers
    '''
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim  # output_dim = vocab_size
        self.n_layers = n_layers
        self.embedding = torch.nn.Embedding(output_dim, emb_dim)
        self.rnn = torch.nn.LSTM(emb_dim, hid_dim, n_layers, batch_first=True)
        self.out = torch.nn.Linear(hid_dim, output_dim)

   
    def forward(self, token, hidden, cell):
        '''
        Inputs:
            token - shape: (B, 1), dtype torch.int64
            hidden - shape: (n_layers, batch_size, hid_dim), dtype torch.float32
            cell - shape: (n_layers, batch_size, hid_dim), dtype torch.float32
        where:
            token is a pytorch tensor containing a batch of input tokens
            hidden is the previous hidden state (possibly from encoder)
            cell is the previous cell state (possibly from encoder)
        
        Outputs:
            prediction - shape: (batch_size, 1, vocab_size) predicted output
            hidden - shape: (n_layers, batch_size, hid_dim) hidden state of LSTM
            cell - shape: (n_layers, batch_size, hid_dim) cell state of LSTM
        '''
        # Translate token number to
        embed = self.embedding(token)
        # from (B, input_size) get to (B, 1, input size)
        embed = embed[:, None, :]
        output, (hidden, cell) = self.rnn(embed, (hidden, cell))
        prediction = self.out(output)
        return prediction, hidden, cell

### Dynamic Oracle ###
class CERDynamicOracle:
    '''
    Implementation of the dynamic oracle for CER.
    '''
    def __init__(self):
        self.edit_dist_table = None
        self.cur_t = 0


    def double_table_rows(self):
        # Copies the old self.edit_dist_table and doubles the number of avaliable rows
        new_space = torch.zeros(self.edit_dist_table.size(0), self.edit_dist_table.size(1))
        self.edit_dist_table = torch.cat([self.edit_dist_table, new_space])
        self.edit_dist_table[:, 0] = torch.tensor(range(self.edit_dist_table.size(0)))

    def build_edit_dist_table(self, predicted_seq, gold_seq):
        '''
        Builds the edit distance table for a given predicted sequence and gold sequence
        via the classic dynamic programming algorithm.

        The underlying table should be reused. When calling the function with a longer
        predicted sequence after the model generates more toknes, the value of self.cur_t
        is used as a starting point to build the rest of the table. 

        Inputs:
            predicted_seq - shape: (B, 1), dtype torch.int64
            gold_seq - shape: (n_layers, batch_size, hid_dim), dtype torch.float32
        '''
        start_idx = self.cur_t
        # 1. If self.cur_t == 0, initialize self.edit_dist_table
        if self.cur_t == 0:
            self.edit_dist_table = torch.zeros(gold_seq.size(0), gold_seq.size(0))
            self.edit_dist_table[0, :] = torch.tensor(range(gold_seq.size(0)))
            self.edit_dist_table[:, 0] = torch.tensor(range(gold_seq.size(0)))
            start_idx = 1
        # 2. If the length of predicted table exceeds number of rows of table, double the number of rows
        if predicted_seq.size(0) > self.edit_dist_table.size(0):
            self.double_table_rows()
        # 2. Calculate the table starting from self.cur_t using equations in handout

        if gold_seq.size(0) != self.edit_dist_table.size(1):
            self.edit_dist_table = torch.zeros(gold_seq.size(0), gold_seq.size(0))
            self.edit_dist_table[0, :] = torch.tensor(range(gold_seq.size(0)))
            self.edit_dist_table[:, 0] = torch.tensor(range(gold_seq.size(0)))
            start_idx = 1

        for i in range(start_idx, predicted_seq.size(0)):
            for j in range(1, self.edit_dist_table.size(1)):
            # for j in range(1, gold_seq.size(0)):
                self.edit_dist_table[i, j] = min(
                    self.edit_dist_table[i-1, j] + 1,
                    self.edit_dist_table[i, j-1] + 1,
                    self.edit_dist_table[i-1, j-1] + int(predicted_seq[i] != gold_seq[j])
                )
        # 3. Set the value of self.cur_t appropriately
        self.cur_t += 1

    def optimal_completion_idxes(self, index): 
        '''
        Returns the optimal completions indices for a specific predicted token index

        Inputs:
            index - index of token in predicted sequence

        Outputs:
            min_idxes - return 1-d tensor containing the optimal completion token indices
                        indices refer to the gold sequence
        '''
        if self.edit_dist_table is None:
            return torch.tensor([0])
        min_val = min(self.edit_dist_table[index, :])

        min_idxes = (self.edit_dist_table[index, :] == min_val).nonzero().squeeze()
        return min_idxes

    def optimal_completion(self, index, gold_seq):
        '''
        Returns the optimal completions for a specific predicted token index

        Inputs:
            index - index of token in predicted sequence
            gold_seq - shape: (seq_len) true sequence of tokens

        Outputs:
            optimal_completions - return 1-d tensor containing the optimal completions
                                  for the current index
        '''
        min_idxes = self.optimal_completion_idxes(index)
        gold_seq_with_eos = torch.zeros(gold_seq.size(0)+1)
        gold_seq_with_eos[:-1] = gold_seq
        gold_seq_with_eos[-1] = 2
        optimal_completions = gold_seq_with_eos[min_idxes+1]
        optimal_completions = torch.unique(optimal_completions)
        return optimal_completions.long()
    
    def get_ocd_target_from_table(self, output_logits, gold_seq):
        '''
        Returns a softmax over the optimal completions for each predicted token

        Inputs:
            output_logits - shape: (seq_len, vocab_size) softmax of model outputs
            gold_seq - shape: (seq_len) true labels for the sequence

        Outputs:
            target_softmax - shape: (seq_len, vocab_size) softmax over ocd targets 
        '''
        # 1. Generate predicted sequence from output_logits
        predicted_seq = torch.argmax(output_logits, dim=1)
        # predicted_seq = predicted_seq[:self.cur_t+1]
        # 2. Build the edit distance table if it isn't already
        self.build_edit_dist_table(predicted_seq, gold_seq)
        # 3. Get the optimal targets for each of tokens in the predicted sequence
        targets = []
        for i in range(len(predicted_seq)):
            optimal_targets = self.optimal_completion(i, gold_seq)
            # 4. Return the softmax of the optimal targets
            target = nn.functional.one_hot(optimal_targets.to(torch.int64), num_classes=output_logits.size(1))
            if target.dim() > 1:
                target = sum(target)
            target_softmax = nn.functional.softmax(target.float())
            targets.append(target_softmax)
        targets = torch.stack(targets)
        return targets

    def get_ocd_softmaxed(self, output_logits, gold_seq):
        '''
        Returns a softmax over the optimal completions for each predicted token

        Inputs:
            output_logits - shape: (seq_len, vocab_size) softmax of model outputs
            gold_seq - shape: (seq_len) true labels for the sequence

        Outputs:
            target_softmax - shape: (seq_len, vocab_size) softmax over ocd targets
        '''
        # 1. Generate predicted sequence from output_logits
        predicted_seq = torch.argmax(output_logits, dim=1)
        predicted_seq = predicted_seq[:self.cur_t + 1]
        # 2. Build the edit distance table if it isn't already
        self.build_edit_dist_table(predicted_seq, gold_seq)
        # 3. Get the optimal targets for each of tokens in the predicted sequence

        optimal_targets = self.optimal_completion(self.cur_t-1, gold_seq)
        # 4. Return the softmax of the optimal targets
        target = nn.functional.one_hot(optimal_targets.to(torch.int64), num_classes=output_logits.size(1))
        if target.dim() > 1:
            target = sum(target)
        target_softmax = nn.functional.softmax(target.float())
        return target_softmax

### Seq2Seq model ###
class Seq2Seq(nn.Module):
    '''
    A Seq2Seq model using an encoder-dec    oder architecture 
    Args:
        encoder - Encoder to transform raw audio to hidden and cell state
        decoder - Decoder to produce outputs given encoder hidden/cell state
                  and input tokens.
        ix_to_char - The dimension of the hidden state
    '''
    def __init__(self, encoder, decoder, ix_to_char):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.ix_to_char = ix_to_char
     
    def forward(self, audio, target, beta, mode):
        '''
        Inputs:
            audio - shape: (B, C, L), dtype torch.float32
            target - shape: (B, max_seq_len) dtype torch.int64
        where:
            audio is a pytorch tensor containing a batch of preprocessed audio samples
            target is a pytorch tensor containing a batch of target token sequences. All
                      sequences are padded to the same length (max_seq_len)
            B is the batch dimension
            C is the number of features for the audio
            L is the number of timesteps sampled
        
        Outputs :
            outputs - shape: (B, max_seq_len, vocab_size), dtype torch.float32
                             Outputs computed from the current model
            dynamic_oracles - list of CERDynamicOracle computed from s2s model
        '''

        batch_size = target.shape[0]
        max_len = target.shape[1]

        # Run the encoder
        hidden, cell = self.encoder.forward(audio)

        vocab_size = self.decoder.output_dim

        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, max_len, vocab_size)

        # First input to the decoder is the <sos> tokens
        output = target[:, 0]
        outputs[:, 0, 1] = 1
        dynamic_oracles = [CERDynamicOracle() for _ in range(batch_size)]
        # dynamic_oracles = []

        for t in range(1, max_len):
            # Use last hidden state of the encoder as the initial hidden state of the decoder

            new_output, hidden, cell = self.decoder.forward(output, hidden, cell)
            new_output = torch.squeeze(new_output, dim=1)
            uniform_sample = np.random.rand()
            if uniform_sample < beta:
                # 1. If not ocd, this is the gold sequence token
                # 2. Else this is the dynamic oracle target
                # Make sure to use np.random.choice when selecting from optimal completions
                if 'ocd' not in mode:
                    output = target[:, t]
                else:
                    # output = target[:, t]
                    output = [d.get_ocd_softmaxed(o, t) for (d, o, t) in zip(dynamic_oracles, outputs, target)]
                    output = [np.asarray(o).astype('float64') for o in output]
                    output = [o/o.sum() for o in output]
                    output = torch.tensor([np.random.choice(vocab_size, p=o) for o in output])
            else:
                output = torch.argmax(new_output, dim=1)
            outputs[:, t, :] = new_output
        return outputs, dynamic_oracles

### Helper Functions ###
def calculate_cer(output_strings, target_strings):
    '''
    A function that returns the average cer between the output and target strings

    Inputs:
        output_strings - strings outputted by model
        target_strings - target string sequences (either oracle completions or gold tokens)
    Outputs :
        cer - average normalized edit distance (cer) across all string pairs
    ''' 
    cer = 0.0
    for str_out, str_val in zip(output_strings, target_strings):
        if len(str_out) != 0:
            ed = edit_distance(str_out, str_val) / len(str_out)
        else:
            ed = edit_distance(str_out, str_val)
        cer += ed
    cer = cer / len(output_strings)
    return cer 

def output_to_strings(output, ix_to_char):
    '''
    A function takes a tensor of output sequences and converts them to strings.

    Inputs:
        output - (batch_size, max_seq_len, vocab_size)
        ix_to_char - dictionary from index to character
    Outputs :
        output_strings - list of output strings
    ''' 
    output = output.detach()

    output_strings = []
    for sent in output:
        txt = []
        for char in sent:
            char = int(char.max(0)[1].numpy())
            char_txt = ix_to_char[char]
            if char_txt != '<SOS>' and char_txt != '<EOS>' and char_txt != '<PAD>':
                txt.append(char_txt)
        output_strings.append("".join(txt))
    return output_strings

def target_to_strings(target, ix_to_char):
    '''
    A function takes a tensor of target sequences and converts them to strings.

    Inputs:
        target - (batch_size, max_seq_len)
        ix_to_char - dictionary from index to character
    Outputs :
        target_strings - list of target strings 
    ''' 

    target = target.numpy()
    
    target_strings = []
    for sent in target:
        txt = []
        for char in sent:
            char_txt = ix_to_char[char]
            if char_txt != '<SOS>' and char_txt != '<EOS>' and char_txt != '<PAD>':
                txt.append(char_txt) 
        target_strings.append("".join(txt))
    return target_strings

def calculate_loss(output, target, criterion, dynamic_oracles, mode):
    '''
    A function runs the criterion (loss function) between the target and output.
    If the current mode contains 'ocd', dyanmic oracle targets are used instead of gold sequence

    Inputs:
        output - (batch_size, seq_len, vocab_size) model output
        target - (batch_size, seq_len)
        criterion - pytorch loss function
        dynamic_oracles - list of CERDynamicOracle computed from s2s model
        mode - current algorithm mode (e.g ocd, mle)
    Outputs :
        loss - loss computed on outputs compared to target
    ''' 
    
    if 'ocd' not in mode:
        criterion is nn.CrossEntropyLoss
        output = output.transpose(1, 2)
        loss = criterion(output, target)
    else:
        # criterion is nn.KLDivLoss() - inputs to loss must be softmaxed
        output = torch.nn.functional.softmax(output)
        ocd_target = [d.get_ocd_target_from_table(o, t) for (d, o, t) in zip(dynamic_oracles, output, target)]
        batched_target = torch.stack(ocd_target)
        target = torch.tensor(batched_target)
        loss = criterion(output, target)
    return loss
  
def train_one_epoch(s2s, train_loader, optimizer, criterion, mode, beta, ix_to_char):
    '''
    A function that runs one training epoch, iterating across all the datapoints
    in the training dataloader passed in.
    Inputs:
        s2s - an s2s pytorch model
        train_loader - A pytorch dataloader
        optimizer - a pytorch optimizer
        criterion - a pytorch criterion (loss function layer)
        mode - current algorithm mode (e.g ocd, mle)
        beta - beta for mixed policies
        ix_to_char - dictionary from index to character
    '''  
    tr_loss = 0.0
    cer_loss = 0.0
    for _, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        optimizer.zero_grad()
        
        audio, target, _, _ = batch
        output, dynamic_oracles = s2s(audio, target, beta, mode)
                    
        output_strings = output_to_strings(output, ix_to_char)
        target_strings = target_to_strings(target, ix_to_char)
        cer = calculate_cer(output_strings, target_strings)
        
        loss = calculate_loss(output, target, criterion, dynamic_oracles, mode)
        
        loss.backward()
        optimizer.step()

        tr_loss += loss.item()
        cer_loss += cer
    return tr_loss, cer_loss 

def evaluate(s2s, val_loader, criterion, ix_to_char, mode):
    '''
    A function that evaluates the model over all the datapoints from the dataloader passed in
    Inputs:
        model - an s2s pytorch model
        val_loader - A pytorch dataloader for the validation dataset
        criterion - loss function used for the model
        ix_to_char - dictionary from index to character
        mode - current algorithm mode (e.g ocd, mle)
    '''
    with torch.no_grad():
        te_loss = 0.0
        cer_loss = 0.0
        for _, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
            
            audio, target, _, _ = batch
            output, dynamic_oracles = s2s(audio, target, 0, mode)

            output_strings = output_to_strings(output, ix_to_char)
            target_strings = target_to_strings(target, ix_to_char)
            cer = calculate_cer(output_strings, target_strings)

            loss = calculate_loss(output, target, criterion, dynamic_oracles, mode)

            te_loss += loss.item()
            cer_loss += cer
    return te_loss, cer_loss

def train(s2s, train_loader, val_loader, n_epochs, char_to_ix, ix_to_char, mode, beta):
    '''
    A function that trains a model and evaluates it on the validation/val_loader data
    at the end of every training epoch
    Inputs:
        s2s - an s2s pytorch model
        train_loader - A pytorch dataloader containing the training data
        val_loader - A pytorch dataloader containing the validation data
        n_epochs - The number of epochs to train the model for
        char_to_ix - dictionary from character to index
        ix_to_char - dictionary from index to character
        mode - current algorithm mode (e.g ocd, mle)
        beta - beta for mixed policies
    '''  
    optimizer = optim.Adam(s2s.parameters())

    if 'ocd' not in mode:
        criterion = nn.CrossEntropyLoss(ignore_index = char_to_ix["<PAD>"])
    else:
        criterion = nn.KLDivLoss()

    training_ce = []
    training_cer = []
    validation_ce = []
    validation_cer = []

    for epoch in range(n_epochs):
        print (f"---------    Epoch   {epoch}    ---------")
        wandb.log({"Epoch": epoch})

        if 'linear' in mode:
            beta = 1 - np.linspace(0, 1, n_epochs)[epoch]
            wandb.log({"Beta": beta})
        
        # Training 
        tr_loss, cer_loss = train_one_epoch(s2s, train_loader, optimizer, criterion, mode, beta, ix_to_char)
        training_ce.append(tr_loss / len(train_loader))
        training_cer.append(cer_loss / len(train_loader))
        
        # validation
        te_loss, cer_loss = evaluate(s2s, val_loader, criterion, ix_to_char, mode)
        validation_ce.append(te_loss / len(val_loader))
        validation_cer.append(cer_loss / len(val_loader))
        
        wandb.log({"Epoch": epoch, "Training CE": training_ce[-1], "Training CER": training_cer[-1]})
        wandb.log({"Epoch": epoch, "Validation CE": validation_ce[-1], "Validation CER": validation_cer[-1]})

    return training_ce, training_cer, validation_ce,validation_cer

def main(mode, beta):
    #If true, save performance metrics as .pkl files
    save_output = True

    #Training parameters
    n_epochs = 20
    batch_size = 32

    #seq2seq parameters
    hidden_dim = 128
    n_layers = 1
    emb_dim = 256
    if mode == 'mle':
        assert beta == 1.

    #Create vocabulary
    vocab = ['<UNK>', '<SOS>', '<EOS>', '<PAD>','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
        
    vocab_size = len(vocab)

    #Create dictionaries for translating between index and char and vice versa   
    char_to_ix = defaultdict(lambda: None)
    ix_to_char = defaultdict(lambda: None)

    for ix, char in enumerate(vocab):
        char_to_ix[char] = ix
        ix_to_char[ix] = char
        
    train_data = TIMITDataset("asr_data/train", char_to_ix, ix_to_char)
    validation_data = TIMITDataset("asr_data/validation", char_to_ix, ix_to_char)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, collate_fn=pad_batch)
    val_loader = DataLoader(dataset=validation_data, batch_size=batch_size, shuffle=True, collate_fn=pad_batch)

    enc = Encoder(20, hidden_dim, n_layers)
    dec = Decoder(vocab_size, emb_dim, hidden_dim, n_layers)
    s2s = Seq2Seq(enc, dec, ix_to_char)

    training_ce, training_cer, validation_ce,validation_cer = train(s2s, train_loader,val_loader, n_epochs, char_to_ix, ix_to_char, mode, beta)

    if save_output:   
        pickle.dump(training_ce, open(f"{mode}_{beta}_training_ce.pkl", "wb"))
        pickle.dump(training_cer, open(f"{mode}_{beta}_training_cer.pkl", "wb"))
        pickle.dump(validation_ce, open(f"{mode}_{beta}_validation_ce.pkl", "wb"))
        pickle.dump(validation_cer, open(f"{mode}_{beta}_validation_cer.pkl", "wb"))
    
    wandb.log({"Best training CER:": min(training_cer)})
    wandb.log({"Best validation CER:": min(validation_cer)})

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['mle', 'ocd', 'ss', 'ss_linear_decay', 'ocd_linear_decay'])
    parser.add_argument('--beta', type=float, default=1.)
    args = parser.parse_args()
    wandb.login()
    wandb_config = {
        "mode": args.mode,
        "beta": args.beta
    }

    run = wandb.init(project="simple-asr", config=wandb_config, reinit=True)

    main(args.mode, args.beta)
