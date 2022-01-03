import torch
import torch.functional as f
import torch.nn as nn
import torch.optim as optim
from tqdm.autonotebook import tqdm


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.metric_history = {"train": [], "eval": []}

    def fit(self, train_data, valid_data=None, n_epochs=10, metrics=[]):
        for epoch in range(1, n_epochs + 1):
            self.train()
            self.run_epoch(train_data, epoch, n_epochs, metrics=metrics)
            if valid_data is not None:
                self.eval()
                with torch.no_grad():
                    self.run_epoch(valid_data, epoch, n_epochs, metrics=metrics)

    def after_update(self):
        pass

    def run_epoch(self, data, epoch, n_epochs, metrics=[]):
        total_loss = 0
        for metric in metrics:
            metric.reset()
        with tqdm(total=len(data)) as pbar:
            pbar.set_description(
                ("Train {0}/{1}" if self.training else "Eval {0}/{1}").format(
                    epoch, n_epochs
                )
            )
            for idx, ((x, lengths), y) in enumerate(data):
                self.optimizer.zero_grad()
                lengths = lengths.cpu()
                y_hat = self(x, lengths)
                loss = self.criterion(y_hat, y)
                total_loss += loss.item()
                if self.training:
                    loss.backward()
                    self.optimizer.step()
                    self.after_update()
                metrics_values = dict([metric.update(y_hat, y) for metric in metrics])
                pbar.update(1)
                pbar.set_postfix(loss=total_loss / (idx + 1), **metrics_values)
        metrics_values["loss"] = total_loss / (idx + 1)
        self.metric_history["train" if self.training else "eval"].append(metrics_values)

    def save(self, path):
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metric_history": self.metric_history,
            },
            path,
        )

    def load(self, path):
        loaded_dict = torch.load(path)
        self.load_state_dict(loaded_dict["model_state_dict"])
        self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.metric_history = loaded_dict["metric_history"]


class AVGM(Model):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size, pad_index):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_index)
        self.rnn = nn.LSTM(
            embed_size, hidden_size, bidirectional=True, batch_first=True
        )
        self.fc = nn.Linear(2 * hidden_size, output_size)
        self.optimizer = optim.Adam(self.parameters())
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, tokens, lengths):
        embeds = self.embedding(tokens)
        packed_embeds = nn.utils.rnn.pack_padded_sequence(
            embeds, lengths, batch_first=True
        )
        packed_output, (hidden, cell) = self.rnn(packed_embeds)
        # output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        hidden = torch.cat((hidden[0, :, :], hidden[1, :, :]), dim=1)
        linear = self.fc(hidden)
        return linear


class OrdinalLayer(nn.Module):
    def __init__(self, n_classes, threshold=0.001):
        super().__init__()
        self.n_classes = n_classes
        offsets = torch.Tensor(1, n_classes - 1)
        bias = torch.Tensor(1, 1)
        self.offsets = nn.Parameter(offsets)
        self.bias = nn.Parameter(bias)
        self.threshold = threshold
        nn.init.xavier_uniform_(offsets, gain=1.0)
        nn.init.xavier_uniform_(bias, gain=1.0)

    def forward(self, x):
        batch_size = x.shape[0]
        offsets = f.threshold(self.offsets, self.threshold, self.threshold)
        offsets = torch.cumsum(offsets, dim=1)
        bias = self.bias + offsets
        logit = torch.add(x, bias)
        probc = torch.exp(logit) / (1 + torch.exp(logit))
        zeros = torch.zeros(batch_size, 1).to("cuda")
        ones = torch.ones(batch_size, 1).to("cuda")
        probc = torch.cat((zeros, probc, ones), dim=1)
        prob = torch.diff(probc, n=1, dim=1)
        logprob = torch.log(prob)
        return logprob


class AVGMOrdinal(Model):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size, pad_index):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_index)
        self.rnn = nn.LSTM(
            embed_size, hidden_size, bidirectional=True, batch_first=True
        )
        self.fc = nn.Linear(2 * hidden_size, 1)
        self.ordinal = OrdinalLayer(output_size)
        self.optimizer = optim.Adam(self.parameters())
        self.criterion = nn.NLLLoss()

    def forward(self, tokens, lengths):
        embeds = self.embedding(tokens)
        packed_embeds = nn.utils.rnn.pack_padded_sequence(
            embeds, lengths, batch_first=True
        )
        packed_output, (hidden, cell) = self.rnn(packed_embeds)
        # output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        hidden = torch.cat((hidden[0, :, :], hidden[1, :, :]), dim=1)
        linear = self.fc(hidden)
        logprobs = self.ordinal(linear)
        return logprobs
