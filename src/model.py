import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertModel
import torch.nn.functional as F


class BertAsc(nn.Module):
    """Our model"""

    def __init__(self, model_name="activebus/BERT_Review", n_classes=3):
        super(BertAsc, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, token_type_ids, attention_masks):
        output = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_masks,
            return_dict=True
        )
        x = output['pooler_output']
        x = self.dropout(x)
        x = self.linear(x)

        return F.softmax(x, dim=1)


# train functions

def train_epoch(model: BertAsc, data_loader: DataLoader, loss_fn, optimizer, device, scheduler,
                n_examples):
    model = model.train()

    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        token_type_ids = d["token_type_ids"].to(device)
        attention_masks = d["attention_masks"].to(device)
        targets = d["targets"].to(device)

        outputs = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_masks=attention_masks,
        )

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()

    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            token_type_ids = d["token_type_ids"].to(device)
            attention_masks = d["attention_masks"].to(device)
            targets = d["targets"].to(device)

            outputs = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_masks=attention_masks
            )
            _, preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)