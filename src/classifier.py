from collections import defaultdict

import torch
from torch import nn
import pandas as pd
from transformers import BertTokenizer, get_linear_schedule_with_warmup, AdamW

from dataloader import labels_to_number, create_data_loader, split_dataframe
from model import BertAsc, train_epoch, eval_model

EPOCHS = 3
BATCH_SIZE = 16
PRETRAIN_MODEL = "activebus/BERT_Review"
MAX_LEN = 120
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class Classifier:
    """The Classifier"""

    def read_data(self, trainfile: str):
        df = pd.read_csv(trainfile, sep='\t', lineterminator='\n', header=None,
                         names=["sentiment", "aspect_category", "target_term", "pos", "review"])
        df.review = df.review.apply(lambda x: x.strip("\r"))
        return df

    def preprocess(self, df: pd.DataFrame):
        df['aspect_term_category'] = df['aspect_category'] + '#' + df['target_term']
        df['target'] = df['sentiment'].map(labels_to_number)
        return df

    def train(self, trainfile):
        """Trains the classifier model on the training set stored in file trainfile"""
        df = self.read_data(trainfile)
        preprocessed_df = self.preprocess(df)

        df_train, df_val = split_dataframe(preprocessed_df)

        tokenizer = BertTokenizer.from_pretrained(PRETRAIN_MODEL)
        train_data_loader = create_data_loader(df_train, tokenizer, BATCH_SIZE, MAX_LEN)
        val_data_loader = create_data_loader(df_val, tokenizer, BATCH_SIZE, MAX_LEN)

        model = BertAsc(model_name=PRETRAIN_MODEL, n_classes=3)

        model = model.to(device)
        optimizer = AdamW(model.parameters(), lr=2e-5)
        total_steps = len(train_data_loader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        loss_fn = nn.CrossEntropyLoss()

        history = defaultdict(list)
        best_accuracy = 0

        for epoch in range(EPOCHS):

            print(f'Epoch {epoch + 1}/{EPOCHS}')
            print('-' * 10)

            train_acc, train_loss = train_epoch(model, train_data_loader, loss_fn, optimizer,
                                                device, scheduler, len(df_train))

            print(f'Train loss {train_loss} accuracy {train_acc}')

            val_acc, val_loss = eval_model(model, val_data_loader, loss_fn, device, len(df_val))

            print(f'Val   loss {val_loss} accuracy {val_acc}')

            history['train_acc'].append(train_acc)
            history['train_loss'].append(train_loss)
            history['val_acc'].append(val_acc)
            history['val_loss'].append(val_loss)

            if val_acc > best_accuracy:
                torch.save(model.state_dict(), '../resources/best_model_state.bin')
                best_accuracy = val_acc

    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """

        df = self.read_data(datafile)
        preprocessed_df = self.preprocess(df)
        tokenizer = BertTokenizer.from_pretrained(PRETRAIN_MODEL)
        data_loader = create_data_loader(preprocessed_df, tokenizer, BATCH_SIZE, MAX_LEN)

        state_dict = torch.load('../resources/best_model_state.bin')
        model = BertAsc(PRETRAIN_MODEL)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        predictions = []

        with torch.no_grad():
            for d in data_loader:
                input_ids = d["input_ids"].to(device)
                token_type_ids = d["token_type_ids"].to(device)
                attention_masks = d["attention_masks"].to(device)
                outputs = model(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_masks=attention_masks
                )
                _, preds = torch.max(outputs, dim=1)

                predictions.extend(preds)

        dict_numbers_to_label = {
            0: 'positive',
            1: 'negative',
            2: 'neutral',
        }

        predictions = torch.stack(predictions).to(device).numpy()
        predictions = [dict_numbers_to_label[pred] for pred in predictions]
        return predictions
