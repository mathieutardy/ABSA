import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


def labels_to_number(label: str):
    dict_labels_to_number = {
        'positive': 0,
        'negative': 1,
        'neutral': 2,
    }
    return dict_labels_to_number[label]


class TargetedSentimentAnalysisDataset(Dataset):
    def __init__(self, reviews, aspect_terms, targets, tokenizer, max_len):
        self.reviews = reviews
        self.aspect_terms = aspect_terms
        self.tokenizer = tokenizer
        self.targets = targets
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = self.reviews[item]
        aspect_term = self.aspect_terms[item]
        target = self.targets[item]

        tokenized_encoded = self.tokenizer.encode_plus(review, aspect_term,
                                                       add_special_tokens=True,
                                                       return_token_type_ids=True,
                                                       return_attention_mask=True,
                                                       padding='max_length',
                                                       max_length=self.max_len,
                                                       return_tensors='pt')
        return {
            'review': review,
            'input_ids': tokenized_encoded['input_ids'].flatten(),
            'token_type_ids': tokenized_encoded['token_type_ids'].flatten(),
            'attention_masks': tokenized_encoded['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }


def create_data_loader(df, tokenizer, batch_size, max_len):
    ds = TargetedSentimentAnalysisDataset(
        reviews=df.review.to_numpy(),
        aspect_terms=df.aspect_term_category.to_numpy(),
        targets=df.target.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(ds, batch_size=batch_size)


def split_dataframe(df, test_size=0.1):
    df_train, df_val = train_test_split(df, test_size=test_size, random_state=42)
    return df_train, df_val
