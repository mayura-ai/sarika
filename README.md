def collate_fn(batch):
    X = []
    Y = []
    for item in batch:
        split_index = torch.where(item == SEP_TOKEN_ID)[0][0]
        X.append(item[: split_index + 1])
        Y.append(item[split_index + 1 :])
    fin_X = []
    fin_Y = []
    for x, y in zip(X, Y):
        for i in range(len(y) - 1):
            fin_X.append(torch.cat((x, y[:i])))
            fin_Y.append(y[i:])
    return torch.nn.utils.rnn.pad_sequence(fin_X), torch.nn.utils.rnn.pad_sequence(
        fin_Y
    )
