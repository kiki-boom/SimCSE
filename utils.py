import csv

delimiters = {
    "ATEC_CCKS": "\t",
    "BQ": ",",
    "LCQMC": "\t",
}


def load_csv_data(filename, delimiter="\t"):
    D = []
    with open(filename, encoding="utf-8") as f:
        f_csv = csv.reader(f, delimiter=delimiter)
        line_num = 0
        for row in f_csv:
            line_num += 1
            if line_num == 1:
                continue
            D.append((row[0], row[1], float(row[2])))
    return D


def load_data(filename, header=False):
    """加载数据（带标签）
    单条格式：(文本1, 文本2, 标签)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        if header:
            _ = f.readline()
        l = f.readline()
        while l:
            l = l.strip().split('\t')
            if len(l) == 3:
                D.append((l[0], l[1], float(l[2])))
            l = f.readline()
    return D


def convert_to_ids(text, tokenizer, maxlen=64):
    tokens = tokenizer.tokenize(text)
    if len(tokens) > maxlen - 2:
        tokens = tokens[: maxlen - 2]
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    mask = [1] * len(tokens)
    while len(token_ids) < maxlen:
        token_ids.append(0)
        mask.append(0)
    return token_ids, mask
