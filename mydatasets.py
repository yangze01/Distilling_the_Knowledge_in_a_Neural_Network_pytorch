import re
import os
import random
import tarfile
import urllib
from torchtext import data
def get_file_name(file_dir, name="dirs"):
    res_dirs = list()
    res_files = list()
    for root, dirs, files in os.walk(file_dir):
        res_dirs += dirs
        res_files += files

    if name == "dirs":
        return res_dirs
    else:
        return res_files


def get_dataset_iter(args, data_name = "NewsGroup"):
    print("Loading data...")
    TEXT = data.ReversibleField(lower=True, include_lengths=True, batch_first=True)
    LABEL = data.Field(sequential=False)
    train, test = NewsGroup.splits(TEXT, LABEL)

    print("Building vocabulary...")
    TEXT.build_vocab(train)
    LABEL.build_vocab(train)

    print(type(TEXT.vocab.stoi))
    train_iter, test_iter = data.BucketIterator.splits((train, test), sort_key = lambda x:len(x.text),
                                                       sort_within_batch=True,
                                                       batch_size=args.batch_size, device=-1,
                                                       repeat = False)
    args.embed_num = len(TEXT.vocab)
    args.class_num = len(LABEL.vocab) - 1
    print("Loading data finish...")
    return train_iter, test_iter

def replace(matched):
    return " " + matched.group("m") + " "


def tokenize_line_en(line):
    line = re.sub(r"\t", "", line)
    line = re.sub(r"^\s+", "", line)
    line = re.sub(r"\s+$", "", line)
    line = re.sub(r"<br />", "", line)
    line = re.sub(r"(?P<m>\W)", replace, line)
    line = re.sub(r"\s+", " ", line)
    return line.split()


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


class BasicDataset(data.Dataset):
    """Defines a Dataset loaded from a downloadable tar archive.
    Attributes:
        url: URL where the tar archive can be downloaded.
        filename: Filename of the downloaded tar archive.
        dirname: Name of the top-level directory within the zip archive that
            contains the data files.
    """

    @classmethod
    def download_or_unzip(cls, root):
        if not os.path.exists(root):
            os.mkdir(root)
        path = os.path.join(root, cls.dirname)
        if not os.path.isdir(path):
            tpath = os.path.join(root, cls.filename)
            if not os.path.isfile(tpath):
                print('downloading')
                urllib.request.urlretrieve(cls.url, tpath)
            with tarfile.open(tpath, 'r') as tfile:
                print('extracting')
                tfile.extractall(root)
        return os.path.join(path, '')
    @classmethod
    def splits(cls, text_field, label_field, dev_ratio=.1, shuffle=True, root='./data/', **kwargs):
        path = cls.download_or_unzip(root)
        examples = cls(text_field, label_field, path=path, **kwargs).examples
        if shuffle: random.shuffle(examples)
        dev_index = -1 * int(dev_ratio*len(examples))

        return (cls(text_field, label_field, examples=examples[:dev_index]),
                cls(text_field, label_field, examples=examples[dev_index:]))



class NewsGroup(BasicDataset):

    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/20newsgroups-mld/mini_newsgroups.tar.gz"
    url2 = "http://archive.ics.uci.edu/ml/machine-learning-databases/20newsgroups-mld/20_newsgroups.tar.gz"
    filename = 'mini_newsgroups.tar.gz'
    dirname = 'mini_newsgroups'

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, label_field, path=None, examples=None, **kwargs):
        text_field.preprocessing = data.Pipeline(clean_str)
        fields = [('text', text_field), ('label', label_field)]
        path = self.dirname if path is None else path

        if examples is None:
            examples = []
            class_dirs = get_file_name(path)
            for class_dir_name in class_dirs:
                class_dir_path = os.path.join(path, class_dir_name)
                file_names = get_file_name(class_dir_path, "files")
                for file in file_names:
                    file_path = os.path.join(class_dir_path, file)
                    with open(file_path, errors='ignore') as f:
                        raw_data = f.read()
                        if len(raw_data.split(' ')) > 100:
                            continue
                        examples += [data.Example.fromlist([raw_data, class_dir_name], fields)]
        super(NewsGroup, self).__init__(examples, fields, **kwargs)

