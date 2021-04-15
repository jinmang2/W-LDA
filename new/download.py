from datasets import load_dataset

from ..wikitext import _DATA_URL


class DataManager:

    def __init__(self, dataset, data_dir="data", rmtree=False):
        self.cwd = os.getcwd()
        self.dataset = dataset
        self.data_dir = os.path.join(self.cwd, data_dir)
        self.rmtree = rmtree

    def download(self, website, saveto=None):
        if saveto is None:
            saveto = os.getcwd()
        os.system(f"curl -O {website}")
        out_file = website.split("/")[-1]
        if os.name == "nt":
            os.system("powershell.exe Expand-Archive "
                      f"-LiteralPath {out_file} "
                      f"-DestinationPath {saveto}")
        else:
            os.system(f"unzip {out_file} -d {saveto}")

    def __enter__(self):
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
        data_dir = os.path.join(self.data_dir, self.dataset)
        if self.rmtree and os.path.exists(data_dir):
            shutil.rmtree(data_dir)
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        os.chdir(data_dir)
        return self

    def __exit__(self, type, value, traceback):
        os.chdir(self.cwd)


class Wikitext103DataManager(DataManager):
    """ Data Download and  """

    def script_to_docs(self, input_dir, token_file):
        res = []
        filename = os.path.join(input_dir, token_file)
        with open(filename, mode="r", encoding="utf-8") as f:
            for l in f:
                line = l.strip()
                if self.is_document_start(line):
                    res.append(line)
                elif line:
                    res[-1] = res[-1] + " " + line
        return res

    @staticmethod
    def is_document_start(line):
        if len(line) < 4:
            return False
        if line[0] is '=' and line[-1] is '=':
            if line[2] is not '=':
                return True
            else:
                return False
        else:
            return False
