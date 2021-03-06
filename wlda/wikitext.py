"""TODO(wikitext): Add a description here."""

from __future__ import absolute_import, division, print_function

import os

import datasets


# TODO(wikitext): BibTeX citation
_CITATION = """\
@InProceedings{wikitext,
    author={Stephen, Merity and Caiming ,Xiong and James, Bradbury and Richard Socher}
    year={2016}
}
"""

# TODO(wikitext):
_DESCRIPTION = """\
 The WikiText language modeling dataset is a collection of over 100 million tokens extracted from the set of verified
 Good and Featured articles on Wikipedia. The dataset is available under the Creative Commons Attribution-ShareAlike License.
"""
_URL = "https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/"
_DATA_URL = "https://s3.amazonaws.com/research.metamind.io/wikitext"


class WikitextConfig(datasets.BuilderConfig):
    """BuilderConfig for GLUE."""

    def __init__(self, data_url, **kwargs):
        """BuilderConfig for Wikitext

        Args:
          data_url: `string`, url to the dataset (word or raw level)
          **kwargs: keyword arguments forwarded to super.
        """
        super(WikitextConfig, self).__init__(
            version=datasets.Version(
                "1.0.0",
            ),
            **kwargs,
        )
        self.data_url = data_url


class Wikitext(datasets.GeneratorBasedBuilder):
    """TODO(wikitext_103): Short description of my dataset."""

    # TODO(wikitext_103): Set up version.
    VERSION = datasets.Version("0.1.0")
    BUILDER_CONFIGS = [
        WikitextConfig(
            name="wikitext-103-v1",
            data_url=_DATA_URL + "/" + "wikitext-103-v1.zip",
            description="raw level dataset. The raw tokens before the addition of <unk> tokens. "
            "They should only be used for character level work or for creating newly derived datasets.",
        ),
    ]

    def _info(self):
        features = datasets.Features(
            {
                "text": datasets.Value("string")
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_URL,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # dl_manager is a datasets.download.DownloadManager that can be used to
        # download and extract URLs
        if self.config.name == "wikitext-103-v1":
            data_file = dl_manager.download_and_extract(self.config.data_url)
            data_dir = os.path.join(data_file, "wikitext-103")
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "data_file": os.path.join(data_dir, "wiki.test.tokens"), 
                        "split": "test"
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "data_file": os.path.join(data_dir, "wiki.train.tokens"), 
                        "split": "train"
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "data_file": os.path.join(data_dir, "wiki.valid.tokens"), 
                        "split": "valid"
                    },
                ),
            ]

    def _generate_examples(self, data_file, split):
        """ Yields examples. """
        idx = 0
        sample = ""
        with open(data_file, encoding="utf-8") as f:
            for row in f:
                line = row.strip()
                if self.is_document_start(line):
                    if sample:
                        yield idx, {"text": sample}
                        idx += 1
                        sample = ""
                    sample = sample + line
                elif line:
                    sample = sample + " " + line
            yield idx, {"text": sample}

    @staticmethod                     
    def is_document_start(line):
        if len(line) < 4:
            return False
        if line[0] is "=" and line[-1] is "=":
            if line[2] is not "=":
                return True
            else:
                return False
        else:
            return False
