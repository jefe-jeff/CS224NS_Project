"""
To merge Librispeech with forced alignment data, using rsync. Example:
rsync -a -P /Users/jbrown/Documents/CS224NS_Data/LibriSpeech-Alignments/LibriSpeech/train-clean-100/ /Users/jbrown/Documents/CS224NS_Data/LibriSpeech/train-clean-100
"""

import os
from typing import Tuple, Union
from pathlib import Path

import torchaudio
import librosa
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio.datasets.utils import (
    download_url,
    extract_archive,
)
from transformers import RobertaTokenizerFast

URL = "train-clean-100"
FOLDER_IN_ARCHIVE = "LibriSpeech"
_CHECKSUMS = {
    "http://www.openslr.org/resources/12/dev-clean.tar.gz":
    "76f87d090650617fca0cac8f88b9416e0ebf80350acb97b343a85fa903728ab3",
    "http://www.openslr.org/resources/12/dev-other.tar.gz":
    "12661c48e8c3fe1de2c1caa4c3e135193bfb1811584f11f569dd12645aa84365",
    "http://www.openslr.org/resources/12/test-clean.tar.gz":
    "39fde525e59672dc6d1551919b1478f724438a95aa55f874b576be21967e6c23",
    "http://www.openslr.org/resources/12/test-other.tar.gz":
    "d09c181bba5cf717b3dee7d4d592af11a3ee3a09e08ae025c5506f6ebe961c29",
    "http://www.openslr.org/resources/12/train-clean-100.tar.gz":
    "d4ddd1d5a6ab303066f14971d768ee43278a5f2a0aa43dc716b0e64ecbbbf6e2",
    "http://www.openslr.org/resources/12/train-clean-360.tar.gz":
    "146a56496217e96c14334a160df97fffedd6e0a04e66b9c5af0d40be3c792ecf",
    "http://www.openslr.org/resources/12/train-other-500.tar.gz":
    "ddb22f27f96ec163645d53215559df6aa36515f26e01dd70798188350adcb6d2"
}


def load_librispeech_item(tokenizer, fileid: str,
                          path: str,
                          ext_audio: str,
                          ext_txt: str) -> Tuple[Tensor, int, str, int, int, int]:
    speaker_id, chapter_id, utterance_id = fileid.split("-")

    file_text = speaker_id + "-" + chapter_id + ext_txt
    file_text = os.path.join(path, speaker_id, chapter_id, file_text)

    #file_text_alignment = speaker_id + "-" + chapter_id + ".alignment.txt"
    #file_text_alignment = os.path.join(path, speaker_id, chapter_id, file_text_alignment)

    fileid_audio = speaker_id + "-" + chapter_id + "-" + utterance_id
    file_audio = fileid_audio + ext_audio
    file_audio = os.path.join(path, speaker_id, chapter_id, file_audio)

    # Load audio
    waveform, sample_rate = librosa.load(file_audio)

    # Load text
    with open(file_text) as ft:
        
        for line in ft:
            fileid_text, utterance = line.strip().split(" ", 1)
            if fileid_audio == fileid_text:
                break
        else:
            # Translation not found
            raise FileNotFoundError("Translation not found for " + fileid_audio)

    """
    with open(file_text) as ft:
        with open(file_text_alignment) as fta:
            for line, line_align in zip(ft,fta):
                fileid_text, utterance = line.strip().split(" ", 1)
                fileid_text_align, utterance_align,  time_align= line_align.strip().split(" ", 2)
                if fileid_audio == fileid_text:
                    break
            else:
                # Translation not found
                raise FileNotFoundError("Translation not found for " + fileid_audio)
    """
    #time_alignment = Tensor(list(map(float, time_align[1:-1].split(","))))

    
    tokenized_utterance = Tensor(tokenizer(utterance).input_ids).long()
    return (
        waveform.flatten(),
        len(waveform.flatten()), 
        tokenized_utterance, 
        len(tokenized_utterance)
    )





class librispeech_dataset(Dataset):
    """Create a Dataset for LibriSpeech.

    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        url (str, optional): The URL to download the dataset from,
            or the type of the dataset to dowload.
            Allowed type values are ``"dev-clean"``, ``"dev-other"``, ``"test-clean"``,
            ``"test-other"``, ``"train-clean-100"``, ``"train-clean-360"`` and
            ``"train-other-500"``. (default: ``"train-clean-100"``)
        folder_in_archive (str, optional):
            The top-level directory of the dataset. (default: ``"LibriSpeech"``)
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
    """

    _ext_txt = ".trans.txt"
    _ext_audio = ".flac"

    def __init__(self,
                 root: Union[str, Path],
                 url: str = URL,
                 folder_in_archive: str = FOLDER_IN_ARCHIVE,
                 download: bool = False) -> None:

        if url in [
            "dev-clean",
            "dev-other",
            "test-clean",
            "test-other",
            "train-clean-100",
            "train-clean-360",
            "train-other-500",
        ]:

            ext_archive = ".tar.gz"
            base_url = "http://www.openslr.org/resources/12/"

            url = os.path.join(base_url, url + ext_archive)

        # Get string representation of 'root' in case Path object is passed
        root = os.fspath(root)

        basename = os.path.basename(url)
        archive = os.path.join(root, basename)

        basename = basename.split(".")[0]
        folder_in_archive = os.path.join(folder_in_archive, basename)
        self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
        self._path = os.path.join(root, folder_in_archive)

        if download:
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    checksum = _CHECKSUMS.get(url, None)
                    download_url(url, root, hash_value=checksum)
                extract_archive(archive)

        self._walker = sorted(str(p.stem) for p in Path(self._path).glob('*/*/*' + self._ext_audio))

    def load_librispeech_item(self, fileid,path,ext_audio,ext_txt):
        return load_librispeech_item(self.tokenizer,fileid,path,ext_audio,ext_txt)
    
    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, int, int, int]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            tuple: ``(waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id)``
        """
        fileid = self._walker[n]
        return self.load_librispeech_item(fileid, self._path, self._ext_audio, self._ext_txt)


    def __len__(self) -> int:
        return len(self._walker)
