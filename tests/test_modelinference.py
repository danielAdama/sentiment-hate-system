import pytest
from nltk import pos_tag
import sys
sys.path.append('/home/daniel/Desktop/programming/pythondatascience/datascience/NLP/sentiment-hate-system/src')
from pipeline.modelinference import ModelInference
from config import config

def test_get_pos_tag():
    mi = ModelInference('vectorizerV2.bin', 'modelV2.bin')
    pos = mi.get_pos_tag("half way through the website now and #allgoingwell very  ")
    assert pos == [('half', 'NN'), ('way', 'NN'), ('through', 'IN'), ('the', 'DT'), ('website', 'NN'), ('now', 'RB'), ('and', 'CC'), ('#', '#'), ('allgoingwell', 'RB'), ('very', 'RB')]

