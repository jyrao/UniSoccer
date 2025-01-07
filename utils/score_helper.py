from pycocoevalcap.bleu.bleu import Bleu as Bleuold
from pycocoevalcap.bleu.bleu_scorer import BleuScorer
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
import csv, argparse
import numpy as np

class Bleu(Bleuold):
    # Same as SoccerNet Evaluation
    def compute_score(self, gts, res):

        assert(gts.keys() == res.keys())
        imgIds = gts.keys()

        bleu_scorer = BleuScorer(n=self._n)
        for id in imgIds:
            hypo = res[id]
            ref = gts[id]
            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) >= 1)
            bleu_scorer += (hypo[0], ref)
        score, scores = bleu_scorer.compute_score(option='closest', verbose=0)
        
        return score, scores

def calculate_metrics_of_set(references, hypotheses):
    # Input form:
    # refe = {0:["My name is Marry"], 1:["His name is Jack"]}
    # hypo = {0:["My name is Marry"], 1:["This is a dog"]}

    # Initialize scorers
    bleu4_scorer = Bleu(4)
    meteor_scorer = Meteor()
    rouge_scorer = Rouge()
    cider_scorer = Cider()

    # Calculate BLEU scores
    bleu4_score, _ = bleu4_scorer.compute_score(references, hypotheses)

    # Calculate METEOR scores
    meteor_score, _ = meteor_scorer.compute_score(references, hypotheses)

    # Calculate ROUGE scores, focusing on ROUGE-L
    _, rouge_scores = rouge_scorer.compute_score(references, hypotheses)
    rouge_l_score = rouge_scores.mean()

    # Calculate CIDER scores
    cider_score, _ = cider_scorer.compute_score(references, hypotheses)

    return {
        "BLEU-1": bleu4_score[0]*100,
        "BLEU-4": bleu4_score[3]*100,
        "METEOR": meteor_score*100,
        "ROUGE-L": rouge_l_score*100,
        "CIDER": cider_score*100
    }

