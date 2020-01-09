from collections import defaultdict

class Scores(object):
    def __init__(self):
        self.missing = 0
        self.spurious = 0
        self.correct = 0
        self.incorrect = 0
        self.partial = 0


    @property
    def precision(self):
        if self.actual == 0:
            return -1.0
        return (self.correct + (float(self.partial) / 2)) / self.actual

    @property
    def recall(self):
        if self.possible == 0:
            return -1.0
        return (self.correct + (float(self.partial) / 2)) / self.possible

    @property
    def f1(self):
        p = self.precision
        r = self.recall
        if p + r == 0:
            return float('nan')
        return (2 * p * r) / (p + r)

    @property
    def actual(self):
        return self.correct + self.partial + self.incorrect + self.spurious

    @property
    def possible(self):
        return self.correct + self.partial + self.incorrect + self.missing

    def increment(self, count_type):
        setattr(self, count_type, getattr(self, count_type) + 1)

    def to_dict(self):
        return {
            'missing': self.missing,
            'spurious': self.spurious,
            'correct': self.correct,
            'incorrect': self.incorrect,
            'partial': self.partial,
            'actual': self.actual,
            'possible': self.possible,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1
        }

    def __str__(self):
        return "[missing={}; spurious={}; correct={}; incorrect={}; partial={}; precision={}; recall={}; f1={}; actual={}; possible={}]".format(self.missing, self.spurious, self.correct, self.incorrect, self.partial, self.precision, self.recall, self.f1, self.actual, self.possible)

def get_chunk_type(tok, idx_to_tag):
    if tok not in idx_to_tag:
        #print(f"\t{tok} not in {idx_to_tag}")
        tag_name = 'O'
    else:
        tag_name = idx_to_tag[tok]
        #print(f"\t{tok} in {idx_to_tag}")
    return tag_name.split('-')[-1]

class ScoreBookkeeper(object):
    __ALL = "ALL"

    def __init__(self):
        self._types_to_scores = defaultdict(lambda: Scores())
        self._confusion_matrix = defaultdict(lambda: defaultdict(lambda: 0))

    def add_confusion(self, expected, actual):
        self._confusion_matrix[expected][actual] += 1

    def record(self, count_type, entity_type=None):
        self._types_to_scores[ScoreBookkeeper.__ALL].increment(count_type)
        if entity_type is not None:
            self.record_no_all(count_type, entity_type)

    def record_no_all(self, count_type, entity_type):
        self._types_to_scores[entity_type].increment(count_type)

    def record_correct(self, entity_type):
        self.record('correct', entity_type)
        self.add_confusion(entity_type, entity_type)

    def record_spurious(self, entity_type):
        self.record('spurious', entity_type)
        self.add_confusion("NONE", entity_type)

    def record_incorrect(self, ref_type, pred_type):
        self.record('incorrect')
        self.record_no_all('spurious', pred_type)
        self.record_no_all('missing', ref_type)
        self.add_confusion(ref_type, pred_type)

    def record_missing(self, entity_type):
        self.record('missing', entity_type)
        self.add_confusion(entity_type, "NONE")

    def record_partial(self, entity_type):
        self.record('partial', entity_type)
        self.add_confusion(entity_type, entity_type)
        self.add_confusion(entity_type, "NONE")

    @property
    def all_scores(self):
        return self._types_to_scores[ScoreBookkeeper.__ALL]

    @property
    def confusion_matrix(self):
        return self._confusion_matrix

    def class_scores(self, c):
        return self._types_to_scores[c]

    @property
    def classes(self):
        return list(self._types_to_scores)

    def to_dict(self):
        return {k: v.to_dict() for (k, v) in self._types_to_scores.items()}

# Port of perceptron-training doc scorer
def score_sequence(gold, predicted, scores=None):
    # Inputs are from get_chunks
    gold_idx = 0
    predicted_idx = 0
    prev_ref = None
    prev_test = None
    if scores is None:
        scores = ScoreBookkeeper()

    while (gold_idx < len(gold)) and (predicted_idx < len(predicted)):
        ref = gold[gold_idx]
        test = predicted[predicted_idx]
        if prev_ref is not None and prev_ref[1] > ref[1]:
            raise RuntimeError("Mentions out of order: {}, {}".format(prev_ref, ref))
        if prev_test is not None and prev_test[1] > test[1]:
            raise RuntimeError("Mentions out of order: {}, {}".format(prev_test, test))

        if ref[2] <= test[1]:
            scores.record_missing(ref[0])
            gold_idx += 1
        elif ref[1] >= test[2]:
            scores.record_spurious(test[0])
            predicted_idx += 1
        else:
            ref_type = ref[0]
            test_type = test[0]

            if ref_type == test_type:
                if (ref[1] == test[1]) and (ref[2] == test[2]):
                    scores.record_correct(ref_type)
                else:
                    scores.record_partial(ref_type)
            else:
                scores.record_incorrect(ref_type, test_type)
            gold_idx += 1
            predicted_idx += 1

        prev_ref = ref
        prev_test = test
    for i in range(gold_idx, len(gold)):
        scores.record_missing(gold[i][0])
    for i in range(predicted_idx, len(predicted)):
        scores.record_spurious(predicted[i][0])
    return scores


def get_chunks(seq, tag_lexicon, inv_tag_lexicon=None):
    """
    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tag_lexicon: dict["O"] = 4
        inv_tag_lexicon: the inverse of tag_lexicon (for efficiency)
    Returns:
        list of (chunk_type, chunk_start, chunk_end)
    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]
    """
    defaults = []
    # Which default will be present depends on whether
    # we're doing IOB or POS-tagging
    if '[PAD]' in tag_lexicon:
        defaults += [tag_lexicon['[PAD]']]
    if 'O' in tag_lexicon:
        defaults += [tag_lexicon['O']]
    chunks = []
    if inv_tag_lexicon is None:
        inv_tag_lexicon = {v: k for (k, v) in tag_lexicon.items()}
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok in defaults and chunk_type is not None:
            #print(f"{tok} in defaults ({defaults}) and chunk_type ({chunk_type}) is not None")
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok not in defaults:
            #print(f"{tok} not in defaults ({defaults})")
            tok_chunk_type = get_chunk_type(tok, inv_tag_lexicon)
            if chunk_type is None:
                #print(f"\tchunk_type is None: --> {tok_chunk_type}")
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tag_lexicon.get(tok, 'O')[0] == "B":
                #print(f"\ttok_chunk_type ({tok_chunk_type}) != chunk_type ({chunk_type}) or tag_lex ({tag_lexicon.get(tok, 'O')[0]}) == 'B'")
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
            #else:
            #    print(f"\tpass (tok_chunk_type={tok_chunk_type})")
        else:
            #print(f"tok={tok}; pass")
            pass
    # end condition
    if chunk_type is not None:
        #print("not-none")
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)
    return chunks