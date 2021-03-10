import re
from tokenizer import tokenizer         # << https://github.com/erikavaris/tokenizer

from utils.scrape_text_utils import extract_emoticons_and_emojis, apply_emo_mapper, get_abbreviations

class RawTextPreprocessor():
    def __init__(self, sentences):
        self.sentences = sentences      # list of strings

    def apply_steps(self, steps):
        num_steps = len(steps)
        print('begin preprocessing...')
        for i, step in enumerate(steps):
            if step == '--prep-verbs':
                print('   + step %d/%d >> applying verbs regularization... ' % (i+1,num_steps))
                self.sentences = self.apply_verbs_regularization(self.sentences)
            elif step == '--prep-lowering':
                print('   + step %d/%d >> applying text lowering... ' % (i+1,num_steps))
                self.sentences = self.apply_text_lowering(self.sentences)
            elif step == '--prep-emoticon-emojis':
                print('   + step %d/%d >> applying emoticon and emojis substitution... ' % (i+1,num_steps))
                self.sentences = self.apply_emoticons_emojis_substitution(self.sentences)
            elif step == '--prep-abbreviations':
                print('   + step %d/%d >> applying abbreviations substitution... ' % (i+1,num_steps))
                self.sentences = self.apply_abbreviations_substitution(self.sentences)
            elif step == '--prep-elongations':
                print('   + step %d/%d >> applying elongations removal... ' % (i+1,num_steps))
                self.sentences = self.apply_elongations_removal(self.sentences)
            elif step == '--prep-misspellings':
                print('   + step %d/%d >> applying misspellings correction... ' % (i+1,num_steps))
                self.sentences = self.apply_misspellings_correction(self.sentences)
        print('end preprocessing...\n')
        return self

    def apply_verbs_regularization(self, sentences_to_be_processed):
        T = tokenizer.TweetTokenizer(regularize=True)
        regularized_sentences = []
        for sentence in sentences_to_be_processed:
            new_sentence = ' '.join(T.tokenize(sentence)) 
            regularized_sentences.append(new_sentence)
        return regularized_sentences

    def apply_text_lowering(self, sentences_to_be_processed):
        return list(map(lambda s: s.lower(), sentences_to_be_processed))

    def apply_emoticons_emojis_substitution(self, sentences_to_be_processed):
        emoticons_mapper, emojis_mapper = extract_emoticons_and_emojis()
        sentences_without_emo = []
        for sentence in sentences_to_be_processed:
            # removing emoticons
            new_sentence = apply_emo_mapper(sentence, emoticons_mapper)
            # removing emojis
            new_sentence = apply_emo_mapper(new_sentence, emojis_mapper)
            sentences_without_emo.append(new_sentence)
        return sentences_without_emo

    def apply_abbreviations_substitution(self, sentences_to_be_processed):
        abbreviations = get_abbreviations()
        sentences_without_abbreviations = []
        for sentence in sentences_to_be_processed:
            new_sentence = sentence
            for abbr in abbreviations.keys():
                # checks for abbr at sentence beginning
                new_sentence = re.sub(r'^'+abbr+r'[^a-zA-Z\d]', abbreviations[abbr]+' ', new_sentence)
                # checks for abbr at sentence end
                new_sentence = re.sub(r'[^a-zA-Z\d]'+abbr+r'$', ' '+abbreviations[abbr], new_sentence)
                # checks for abbr at sentence middle
                new_sentence = re.sub(r'[^a-zA-Z\d]'+abbr+r'[^a-zA-Z\d]', ' '+abbreviations[abbr]+' ', new_sentence)
            sentences_without_abbreviations.append(new_sentence)
        return sentences_without_abbreviations

    def apply_elongations_removal(self, sentences_to_be_processed):
        # this step will bring to two consecutive identical letters, so 'eloooongation' will become 'eloongation'
        # a misspelling step can solve the doubles problem whenever it is necessary
        repeat_pattern = re.compile(r'(\w)\1+')
        match_substitution = r'\1\1'

        sentences_without_elongations = []
        for sentence in sentences_to_be_processed:
            new_sentence = sentence
            new_sentence = repeat_pattern.sub(match_substitution, new_sentence)
            sentences_without_elongations.append(new_sentence)
        return sentences_without_elongations

    def apply_misspellings_correction(self, sentences_to_be_processed):
        sentences_without_misspellings = []
        for sentence in sentences_without_elongations:
            new_sentence = sentence
            for tok in T.tokenize(new_sentence):
                suggestions = hobj.suggest(tok)
                if not hobj.spell(tok) and tok.isalnum() and len(tok)>2 and len(suggestions)>0 and (not('h' in tok)):
                    new_sentence = new_sentence.replace(tok, suggestions[0]) 
            sentences_without_misspellings.append(sentence)
        return sentences_without_misspellings

    def get_sentences(self, tokenized=False):
        if tokenized: 
            T = tokenizer.TweetTokenizer(regularize=True)
            return [T.tokenize(sentence) for sentence in self.sentences]            
        else:
            return self.sentences

    def save_to(self, file_path):
        print("   + saving the preprocessed input to", file_path)
        with open(file_path, 'w') as handle:
            handle.write('\n'.join(self.sentences)) 
        return self

    def load_from(self, file_path):
        print("Loading the preprocessed input from", file_path, "\n")
        with open(file_path, 'r') as handle:
            text = handle.read()
        self.sentences = text.split('\n')
        return self
