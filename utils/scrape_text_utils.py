from bs4 import BeautifulSoup
import requests

from collections import ChainMap

def extract_emoticons_and_emojis(): 
    url = 'https://en.wikipedia.org/wiki/List_of_emoticons'
    res = requests.get(url)

    html_page = res.content
    soup = BeautifulSoup(html_page, 'html.parser')
    
    tables = soup.findAll('table', {'class':'wikitable'})
    
    descr_to_idx = get_emotion_descr_to_idx()
    emt_side, emj_side = scrape_emoticons_and_emojis_from_table(tables[0], descr_to_idx['side']) 
    emt_side2, emj_side2 = scrape_emoticons_and_emojis_from_table(tables[1], descr_to_idx['side2']) 
    emt_up, _ = scrape_emoticons_and_emojis_from_table(tables[2], descr_to_idx['up'], 0)
    emt_east, emj_east = scrape_emoticons_and_emojis_from_table(tables[4], descr_to_idx['east'])
    emt_east_oth, emj_east_oth = scrape_emoticons_and_emojis_from_table(tables[5], descr_to_idx['east_oth'])
    emt_2ch, _ = scrape_emoticons_and_emojis_from_table(tables[6], descr_to_idx['2-ch'],0)
    freq_emojis = scrape_freq_emojis(tables[10], tables[11], descr_to_idx['freq_emojis1'], descr_to_idx['freq_emojis2'])
    
    emoticonbook = [emt_side, emt_side2, emt_up, emt_east, emt_east_oth, emt_2ch]
    emojibook = [emj_side, emj_side2, emj_east, emj_east_oth]

    emoticons_mapper = dict()
    for book in emoticonbook:
        emoticons_mapper = dict(ChainMap({}, emoticons_mapper, book))
    emojis_mapper = dict()
    for book in emojibook:
        emojis_mapper = dict(ChainMap({}, emojis_mapper, book))
    emojis_mapper = dict(ChainMap({}, freq_emojis, emojis_mapper)) 
    
    return emoticons_mapper, emojis_mapper
    
def scrape_emoticons_and_emojis_from_table(table, descr_to_idx, emoji_flag=1):
    trs = table.find_all('tr')[1:]
    emoticonbook = dict()
    emojibook = dict()
    for i, tr in enumerate(trs):
        tds = tr.find_all('td')
        descr = tds[-1].text
        emoticons = []
        for td in tds[:-1-emoji_flag]:
            emoticons += [string.replace('\n','') for string in [c.string for c in td.contents] if string]
        emoticons = [emoticon for emoticon in emoticons if emoticon!='' and emoticon!=' ']

        for emoticon in emoticons:
            if descr_to_idx[i] != '' and descr_to_idx[i] != ' ':
                if emoticon not in emoticonbook.keys():
                    emoticonbook[emoticon] = descr_to_idx[i]
        if emoji_flag:
            emojis = tds[-2].text
            emojis = [emoji for emoji in emojis if emoji!='️' and emoji!='—' and emoji!='\n' and emoji!='️ ']
            for emoji in emojis:
                if descr_to_idx[i] != '' and descr_to_idx[i] != ' ':
                    if emoji not in emojibook.keys():
                        emojibook[emoji] = descr_to_idx[i]
    return emoticonbook, emojibook

def scrape_freq_emojis(table1, table2, descr_to_idx1, descr_to_idx2):
    freq_emojis = dict()
    trs = table1.find_all('td')[19:-1]
    trs = [tr.text.replace('\n','') for tr in trs]
    for i, emoji in enumerate(trs):
        if descr_to_idx1[i] != '':
            freq_emojis[emoji] = descr_to_idx1[i]
    
    trs = table2.find_all('td')[32:68]
    trs = [tr.text.replace('\n','') for tr in trs]
    for i, emoji in enumerate(trs):
        if descr_to_idx2[i] != '':
            freq_emojis[emoji] = descr_to_idx2[i]
            
    return freq_emojis

def apply_emo_mapper(sent, mapper):
    new_sent = sent
    for key in mapper.keys():
        new_sent = new_sent.replace(key, ' '+mapper[key]+' ')
    return new_sent

def get_emotion_descr_to_idx():
    descr_to_idx = dict()
    descr_to_idx['side'] = [
        'smile','laugh','smile','frown','crying','laugh','disgust, sadness','surprise','kiss','wink','playful',
        'annoyed','indecision','embarassed','sealed lips','innocent','evil','bored','disappointed',
        'partied all night','confused','sick','dumb','disapproval','nervous','guilty'
    ]
    descr_to_idx['side2'] = [
        'rose', '', 'santa claus', '', '', '', 'sad', 'love'
    ]
    descr_to_idx['up'] = [
        'fish', 'cheer', 'cheerleader', 'lennon', 'disgust, sadness', 'surprise', 'annoyed', 'high five', 
        'crab'
    ]
    descr_to_idx['east'] = [
        'troubled', 'baby', 'nervous', 'smoking', 'sleeping', 'wink', 'confused', 'ultraman', '', 'joyful', 
        'respect', 'questioning', 'sad', 'shame', 'tired', 'cat', 'looking down', 'giggling', 'confusion',
        'facepalm', 'laugh', 'waving', 'alien', 'excited', 'amazed', '', 'laughing', '', '', '', 'music',
        'disappointed', 'eyeglasses', 'jotting note', 'happy', 'grinning', 'surprised', 'infatuation', 'surprised',
        'dissatisfied', 'mellow', 'snubbed', 'kiss', 'studying', 'joy', 'cute'
    ]
    descr_to_idx['east_oth'] = [
        'bubbles', 'tea', 'star', 'fish', 'octopus', 'snake', 'bat', 'tadpole', 'bomb', 'despair', 'table flip', 
        ''
    ]
    descr_to_idx['2-ch'] = [
        'respect', 'snubbed', 'perky', 'salute', 'terribly sad', 'peace', 'irritable', 'angry', 'yelling', 
        'surprised','don\'t know answer', 'carefree', 'indifferent', 'shocked', 'happy', 'carefree', 'spook', 
        'surprised', 'jog cheek', 'amazed', 'smoking', 'cheers', 'intuition', 'friendly', 'lonely', 'depressed', 
        'thinking', 'impatience', 'whispers', 'money', 'sliding on belly', 'unforeseen', 'don\'t need it', 
        'come on', 'mocking', 'bad', 'goofing', 'sad', 'not convincing', 'simper', 'deflagged', 'happy', 'happy', 
        'despair', '', '', '', 'distaste', 'shouting', 'asleep', 'kick', 'discombobulated', 'running', 'happy', 
        'happy', 'shocked', 'angry', 'do it', '', 'angel'
    ]
    descr_to_idx['freq_emojis1'] = [
        'smile', 'smile', 'laugh', 'smile', 'smile', 'sweat', 'laugh', 'angel', 'evil smile', 'wink', 'smile', 
        'smile', 'smile', 'smile', 'smile', 'smirk', '', 'neutral', 'disappointed', 'disappointed', 
        'disappointed', 'worried', 'worried', 'sad', 'kiss', 'kiss', 'kiss', 'kiss', 'smile', 'smile, wink', 
        'smile', 'sad', 'sad', '', 'angry', 'angry', 'worried', 'worried', 'angry', 'worried', 'worried', 
        'worried', 'worried', 'worried', 'disappointed', 'worried', 'nervous', 'crying', 'surprised', 
        'surprised', '', 'worried', 'worried', 'surprised', 'surprised, sad', 'sleeping', 'sleeping', 
        'speechless', 'sick', 'laugh', 'laugh','smile', 'smile', 'smirk', 'kiss', 'angry', 'worried', '', 
        'worried', 'sad', 'smile', 'neutral', 'thinking','bad', 'ok', 'bowing', 'bad', 'surprised', 'bad', 
        'neutral', 'good', 'frown', 'frown', 'please'
    ]
    descr_to_idx['freq_emojis2'] = [
        'love', 'love', 'pinch', 'quiet', 'smile, money', 'sick', 'smile', 'thinking', 'sick', 'robot', 
        'smile', 'agree', 'agree', 'bye', 'agree', 'agree', 'agree', 'hope', 'agree', '', 'smile', 'smile',
        'sick', 'laugh', 'drooling', 'lying', 'disappointed', 'sneezing', 'thinking', 'smile', 'laugh, wink',
        'quiet', 'angry', 'laugh', 'sick', 'surprised'
    ]
    return descr_to_idx

def get_abbreviations():
    # https://www.kaggle.com/nmaguette/up-to-date-list-of-slangs-for-text-preprocessing
    abbreviations = {
        "\$" : " dollar ",
        "€" : " euro ",
        "4ao" : "for adults only",
        "a.m" : "before midday",
        "a3" : "anytime anywhere anyplace",
        "aamof" : "as a matter of fact",
        "acct" : "account",
        "adih" : "another day in hell",
        "afaic" : "as far as i am concerned",
        "afaict" : "as far as i can tell",
        "afaik" : "as far as i know",
        "afair" : "as far as i remember",
        "afk" : "away from keyboard",
        "app" : "application",
        "approx" : "approximately",
        "apps" : "applications",
        "asap" : "as soon as possible",
        "asl" : "age, sex, location",
        "atk" : "at the keyboard",
        "ave." : "avenue",
        "aymm" : "are you my mother",
        "ayor" : "at your own risk", 
        "b&b" : "bed and breakfast",
        "b+b" : "bed and breakfast",
        "b.c" : "before christ",
        "b2b" : "business to business",
        "b2c" : "business to customer",
        "b4" : "before",
        "b4n" : "bye for now",
        "b@u" : "back at you",
        "bae" : "before anyone else",
        "bak" : "back at keyboard",
        "bbbg" : "bye bye be good",
        "bbc" : "british broadcasting corporation",
        "bbias" : "be back in a second",
        "bbl" : "be back later",
        "bbs" : "be back soon",
        "be4" : "before",
        "bfn" : "bye for now",
        "blvd" : "boulevard",
        "bout" : "about",
        "brb" : "be right back",
        "bros" : "brothers",
        "brt" : "be right there",
        "bsaaw" : "big smile and a wink",
        "btw" : "by the way",
        "bwl" : "bursting with laughter",
        "c/o" : "care of",
        "cet" : "central european time",
        "cf" : "compare",
        "cia" : "central intelligence agency",
        "csl" : "can not stop laughing",
        "cu" : "see you",
        "cul8r" : "see you later",
        "cv" : "curriculum vitae",
        "cwot" : "complete waste of time",
        "cya" : "see you",
        "cyt" : "see you tomorrow",
        "dae" : "does anyone else",
        "dbmib" : "do not bother me i am busy",
        "diy" : "do it yourself",
        "dm" : "direct message",
        "dwh" : "during work hours",
        "e123" : "easy as one two three",
        "eet" : "eastern european time",
        "eg" : "example",
        "embm" : "early morning business meeting",
        "encl" : "enclosed",
        "encl." : "enclosed",
        "etc" : "and so on",
        "faq" : "frequently asked questions",
        "fawc" : "for anyone who cares",
        "fb" : "facebook",
        "fc" : "fingers crossed",
        "fig" : "figure",
        "fimh" : "forever in my heart", 
        "ft." : "feet",
        "ft" : "featuring",
        "ftl" : "for the loss",
        "ftw" : "for the win",
        "fwiw" : "for what it is worth",
        "fyi" : "for your information",
        "g9" : "genius",
        "gahoy" : "get a hold of yourself",
        "gal" : "get a life",
        "gcse" : "general certificate of secondary education",
        "gfn" : "gone for now",
        "gg" : "good game",
        "gl" : "good luck",
        "glhf" : "good luck have fun",
        "gmt" : "greenwich mean time",
        "gmta" : "great minds think alike",
        "gn" : "good night",
        "g.o.a.t" : "greatest of all time",
        "goat" : "greatest of all time",
        "goi" : "get over it",
        "gps" : "global positioning system",
        "gr8" : "great",
        "gratz" : "congratulations",
        "gyal" : "girl",
        "h&c" : "hot and cold",
        "hp" : "horsepower",
        "hr" : "hour",
        "hrh" : "his royal highness",
        "ht" : "height",
        "ibrb" : "i will be right back",
        "ic" : "i see",
        "icq" : "i seek you",
        "icymi" : "in case you missed it",
        "idc" : "i do not care",
        "idgadf" : "i do not give a damn fuck",
        "idgaf" : "i do not give a fuck",
        "idk" : "i do not know",
        "ie" : "that is",
        "i.e" : "that is",
        "ifyp" : "i feel your pain",
        "IG" : "instagram",
        "iirc" : "if i remember correctly",
        "ilu" : "i love you",
        "ily" : "i love you",
        "imho" : "in my humble opinion",
        "imo" : "in my opinion",
        "imu" : "i miss you",
        "iow" : "in other words",
        "irl" : "in real life",
        "j4f" : "just for fun",
        "jic" : "just in case",
        "jk" : "just kidding",
        "jsyk" : "just so you know",
        "l8r" : "later",
        "lb" : "pound",
        "lbs" : "pounds",
        "ldr" : "long distance relationship",
        "lmao" : "laugh my ass off",
        "lmfao" : "laugh my fucking ass off",
        "lol" : "laughing out loud",
        "ltd" : "limited",
        "ltns" : "long time no see",
        "m8" : "mate",
        "mf" : "motherfucker",
        "mfs" : "motherfuckers",
        "mfw" : "my face when",
        "mofo" : "motherfucker",
        "mph" : "miles per hour",
        "mr" : "mister",
        "mrw" : "my reaction when",
        "ms" : "miss",
        "mte" : "my thoughts exactly",
        "nagi" : "not a good idea",
        "nbc" : "national broadcasting company",
        "nbd" : "not big deal",
        "nfs" : "not for sale",
        "ngl" : "not going to lie",
        "nhs" : "national health service",
        "nrn" : "no reply necessary",
        "nsfl" : "not safe for life",
        "nsfw" : "not safe for work",
        "nth" : "nice to have",
        "nvr" : "never",
        "nyc" : "new york city",
        "oc" : "original content",
        "og" : "original",
        "ohp" : "overhead projector",
        "oic" : "oh i see",
        "omdb" : "over my dead body",
        "omg" : "oh my god",
        "omw" : "on my way",
        "p.a" : "per annum",
        "p.m" : "after midday",
        "pm" : "prime minister",
        "poc" : "people of color",
        "pov" : "point of view",
        "pp" : "pages",
        "ppl" : "people",
        "prw" : "parents are watching",
        "ps" : "postscript",
        "pt" : "point",
        "ptb" : "please text back",
        "pto" : "please turn over",
        "qpsa" : "what happens", #"que pasa",
        "ratchet" : "rude",
        "rbtl" : "read between the lines",
        "rlrt" : "real life retweet", 
        "rofl" : "rolling on the floor laughing",
        "roflol" : "rolling on the floor laughing out loud",
        "rotflmao" : "rolling on the floor laughing my ass off",
        "rt" : "retweet",
        "ruok" : "are you ok",
        "sfw" : "safe for work",
        "sk8" : "skate",
        "smh" : "shake my head",
        "sq" : "square",
        "srsly" : "seriously", 
        "ssdd" : "same stuff different day",
        "tbh" : "to be honest",
        "tbs" : "tablespooful",
        "tbsp" : "tablespooful",
        "tfw" : "that feeling when",
        "thks" : "thank you",
        "tho" : "though",
        "thx" : "thank you",
        "tia" : "thanks in advance",
        "til" : "today i learned",
        "tl;dr" : "too long i did not read",
        "tldr" : "too long i did not read",
        "tmb" : "tweet me back",
        "tntl" : "trying not to laugh",
        "ttyl" : "talk to you later",
        "u" : "you",
        "u2" : "you too",
        "u4e" : "yours for ever",
        "utc" : "coordinated universal time",
        "w/" : "with",
        "w/o" : "without",
        "w8" : "wait",
        "wassup" : "what is up",
        "wb" : "welcome back",
        "wtf" : "what the fuck",
        "wtg" : "way to go",
        "wtpa" : "where the party at",
        "wuf" : "where are you from",
        "wuzup" : "what is up",
        "wywh" : "wish you were here",
        "yd" : "yard",
        "ygtr" : "you got that right",
        "ynk" : "you never know",
        "zzz" : "sleeping bored and tired"
    }
    return abbreviations