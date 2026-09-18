"""Hindi surface templates for the comparative-reasoning dataset.

Written for Hindi rather than translated from the Nepali set: the comparison
marker is the free postposition ``से``, the copula is ``है/हैं``, and the
genitive is the separate particle ``का/की/के``.  The Nepali file makes the
opposite choices because its grammar differs, which is what the brief means by
natural phrasing for that language.

Every template is a format string over these fields:

    {a} {b} {c} {d}   entity names
    {x} {y} {z} {w}   numeric values
    {unit}            attribute unit, supplied by ATTRIBUTES

Templates are split into TRAIN and TEST pools by index.  A test item therefore
uses a phrasing the model never saw, which is the only way to tell "learned the
task" from "learned this sentence shape".
"""

# Names verified to encode as clean single pieces with no byte fallback under
# hi_unigram_10000.  A name that fragments would measure tokenizer quality
# rather than reasoning.
# Masculine names only. Hindi adjectives agree in gender, so a mixed pool would
# need "बड़ा" for राम and "बड़ी" for सीता, and every comparative template would
# have to carry the agreement. Restricting the pool keeps the templates
# grammatical without complicating the generator; the trade-off is a smaller
# name pool, which the combinatorics can absorb.
NAMES = [
    "राम", "श्याम", "मोहन", "हरि", "अनिल", "सुनील", "रवि", "विजय",
    "अजय", "संजय", "दीपक", "मनोज", "अशोक", "कमल", "विनोद", "राजू",
    "सुरेश", "रमेश", "महेश", "नरेश", "दिनेश", "गोपाल", "प्रकाश", "शंकर",
    "मुकेश", "राकेश", "योगेश", "हितेश", "जितेंद्र", "उमेश", "लोकेश", "भरत",
]

# The four attributes the brief names: heights, ages, prices, quantities.
# Each carries the comparative adjectives Hindi actually uses for it -- a single
# generic pair such as "bigger/smaller" would read as translated English.
ATTRIBUTES = {
    # "उम्र" is feminine and takes की; "कद" is masculine and takes का. Getting
    # this wrong produces "की कद", which is immediately wrong to a reader.
    "age":      dict(noun="उम्र",  gen="की", unit="वर्ष",  more="बड़ा",   less="छोटा",
                     more_q="सबसे बड़ा", less_q="सबसे छोटा"),
    "height":   dict(noun="कद",    gen="का", unit="सेंटीमीटर", more="लंबा", less="छोटा",
                     more_q="सबसे लंबा", less_q="सबसे छोटा"),
    "price":    dict(noun="मूल्य", gen="का", unit="रुपये", more="महंगा", less="सस्ता",
                     more_q="सबसे महंगा", less_q="सबसे सस्ता"),
    "quantity": dict(noun="संख्या", gen="की", unit="",     more="ज्यादा", less="कम",
                     more_q="सबसे ज्यादा", less_q="सबसे कम"),
}

# Objects for the price and quantity attributes, which describe things rather
# than people.
OBJECTS = ["किताब", "कलम", "बैग", "घड़ी", "कुर्सी", "मेज", "साइकिल", "छाता",
           "जूता", "कमीज", "टोपी", "बोतल"]

ANSWER_PREFIX = "उत्तर:"

# ---------------------------------------------------------------- 2 entities
# Numeric facts about two people, asking which is greater or smaller.
TWO_ENTITY = [
    "{a} {gen} {noun} {x} {unit} है। {b} {gen} {noun} {y} {unit} है। {question}?",
    "{a} {gen} {noun} {x} {unit} और {b} {gen} {noun} {y} {unit} है। {question}?",
    "{a} {gen} {noun} {x} {unit} है, जबकि {b} {gen} {noun} {y} {unit} है। {question}?",
    "दो व्यक्ति हैं। {a} {gen} {noun} {x} {unit} है और {b} {gen} {noun} {y} {unit} है। {question}?",
    "{a} और {b} में, {a} {gen} {noun} {x} {unit} है और {b} {gen} {noun} {y} {unit} है। {question}?",
    "{b} {gen} {noun} {y} {unit} है। {a} {gen} {noun} {x} {unit} है। {question}?",
    "{a} {gen} {noun} {x} {unit} दर्ज की गई और {b} {gen} {noun} {y} {unit}। {question}?",
    "यदि {a} {gen} {noun} {x} {unit} है और {b} {gen} {noun} {y} {unit} है, तो {question}?",
    "{a} {gen} {noun} {x} {unit} बताई गई है और {b} {gen} {noun} {y} {unit}। {question}?",
    "{a} {gen} {noun} {x} {unit} पाई गई, {b} {gen} {noun} {y} {unit} पाई गई। {question}?",
]

# --------------------------------------------------------------- 3 entities
THREE_ENTITY = [
    "{a} {gen} {noun} {x} {unit} है। {b} {gen} {noun} {y} {unit} है। {c} {gen} {noun} {z} {unit} है। {question}?",
    "{a} {gen} {noun} {x} {unit}, {b} {gen} {noun} {y} {unit} और {c} {gen} {noun} {z} {unit} है। {question}?",
    "तीन व्यक्ति हैं। {a} {gen} {noun} {x} {unit}, {b} {gen} {noun} {y} {unit}, {c} {gen} {noun} {z} {unit}। {question}?",
    "{a}, {b} और {c} में {a} {gen} {noun} {x} {unit} है, {b} {gen} {y} {unit} और {c} {gen} {z} {unit}। {question}?",
    "{c} {gen} {noun} {z} {unit} है। {a} {gen} {noun} {x} {unit} है। {b} {gen} {noun} {y} {unit} है। {question}?",
    "{a} {gen} {noun} {x} {unit} है, {b} {gen} {noun} {y} {unit} है, और {c} {gen} {noun} {z} {unit} है। {question}?",
    "यदि {a} {gen} {noun} {x} {unit}, {b} {gen} {noun} {y} {unit} और {c} {gen} {noun} {z} {unit} हो, तो {question}?",
    "एक सूची में {a} {gen} {noun} {x} {unit}, {b} {gen} {noun} {y} {unit} और {c} {gen} {noun} {z} {unit} है। {question}?",
    "{a} {gen} {noun} {x} {unit} दर्ज है, {b} {gen} {y} {unit} और {c} {gen} {z} {unit}। {question}?",
    "{b} {gen} {noun} {y} {unit}, {c} {gen} {noun} {z} {unit}, {a} {gen} {noun} {x} {unit}। {question}?",
]

# ------------------------------------------------- relational, no numbers
# Chains are built from separate facts rather than whole-sentence templates.
#
# The first version wrote each chain as one fixed sentence, and most of those
# sentences named the largest person first. A model could then answer "who is
# the largest?" with "the first name mentioned" about 70% of the time without
# reading the facts. Building the chain from links, phrasing each link at
# random and shuffling their order lets any person appear first, middle or last.
#
# Each link states that {x} is larger than {y}. The four phrasings say the same
# thing; two of them name {y} first. The comparative uses the free postposition
# "से", which is where Hindi and Nepali diverge structurally.
CHAIN_LINKS = [
    "{x} {y} से {more} है",
    "{y} {x} से {less} है",
    "{x} की तुलना में {y} {less} है",
    "{y} की तुलना में {x} {more} है",
]
# The same links without the copula, for list-style frames.
CHAIN_LINKS_SHORT = [
    "{x} {y} से {more}",
    "{y} {x} से {less}",
    "{x} की तुलना में {y} {less}",
    "{y} की तुलना में {x} {more}",
]

# Frames that hold the shuffled facts. "join" says how the facts are combined:
#   sentences  "A. B."          and        "A, B और C"
#   known      "A, B, और C"      semicolon  "A; B"
#   short_*    the same, using CHAIN_LINKS_SHORT
# The seen/unseen template split for the test sets is made over these frames.
AND = "और"
CHAIN_FRAMES = [
    # First four: training (seen). Last three: test-only (unseen).
    {"frame": "{facts}। {question}?", "join": "sentences"},
    {"frame": "{facts}। {question}?", "join": "and"},
    {"frame": "यदि {facts}, तो {question}?", "join": "and"},
    {"frame": "क्रम इस प्रकार है: {facts}। {question}?", "join": "short_and"},
    {"frame": "हम जानते हैं कि {facts}। {question}?", "join": "known"},
    {"frame": "दिया गया है: {facts}। {question}?", "join": "semicolon"},
    {"frame": "जानकारी: {facts}। {question}?", "join": "short_semicolon"},
]

# ----------------------------------------------------- price and quantity
# These describe objects rather than people, so the possessive differs.
OBJECT_TWO = [
    "{a} का मूल्य {x} रुपये है। {b} का मूल्य {y} रुपये है। {question}?",
    "{a} {x} रुपये का है और {b} {y} रुपये का है। {question}?",
    "एक दुकान में {a} {x} रुपये का है, {b} {y} रुपये का। {question}?",
    "{b} का मूल्य {y} रुपये है। {a} का मूल्य {x} रुपये है। {question}?",
    "यदि {a} का मूल्य {x} रुपये और {b} का मूल्य {y} रुपये हो, तो {question}?",
    "{a} की कीमत {x} रुपये है, जबकि {b} की कीमत {y} रुपये है। {question}?",
    "बाजार में {a} {x} रुपये और {b} {y} रुपये में मिलता है। {question}?",
    "{a} का दाम {x} रुपये है और {b} का दाम {y} रुपये है। {question}?",
]

QUANTITY_TWO = [
    "{a} के पास {x} {obj} हैं। {b} के पास {y} {obj} हैं। {question}?",
    "{a} के पास {x} {obj} और {b} के पास {y} {obj} हैं। {question}?",
    "{b} के पास {y} {obj} हैं। {a} के पास {x} {obj} हैं। {question}?",
    "यदि {a} के पास {x} {obj} हों और {b} के पास {y} {obj}, तो {question}?",
    "{a} ने {x} {obj} खरीदे और {b} ने {y} {obj} खरीदे। {question}?",
    "गिनती में {a} के पास {x} {obj} हैं, {b} के पास {y} {obj}। {question}?",
    "{a} के पास {x} {obj} पाए गए, {b} के पास {y} {obj}। {question}?",
    "दो लोगों में {a} के पास {x} {obj} हैं और {b} के पास {y} {obj}। {question}?",
]

# ---------------------------------------------------------------- questions
QUESTIONS = {
    "max2":      ["इनमें कौन {more} है", "दोनों में कौन {more} है", "{more} कौन है"],
    "min2":      ["इनमें कौन {less} है", "दोनों में कौन {less} है", "{less} कौन है"],
    # Hindi कौन covers animate and inanimate alike, so the object variants are
    # the same text. Nepali needs a genuinely different word (कुन), and the
    # generator looks up these keys for both languages.
    "max2_obj":  ["इनमें कौन {more} है", "दोनों में कौन {more} है", "{more} कौन है"],
    "min2_obj":  ["इनमें कौन {less} है", "दोनों में कौन {less} है", "{less} कौन है"],
    "max3":      ["{more_q} कौन है", "तीनों में कौन {more} है", "सबसे {more} कौन है"],
    "min3":      ["{less_q} कौन है", "तीनों में कौन {less} है", "सबसे {less} कौन है"],
    # Four people: "तीनों में" (of the three) would be wrong here.
    "max4":      ["{more_q} कौन है", "चारों में कौन {more} है", "सबसे {more} कौन है"],
    "min4":      ["{less_q} कौन है", "चारों में कौन {less} है", "सबसे {less} कौन है"],
    "second_max4": ["दूसरा सबसे {more} कौन है", "चारों में दूसरा सबसे {more} कौन है"],
    "second_min4": ["दूसरा सबसे {less} कौन है", "चारों में दूसरा सबसे {less} कौन है"],
    "equal":     ["दोनों के {noun} में क्या संबंध है", "इनके {noun} की तुलना क्या है"],
    # Relation questions ask in both directions, so "who is more" cannot be
    # answered by always picking the larger-sounding name.
    "relation":  ["{p} और {q} में कौन {more} है", "{p} और {q} की तुलना में कौन {more} है"],
    "relation_less": ["{p} और {q} में कौन {less} है", "{p} और {q} की तुलना में कौन {less} है"],
    "middle":    ["बीच में कौन है", "मध्य में कौन आता है"],
    "second":    ["दूसरे स्थान पर कौन है", "क्रम में दूसरा कौन है"],
}

EQUAL_ANSWER = "बराबर"

# ------------------------------------------------------------- ablation names
# Used only by scripts/make_ablation_data.py; the main dataset never reads them.
# Both lists are disjoint from NAMES and from each other, and masculine for the
# same agreement reason as NAMES. Names that are also everyday words (विकास,
# प्रेम, आनंद) are left out so a name slot never reads as a noun.

# Out-of-distribution validation: names no training or test example contains.
VAL_OOD_NAMES = [
    "अमित", "सचिन", "नितिन", "पंकज", "आलोक", "तरुण", "वरुण", "कुणाल",
    "अभिषेक", "रोहित",
]

# Large training pool: swapped into the training examples so the model cannot
# lean on a few familiar names.
EXTRA_NAMES = [
    "अरविंद", "अर्जुन", "अक्षय", "अंकित", "अनुराग", "आकाश", "आदित्य", "आशीष",
    "कपिल", "करण", "किशोर", "कुलदीप", "कैलाश", "कृष्ण", "गणेश", "गौरव",
    "गिरीश", "गोविंद", "चेतन", "जगदीश", "जयंत", "तुषार", "दिलीप", "देवेंद्र",
    "धर्मेंद्र", "धीरज", "नरेंद्र", "नवीन", "नारायण", "निखिल", "निर्मल", "नीरज",
    "पवन", "प्रदीप", "प्रमोद", "प्रवीण", "प्रशांत", "बलराम", "बृजेश", "भानु",
    "भूपेंद्र", "मदन", "मनीष", "महेंद्र", "माधव", "मिथिलेश", "मुरली", "मोहित",
    "यशवंत", "युवराज", "रघु", "रजनीश", "रणजीत", "राजेंद्र", "राजेश", "रामेश्वर",
    "राहुल", "रितेश", "रूपेश", "ललित", "लक्ष्मण", "वासुदेव", "विक्रम", "विनय",
    "विपिन", "विवेक", "विशाल", "वीरेंद्र", "शिव", "शैलेंद्र", "शैलेश", "श्रीकांत",
    "संतोष", "सतीश", "संदीप", "समीर", "सुधीर", "सुभाष", "सुमित", "सुरेंद्र",
    "सौरभ", "हरीश", "हेमंत", "अतुल", "अवधेश", "कमलेश", "कन्हैया", "कुंदन",
    "केशव", "गजेंद्र", "घनश्याम", "जतिन", "जनार्दन", "जयेश", "दुर्गेश", "देवराज",
    "पुनीत", "पुष्कर", "बद्री", "बिहारी", "भुवन", "मयंक", "मानस", "मुकुल",
    "योगेंद्र", "रमन", "राघव", "अच्युत", "उपेंद्र", "केदार", "गंगाराम", "जगन्नाथ",
    "दीपेंद्र", "धनंजय", "नवराज", "पशुपति", "प्रताप", "विमल", "मधुसूदन", "रमाकांत",
    "राजकुमार", "शरद", "श्रीराम", "सिद्धार्थ", "सूरज", "हरिप्रसाद", "अनूप", "अरुण",
    "उद्धव", "गजानन", "चिरंजीव", "जयराम", "तिलक", "दामोदर", "नंदलाल", "नरहरि",
    "नागेंद्र", "पुरुषोत्तम", "बालकृष्ण", "भक्तराम", "मुरारी", "शंभू", "शिशिर", "सुशील",
    "शेखर", "रंजन", "राजन", "मिलिंद", "भीम", "भोला", "तेजपाल", "कुलभूषण",
]
