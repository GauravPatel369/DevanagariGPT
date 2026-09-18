"""Nepali surface templates for the comparative-reasoning dataset.

Written for Nepali rather than translated from the Hindi set.  Three grammatical
differences drive the wording and are not interchangeable with Hindi:

* **Comparison is a bound clitic.**  Nepali attaches ``भन्दा`` directly to the
  noun (``हरिभन्दा``, one orthographic word) where Hindi uses the free
  postposition ``से``.  This is the same morphological property that gives
  Nepali its higher tokenizer fertility, 1.59 against 1.34 tokens per word.
* **Copula.**  ``छ`` singular, ``छन्`` plural, against Hindi ``है/हैं``.
* **Numeral classifier.**  Counting requires ``वटा`` (``12 वटा आँप``); the Hindi
  equivalent takes no classifier.

Fields match the Hindi file so the generator can treat both identically:

    {a} {b} {c} {d}   entity names
    {x} {y} {z} {w}   numeric values
    {unit}            attribute unit, supplied by ATTRIBUTES
"""

# Names common in Nepali text, verified to encode as clean pieces with no byte
# fallback under ne_unigram_10000.
# Masculine names only. Nepali adjectives agree in gender exactly as Hindi
# does -- ठूलो against ठूली -- so a mixed pool would need agreement threaded
# through every comparative template. Restricting the pool keeps them
# grammatical; the combinatorics absorb the smaller list easily.
NAMES = [
    "राम", "हरि", "कृष्ण", "गोपाल", "बिष्णु", "शंकर", "प्रकाश", "दीपक",
    "सुरेश", "रमेश", "नरेश", "महेश", "गणेश", "राजु", "बिनोद", "कमल",
    "अनिल", "सुनिल", "मनोज", "अशोक", "रवि", "सन्तोष", "दिपेश", "भरत",
    "मुकेश", "राकेश", "योगेश", "उमेश", "लोकेश", "नरेन्द्र", "सुमन", "बिकास",
]

ATTRIBUTES = {
    "age":      dict(noun="उमेर",  unit="वर्ष",  more="ठूलो",   less="सानो",
                     more_q="सबैभन्दा ठूलो", less_q="सबैभन्दा सानो"),
    "height":   dict(noun="उचाइ",  unit="सेन्टिमिटर", more="अग्लो", less="होचो",
                     more_q="सबैभन्दा अग्लो", less_q="सबैभन्दा होचो"),
    "price":    dict(noun="मूल्य", unit="रुपैयाँ", more="महँगो", less="सस्तो",
                     more_q="सबैभन्दा महँगो", less_q="सबैभन्दा सस्तो"),
    "quantity": dict(noun="संख्या", unit="",     more="बढी",   less="कम",
                     more_q="सबैभन्दा बढी", less_q="सबैभन्दा कम"),
}

OBJECTS = ["किताब", "कलम", "झोला", "घडी", "मेच", "टेबुल", "साइकल", "छाता",
           "जुत्ता", "कमिज", "टोपी", "बोतल"]

ANSWER_PREFIX = "उत्तर:"

# ---------------------------------------------------------------- 2 entities
# Note the genitive suffix -को attaching to the name, against Hindi's separate
# particle की.
TWO_ENTITY = [
    "{a}को {noun} {x} {unit} छ। {b}को {noun} {y} {unit} छ। {question}?",
    "{a}को {noun} {x} {unit} र {b}को {noun} {y} {unit} छ। {question}?",
    "{a}को {noun} {x} {unit} छ भने {b}को {noun} {y} {unit} छ। {question}?",
    "दुई जना छन्। {a}को {noun} {x} {unit} छ र {b}को {noun} {y} {unit} छ। {question}?",
    "{a} र {b} मध्ये {a}को {noun} {x} {unit} छ, {b}को {noun} {y} {unit} छ। {question}?",
    "{b}को {noun} {y} {unit} छ। {a}को {noun} {x} {unit} छ। {question}?",
    "{a}को {noun} {x} {unit} उल्लेख गरिएको छ र {b}को {noun} {y} {unit}। {question}?",
    "यदि {a}को {noun} {x} {unit} छ र {b}को {noun} {y} {unit} छ भने, {question}?",
    "{a}को {noun} {x} {unit} भनिएको छ, {b}को {noun} {y} {unit} भनिएको छ। {question}?",
    "{a}को {noun} {x} {unit} पाइयो, {b}को {noun} {y} {unit} पाइयो। {question}?",
]

# --------------------------------------------------------------- 3 entities
THREE_ENTITY = [
    "{a}को {noun} {x} {unit} छ। {b}को {noun} {y} {unit} छ। {c}को {noun} {z} {unit} छ। {question}?",
    "{a}को {noun} {x} {unit}, {b}को {noun} {y} {unit} र {c}को {noun} {z} {unit} छ। {question}?",
    "तीन जना छन्। {a}को {noun} {x} {unit}, {b}को {noun} {y} {unit}, {c}को {noun} {z} {unit}। {question}?",
    "{a}, {b} र {c} मध्ये {a}को {noun} {x} {unit} छ, {b}को {y} {unit} र {c}को {z} {unit}। {question}?",
    "{c}को {noun} {z} {unit} छ। {a}को {noun} {x} {unit} छ। {b}को {noun} {y} {unit} छ। {question}?",
    "{a}को {noun} {x} {unit} छ, {b}को {noun} {y} {unit} छ, र {c}को {noun} {z} {unit} छ। {question}?",
    "यदि {a}को {noun} {x} {unit}, {b}को {noun} {y} {unit} र {c}को {noun} {z} {unit} भए, {question}?",
    "सूचीमा {a}को {noun} {x} {unit}, {b}को {noun} {y} {unit} र {c}को {noun} {z} {unit} छ। {question}?",
    "{a}को {noun} {x} {unit} उल्लेख छ, {b}को {y} {unit} र {c}को {z} {unit}। {question}?",
    "{b}को {noun} {y} {unit}, {c}को {noun} {z} {unit}, {a}को {noun} {x} {unit}। {question}?",
]

# ------------------------------------------------- relational, no numbers
# Chains are built from separate facts rather than whole-sentence templates.
#
# The first version wrote each chain as one fixed sentence, and most of those
# sentences named the largest person first, so "the first name mentioned" was
# right about 70% of the time for "who is the largest?". Building the chain from
# links, phrasing each at random and shuffling their order removes that cue.
#
# Each link states that {x} is larger than {y}. The bound clitic भन्दा attaches
# to the name it follows ({y}भन्दा, one word), and -को attaches in the
# तुलनामा form.
CHAIN_LINKS = [
    "{x} {y}भन्दा {more} छ",
    "{y} {x}भन्दा {less} छ",
    "{x}को तुलनामा {y} {less} छ",
    "{y}को तुलनामा {x} {more} छ",
]
CHAIN_LINKS_SHORT = [
    "{x} {y}भन्दा {more}",
    "{y} {x}भन्दा {less}",
    "{x}को तुलनामा {y} {less}",
    "{y}को तुलनामा {x} {more}",
]

AND = "र"
CHAIN_FRAMES = [
    # First four: training (seen). Last three: test-only (unseen).
    {"frame": "{facts}। {question}?", "join": "sentences"},
    {"frame": "{facts}। {question}?", "join": "and"},
    {"frame": "यदि {facts} भने, {question}?", "join": "and"},
    {"frame": "क्रम यस्तो छ: {facts}। {question}?", "join": "short_and"},
    {"frame": "हामीलाई थाहा छ कि {facts}। {question}?", "join": "known"},
    {"frame": "दिइएको छ: {facts}। {question}?", "join": "semicolon"},
    {"frame": "जानकारी: {facts}। {question}?", "join": "short_semicolon"},
]

# ----------------------------------------------------- price and quantity
OBJECT_TWO = [
    "{a}को मूल्य {x} रुपैयाँ छ। {b}को मूल्य {y} रुपैयाँ छ। {question}?",
    "{a} {x} रुपैयाँको छ र {b} {y} रुपैयाँको छ। {question}?",
    "एउटा पसलमा {a} {x} रुपैयाँको छ, {b} {y} रुपैयाँको। {question}?",
    "{b}को मूल्य {y} रुपैयाँ छ। {a}को मूल्य {x} रुपैयाँ छ। {question}?",
    "यदि {a}को मूल्य {x} रुपैयाँ र {b}को मूल्य {y} रुपैयाँ भए, {question}?",
    "{a}को दाम {x} रुपैयाँ छ भने {b}को दाम {y} रुपैयाँ छ। {question}?",
    "बजारमा {a} {x} रुपैयाँ र {b} {y} रुपैयाँमा पाइन्छ। {question}?",
    "{a}को भाउ {x} रुपैयाँ छ र {b}को भाउ {y} रुपैयाँ छ। {question}?",
]

# Counting requires the classifier वटा, which Hindi does not use.
QUANTITY_TWO = [
    "{a}सँग {x} वटा {obj} छन्। {b}सँग {y} वटा {obj} छन्। {question}?",
    "{a}सँग {x} वटा {obj} र {b}सँग {y} वटा {obj} छन्। {question}?",
    "{b}सँग {y} वटा {obj} छन्। {a}सँग {x} वटा {obj} छन्। {question}?",
    "यदि {a}सँग {x} वटा {obj} र {b}सँग {y} वटा {obj} भए, {question}?",
    "{a}ले {x} वटा {obj} किने र {b}ले {y} वटा {obj} किने। {question}?",
    "गणनामा {a}सँग {x} वटा {obj} छन्, {b}सँग {y} वटा {obj}। {question}?",
    "{a}सँग {x} वटा {obj} पाइयो, {b}सँग {y} वटा {obj}। {question}?",
    "दुई जनामध्ये {a}सँग {x} वटा {obj} छन् र {b}सँग {y} वटा {obj}। {question}?",
]

# ---------------------------------------------------------------- questions
QUESTIONS = {
    "max2":      ["यीमध्ये को {more} छ", "दुवैमध्ये को {more} छ", "{more} को छ"],
    "min2":      ["यीमध्ये को {less} छ", "दुवैमध्ये को {less} छ", "{less} को छ"],
    # Objects take कुन (which), people take को (who). Using को for a table is
    # ungrammatical, so the object generators use these keys instead.
    "max2_obj":  ["यीमध्ये कुन {more} छ", "दुवैमध्ये कुन {more} छ", "कुन {more} छ"],
    "min2_obj":  ["यीमध्ये कुन {less} छ", "दुवैमध्ये कुन {less} छ", "कुन {less} छ"],
    "max3":      ["{more_q} को हो", "तीनैमध्ये को {more} छ", "सबैभन्दा {more} को छ"],
    "min3":      ["{less_q} को हो", "तीनैमध्ये को {less} छ", "सबैभन्दा {less} को छ"],
    # Four people: "तीनैमध्ये" (of the three) would be wrong here.
    "max4":      ["{more_q} को हो", "चारैमध्ये को {more} छ", "सबैभन्दा {more} को छ"],
    "min4":      ["{less_q} को हो", "चारैमध्ये को {less} छ", "सबैभन्दा {less} को छ"],
    "second_max4": ["दोस्रो सबैभन्दा {more} को छ", "चारैमध्ये दोस्रो सबैभन्दा {more} को छ"],
    "second_min4": ["दोस्रो सबैभन्दा {less} को छ", "चारैमध्ये दोस्रो सबैभन्दा {less} को छ"],
    "equal":     ["दुवैको {noun} बीच के सम्बन्ध छ", "यिनको {noun}को तुलना कस्तो छ"],
    # Asked in both directions, so the answer is not always the larger one.
    "relation":  ["{p} र {q} मध्ये को {more} छ", "{p} र {q}को तुलनामा को {more} छ"],
    "relation_less": ["{p} र {q} मध्ये को {less} छ", "{p} र {q}को तुलनामा को {less} छ"],
    "middle":    ["बीचमा को छ", "माझमा को पर्छ"],
    "second":    ["दोस्रो स्थानमा को छ", "क्रममा दोस्रो को हो"],
}

EQUAL_ANSWER = "बराबर"

# ------------------------------------------------------------- ablation names
# Used only by scripts/make_ablation_data.py; the main dataset never reads them.
# Both lists are disjoint from NAMES and from each other, masculine, spelled as
# Nepali writes them (बि for वि, न्द्र for ंद्र), and exclude names that are also
# common words (जीवन, प्रेम, आनन्द).

# Out-of-distribution validation: names no training or test example contains.
VAL_OOD_NAMES = [
    "रोशन", "सुजन", "सुदर्शन", "नबिन", "सागर", "बिजय", "दिनेश", "पवन",
    "युवराज", "ज्ञानेन्द्र",
]

# Large training pool: swapped into the training examples so the model cannot
# lean on a few familiar names.
EXTRA_NAMES = [
    "अर्जुन", "आकाश", "आशिष", "उत्तम", "कपिल", "केशव", "खगेन्द्र", "गौतम",
    "गिरिराज", "गोविन्द", "चन्द्रबहादुर", "चेतन", "जगदीश", "जनक", "टंक", "डिल्ली",
    "दिलिप", "देवेन्द्र", "धर्मेन्द्र", "धीरज", "नारायण", "निरज", "निर्मल", "पदम",
    "पुष्कर", "प्रदिप", "प्रमोद", "प्रशान्त", "बलराम", "बाबुराम", "बिक्रम", "बिनय",
    "बिबेक", "बिशाल", "बिरेन्द्र", "भीम", "भुपेन्द्र", "भोला", "मदन", "मनिष",
    "महेन्द्र", "माधव", "मिलिन्द", "मोहन", "रघु", "रजनीश", "रञ्जन", "राजन",
    "राजेन्द्र", "राजेश", "रामचन्द्र", "राहुल", "रितेश", "रुपेश", "लक्ष्मण", "ललित",
    "शिव", "शेखर", "श्याम", "सञ्जय", "सतिश", "सन्दिप", "समिर", "सुदिप",
    "सुधिर", "सुभाष", "सुमित", "सुरेन्द्र", "सौरभ", "हरिश", "हेमन्त", "अभिषेक",
    "अमित", "अनुप", "अरुण", "इन्द्रराज", "उद्धव", "कमलेश", "किशोर", "कुलदिप",
    "कैलाश", "गजेन्द्र", "घनश्याम", "चिरञ्जीवी", "जयराम", "जितेन्द्र", "तिलक", "तेजेन्द्र",
    "दामोदर", "देवराज", "नन्दलाल", "नरहरि", "नागेन्द्र", "पुरुषोत्तम", "पूर्णबहादुर", "बद्री",
    "बालकृष्ण", "भक्तबहादुर", "भुवन", "मणिराज", "मुरारी", "यज्ञराज", "रबिन", "रामेश्वर",
    "शम्भु", "शिशिर", "सुशील", "हर्क", "अच्युत", "अजय", "अनन्त", "उपेन्द्र",
    "केदार", "गंगाराम", "चूडामणि", "जगन्नाथ", "दीपेन्द्र", "धनञ्जय", "नवराज", "पशुपति",
    "प्रताप", "बिमल", "मधुसूदन", "रमाकान्त", "राजकुमार", "शरद", "श्रीराम", "सिद्धार्थ",
    "सूर्यप्रसाद", "हरिप्रसाद", "अनिरुद्ध", "हिक्मत", "कुबेर", "गोकुल", "जीतबहादुर", "टेकराज",
    "तारानाथ", "दुर्गाप्रसाद", "धनराज", "नेत्रप्रसाद", "पुण्यप्रसाद", "कुलप्रसाद", "मित्रलाल", "यामलाल",
]
