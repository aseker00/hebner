from pathlib import Path
from src.processing import processing_conllu as conllu
import re
import sys


def is_number(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def extract_token(token_node: list) -> str:
    tokens = [node['misc']['token_str'] for node in token_node]
    assert tokens.count(tokens[0]) == len(tokens)
    return tokens[0]


rev_num_pattern = re.compile('^[לכהמבו]?[0-9][0-9-,.]*$')


def fix(sentence: dict) -> (dict, dict, dict):
    fixed_forms = {}
    fixed_tokens = {}
    token_nodes = sentence['token_nodes']
    sent_id = sentence['id']
    sent_text = sentence['text']
    fixed_text = []
    sent_text_fixed = False
    for token_node in token_nodes:
        segments = [node['form'] for node in token_node]
        seg_token = ''.join(segments)
        token = extract_token(token_node)
        if rev_num_pattern.match(token):
            if token != seg_token:
                if rev_num_pattern.match(seg_token):
                    print('s_{} <-> t_{}: x_{}'.format(seg_token, token, sent_text))
                    in_line = sys.stdin.readline()
                    if in_line.strip() == '<':
                        print('s_{} -> t_{}'.format(seg_token, token))
                        token = seg_token
                        for node in token_node:
                            node_id = node['id']
                            fixed_tokens[(sent_id, node_id)] = seg_token
                        sent_text_fixed = True
                    else:
                        print('t_{} -> s_{}'.format(token, seg_token))
                        token_start_offset = 0
                        for node in token_node:
                            seg = node['form']
                            node_id = node['id']
                            token_end_offset = token_start_offset + len(seg)
                            token_seg = token[token_start_offset:token_end_offset]
                            if seg != token_seg:
                                fixed_forms[(sent_id, node_id)] = token_seg
                            token_start_offset = token_end_offset
        fixed_text.append(token)
        space_after = bool(token_node[-1]['misc']['SpaceAfter'])
        if space_after:
            fixed_text.append(' ')
    if sent_text_fixed:
        fixed_text = ''.join(fixed_text)
        return {sent_id: fixed_text}, fixed_forms, fixed_tokens
    return {}, fixed_forms, fixed_tokens


with open('data/clean/treebank/spmrl-03.conllu') as f:
    lines = f.readlines()
sentences = {}
sent_lines = []
for line in [l.strip() for l in lines]:
    sent_lines.append(line)
    if not line:
        sentences[sent_id] = sent_lines
        sent_lines = []
    elif '# sent_id = ' in line:
        sent_id = int(line[len('# sent_id = '):])

lattice_sentences = conllu.read_conllu_sentences(Path('data/clean/treebank/spmrl-03.conllu'), 'spmrl')
fixed_sentences = [fix(sent) for sent in lattice_sentences]
fixed_texts = {k: v for sent in fixed_sentences for k, v in sent[0].items()}
fixed_forms = {k: v for sent in fixed_sentences for k, v in sent[1].items()}
fixed_tokens = {k: v for sent in fixed_sentences for k, v in sent[2].items()}
for sent_id, text in fixed_texts.items():
    print('# {} = {}'.format(sent_id, text))
for (sent_id, node_id), token in fixed_tokens.items():
    print('# {}-{} = {}'.format(sent_id, node_id, token))
for (sent_id, node_id), form in fixed_forms.items():
    print('# {}-{} = {}'.format(sent_id, node_id, form))
with open('data/clean/treebank/spmrl-04.conllu', 'w') as f:
    for sent_id in sentences:
        sent_lines = sentences[sent_id]
        if sent_id in fixed_texts:
            sent_text = fixed_texts[sent_id]
            sent_lines[2] = '# text_from_ud = {}'.format(sent_text)
        fixed_sent_lines = []
        for line in sent_lines:
            if not line:
                fixed_sent_lines.append(line)
            else:
                line_parts = line.split()
                line_node_id = line_parts[0]
                if is_number(line_node_id):
                    node_id = int(line_node_id)
                    if (sent_id, node_id) in fixed_forms:
                        line_parts[1] = fixed_forms[(sent_id, node_id)]
                        fixed_line = '\t'.join(line_parts)
                        fixed_sent_lines.append(fixed_line)
                    elif (sent_id, node_id) in fixed_tokens:
                        fixed_token = fixed_tokens[(sent_id, node_id)]
                        fixed_misc = re.sub(r'token_str=.*', 'token_str={}'.format(fixed_token), line_parts[9])
                        line_parts[9] = fixed_misc
                        fixed_line = '\t'.join(line_parts)
                        fixed_sent_lines.append(fixed_line)
                    else:
                        fixed_sent_lines.append(line)
                else:
                    fixed_sent_lines.append(line)
        sent_lines = '\n'.join(fixed_sent_lines)
        f.write(sent_lines)
        f.write('\n')


# s_כ100,000 <-> t_כ001,000: x_כ001,000 עובדים שבתו משום שהאוצר סירב להעניק תוספת שכר לאחוז קטן מהם, המועסקים בחברות מפסידות.
# <
# s_כ100,000 -> t_כ001,000
# s_ב2.81 <-> t_ב18.2: x_הגירעון המסחרי גדל ב44% בעשרת החודשים הראשונים של השנה, לעומת התקופה המקבילה אשתקד, והסתכם ב18.2 מיליארדי דולר.
#
# t_ב18.2 -> s_ב2.81
# s_ל9.1 <-> t_ל1.9: x_היצוא נטו עלה ב8% והגיע ל1.9 מיליארדי ד.
#
# t_ל1.9 -> s_ל9.1
# s_כ1.2 <-> t_כ2.1: x_"אין כל מחויבות לתת לעובד מעזה שכר שנתי של 10,000 ש"ח בשנה, כשההוצאה הכוללת לתשלומי שכר לפועלי השטחים היא כ2.1 מיליארד דולר בשנה", הוסיף מודעי.
#
# t_כ2.1 -> s_כ1.2
# s_ל01 <-> t_ל10: x_פורת נידון ל10 חודשי מאסר על-תנאי ולקנס של 3,000 ש"ח.
#
# t_ל10 -> s_ל01
# s_ב1.6.90 <-> t_ב1.6.09: x_הפיקוח על המחירים במשק הגז הוחזר ב1.6.09, ואולם מאז סירב מינהל הדלק להתערב בקביעת גובה העמלות.
#
# t_ב1.6.09 -> s_ב1.6.90
# s_ה25.11 <-> t_ה52.11: x_התערוכה מוצגת בבית אריאלה עד ה52.11.
# <
# s_ה25.11 -> t_ה52.11
# s_כ150,000 <-> t_כ051,000: x_ביום רגיל, לפי נתוני החברה, מבקרים בדיסני-וורלד 25,000 בני-אדם, אך העומס העיקרי הוא בפארקים של אפקוט ושל מ-ג-מ כ051,000 ביום.
# <
# s_כ150,000 -> t_כ051,000
# s_ל34,028 <-> t_ל43,820: x_בסניף בנק הפועלים ברחוב פנקס בתל-אביב שדד לייבוביץ, בנוסף ל43,820 שקל מכספי הבנק, גם סך של 2,734 שקל מאחד מלקוחות הסניף, שעמד ליד הדלפק של הכספרית שנשדדה.
#
# t_ל43,820 -> s_ל34,028
# s_ה91 <-> t_ה19: x_סדאם חוסיין ב"מחוז ה19 של עיראק", היא כוויית לשעבר, והמדינות שהתאגדו לעקור אותו משם מראות סימני חולשה הולכים וגוברים.
#
# t_ה19 -> s_ה91
# s_ב21.9 <-> t_ב12.9: x_כן מוצעות נעלי "פינוק" לבית ב12.9 ש"ח, לדגם פתוח מאחור.
#
# t_ב12.9 -> s_ב21.9
# s_ב19.9 <-> t_ב19.90: x_המחיר, החל ב19.90 ש"ח ועד ל23.40 ש"ח.
#
# t_ב19.90 -> s_ב19.9
# s_ל32.40 <-> t_ל23.40: x_המחיר, החל ב19.90 ש"ח ועד ל23.40 ש"ח.
#
# t_ל23.40 -> s_ל32.40
# s_מ201 <-> t_מ102: x_מ102 היצירות שהוצעו למכירה נמכרו 83, ב 63.5 מיליארד ין (42.02 מיליון דולר).
#
# t_מ102 -> s_מ201
# s_ל41 <-> t_ל14: x_חברת החשמל החליטה על ניתוק הזרם בעקבות חוב של מקורות לחודשים ספטמבר ואוקטובר, המגיע ל14 מיליון ש"ח.
#
# t_ל14 -> s_ל41
# s_ו14 <-> t_ו41: x_תיירת כבת 65 נהרגה ו41 תיירים נפצעו, מהם לפחות אחד במצב קשה.
#
# t_ו41 -> s_ו14
# s_ל345,000 <-> t_ל543,000: x_כוח ההתקפה יושלם עד ראשית השנה הבאה ויהיה זה הכוח הצבאי האמריקאי הגדול ביותר בשטח זר מאז וייטנאם; צבא ארה"ב בווייטנאם הגיע ל543,000 ב1969.
#
# t_ל543,000 -> s_ל345,000
# s_כ2,004 <-> t_כ2,400: x_מערכת הביטחון העבירה למשטרה רשימה של כ2,400 שמות של פלשתינאים בגדה, שלא יורשו להיכנס לישראל כבר בימים הקרובים.
#
# t_כ2,400 -> s_כ2,004
# s_ה36 <-> t_ה63: x_ביום ששי התקיימה שביתה כללית לציון החודש ה63 לתחילת ההתקוממות.
#
# t_ה63 -> s_ה36
# s_ל400 <-> t_ל004: x_כמו כן קיימת אי-בהירות לגבי סכום ההוצאות הדרושות, מכיוון שאין יודעים בדיוק כמה עולים יגיעו בשנה הבאה, וההערכות נעות בין 150 אלף ל004 אלף.
# <
# s_ל400 -> t_ל004
# s_ב1.2.19 <-> t_ב1.2.91: x_לאחר משא ומתן עם ההנהלה הוא קיבל הודעת פיטורין האמורים להיכנס לתוקף ב1.2.91.
#
# t_ב1.2.91 -> s_ב1.2.19
# s_כ1,008 <-> t_כ1,800: x_בכך היו לשלושה מבין כ1,800 העולים העושים את צעדי הקליטה הראשונים בישראל במסגרת התנועה הקיבוצית.
#
# t_כ1,800 -> s_כ1,008
# s_ל0052 <-> t_ל2500: x_המינהל מציע גם קרקע ל2500 דירות להשכרה, רובן במרכז הארץ.
#
# t_ל2500 -> s_ל0052
# s_233 <-> t_2.3.3: x_הוא הציע 3 מגרשים על שטח שבין 2.3.3 דונם למגרש, ויקבל בין 32 ל120 אלף ש"ח למגרש.
#
# t_2.3.3 -> s_233
# s_ל218 <-> t_ל812: x_להלן כמה דוגמאות לגבי הירידה במחירים: דירה בת 5 חדרים ברח מעיין בגבעתיים, תמורתה ביקשו לפני כ4 חודשים 250 אלף דולר, נמכרה בסופו של דבר רק אחרי שמחירה ירד ל812 אלף דולר ירידה של 13%.
# <
# s_ל218 -> t_ל812
# s_כ0054 <-> t_כ4500: x_בעלי הרשת התחייבו לשלם שכר דירה של כ4500 דולר לחודש, לא כולל מע"ם, על החנות ששטחה 180 מ"ר כלומר 25 דולר למ"ר.
#
# t_כ4500 -> s_כ0054
# s_ל13 <-> t_ל31: x_ליגטי, מגדולי המחצית השנייה של המאה, הנחיל לנו בין שאר יצירותיו המקסימות קונצרטו קאמרי ל31 כלי קשת, נשיפה ומקלדת.
#
# t_ל31 -> s_ל13
# s_ה13 <-> t_ה31: x_ממלנד, שאותה הקימו במאה ה31 מתיישבים גרמנים מחבל הריין, תחת השם נוי דורטמונד, היתה חלק ממזרח-פרוסיה עד 1945 (מלבד בשנים 23 39).
# <
# s_ה13 -> t_ה31
# s_ב79 <-> t_ב97: x_בשנה שעברה היו בבית-המשפט העליון כ3,009 תיקים תלויים ועומדים (לעומת כ3,001 ב97), בבתי-המשפט המחוזיים כ06 אלף (כ24 אלף ב97) ובבתי-משפט השלום כ013 אלף (כ39 אלף ב97).
#
# t_ב97 -> s_ב79
# s_כ60 <-> t_כ06: x_בשנה שעברה היו בבית-המשפט העליון כ3,009 תיקים תלויים ועומדים (לעומת כ3,001 ב97), בבתי-המשפט המחוזיים כ06 אלף (כ24 אלף ב97) ובבתי-משפט השלום כ013 אלף (כ39 אלף ב97).
# <
# s_כ60 -> t_כ06
# s_כ42 <-> t_כ24: x_בשנה שעברה היו בבית-המשפט העליון כ3,009 תיקים תלויים ועומדים (לעומת כ3,001 ב97), בבתי-המשפט המחוזיים כ06 אלף (כ24 אלף ב97) ובבתי-משפט השלום כ013 אלף (כ39 אלף ב97).
# <
# s_כ42 -> t_כ24
# s_ב79 <-> t_ב97: x_בשנה שעברה היו בבית-המשפט העליון כ3,009 תיקים תלויים ועומדים (לעומת כ3,001 ב97), בבתי-המשפט המחוזיים כ06 אלף (כ24 אלף ב97) ובבתי-משפט השלום כ013 אלף (כ39 אלף ב97).
#
# t_ב97 -> s_ב79
# s_כ310 <-> t_כ013: x_בשנה שעברה היו בבית-המשפט העליון כ3,009 תיקים תלויים ועומדים (לעומת כ3,001 ב97), בבתי-המשפט המחוזיים כ06 אלף (כ24 אלף ב97) ובבתי-משפט השלום כ013 אלף (כ39 אלף ב97).
# <
# s_כ310 -> t_כ013
# s_כ93 <-> t_כ39: x_בשנה שעברה היו בבית-המשפט העליון כ3,009 תיקים תלויים ועומדים (לעומת כ3,001 ב97), בבתי-המשפט המחוזיים כ06 אלף (כ24 אלף ב97) ובבתי-משפט השלום כ013 אלף (כ39 אלף ב97).
# <
# s_כ93 -> t_כ39
# s_ב79 <-> t_ב97: x_בשנה שעברה היו בבית-המשפט העליון כ3,009 תיקים תלויים ועומדים (לעומת כ3,001 ב97), בבתי-המשפט המחוזיים כ06 אלף (כ24 אלף ב97) ובבתי-משפט השלום כ013 אלף (כ39 אלף ב97).
#
# t_ב97 -> s_ב79
# s_ל300 <-> t_ל003: x_שר המשפטים דן מרידור בחר בדרך אחרת לפתור את הבעיה: הרחבת סמכויות בתי-משפט השלום באמצעות העלאת הסכומים שהם מוסמכים לדון בהם ל003 אלף ש"ח, והגדלת סמכויות התביעה לפתוח בהליכים פליליים בבתי-משפט השלום בעבירות מסוג פשע.
# <
# s_ל300 -> t_ל003
# s_ה17 <-> t_ה71: x_ימים אלה מתקיימת במילאנו תערוכה יוצאת דופן של ביצים מחרסינה, שיוצרו בחצר הצאר מהמאה ה71 ועד תחילת המאה ה02.
# <
# s_ה17 -> t_ה71
# s_ה20 <-> t_ה02: x_ימים אלה מתקיימת במילאנו תערוכה יוצאת דופן של ביצים מחרסינה, שיוצרו בחצר הצאר מהמאה ה71 ועד תחילת המאה ה02.
# <
# s_ה20 -> t_ה02
# s_ה70 <-> t_ה07: x_הוא נבנה בשנות ה07 בצדו המערבי של רחוב בן-סרוק.
# <
# s_ה70 -> t_ה07
# s_ה50 <-> t_ה05: x_בין מחולליה היה לה-קורוואזיה, שבנה את הכנסייה ברונשאן בשנות ה05.
# <
# s_ה50 -> t_ה05
# s_ל70 <-> t_ל07: x_בהתבסס על המנות הללו, שאחדות איכזבו בהחלט, יגיע החשבון לשניים ל07 ש"ח כולל בירה ואספרסו.
# <
# s_ל70 -> t_ל07
# s_ל110 <-> t_ל011: x_ארוחה המבוססת על מנות אלה תגיע ל011 ש"ח לזוג.
# <
# s_ל110 -> t_ל011
# s_מ80 <-> t_מ08: x_המחירים כאן נמוכים (אפשר לקבל ארוחת ערב מלאה לזוג תמורת פחות מ08 ש"ח), אבל האוכל מאכזב.
# <
# s_מ80 -> t_מ08
# s_כ150 <-> t_כ051: x_יחד עם בקבוק של קברנה סוביניון, ארוחה לשניים המבוססת על המנות הללו תעלה כ051 ש"ח כולל שירות.
# <
# s_כ150 -> t_כ051
# s_כ20 <-> t_כ02: x_בסך הכל היה שם מזון מספיק לארבעה אנשים והחשבון, כולל קנקן גדול של יין הבית אדום, היה כ02 ש"ח לאיש.
# <
# s_כ20 -> t_כ02
# s_מ12 <-> t_מ21: x_גם מחירן של המנות העיקריות בתפריט זה, המוגבל בכוונה (נקניקים עם תפוחי אדמה, קיש לוריין או אומצת פילה בגרגרי פלפלים ירוקים), סביר ונע מ21 עד 23 ש"ח.
# <
# s_מ12 -> t_מ21
# s_כ136 <-> t_כ631: x_כולל קנקן של חצי ליטר של יין הבית (אשקלון אדום) וקפה לסיום ארוחת ערב המבוססת על המנות הללו המחיר יהיה כ631 ש"ח, כולל דמי שירות בסך 5 ש"ח, סעיף מעליב בהתחשב במחיר.
# <
# s_כ136 -> t_כ631
# '# 1959 = כ100,000 עובדים שבתו משום שהאוצר סירב להעניק תוספת שכר לאחוז קטן מהם, המועסקים בחברות מפסידות.
# 2939 = התערוכה מוצגת בבית אריאלה עד ה25.11.
# 3461 = ביום רגיל, לפי נתוני החברה, מבקרים בדיסני-וורלד 25,000 בני-אדם, אך העומס העיקרי הוא בפארקים של אפקוט ושל מ-ג-מ כ150,000 ביום.
# 5359 = כמו כן קיימת אי-בהירות לגבי סכום ההוצאות הדרושות, מכיוון שאין יודעים בדיוק כמה עולים יגיעו בשנה הבאה, וההערכות נעות בין 150 אלף ל400 אלף.
# 5542 = להלן כמה דוגמאות לגבי הירידה במחירים: דירה בת 5 חדרים ברח מעיין בגבעתיים, תמורתה ביקשו לפני כ4 חודשים 250 אלף דולר, נמכרה בסופו של דבר רק אחרי שמחירה ירד ל218 אלף דולר ירידה של 13%.
# 5903 = ממלנד, שאותה הקימו במאה ה13 מתיישבים גרמנים מחבל הריין, תחת השם נוי דורטמונד, היתה חלק ממזרח-פרוסיה עד 1945 (מלבד בשנים 23 39).
# 5913 = בשנה שעברה היו בבית-המשפט העליון כ3,009 תיקים תלויים ועומדים (לעומת כ3,001 ב97), בבתי-המשפט המחוזיים כ60 אלף (כ42 אלף ב97) ובבתי-משפט השלום כ310 אלף (כ93 אלף ב97).
# 5919 = שר המשפטים דן מרידור בחר בדרך אחרת לפתור את הבעיה: הרחבת סמכויות בתי-משפט השלום באמצעות העלאת הסכומים שהם מוסמכים לדון בהם ל300 אלף ש"ח, והגדלת סמכויות התביעה לפתוח בהליכים פליליים בבתי-משפט השלום בעבירות מסוג פשע.
# 5950 = ימים אלה מתקיימת במילאנו תערוכה יוצאת דופן של ביצים מחרסינה, שיוצרו בחצר הצאר מהמאה ה17 ועד תחילת המאה ה20.
# 5992 = הוא נבנה בשנות ה70 בצדו המערבי של רחוב בן-סרוק.
# 6008 = בין מחולליה היה לה-קורוואזיה, שבנה את הכנסייה ברונשאן בשנות ה50.
# 6042 = בהתבסס על המנות הללו, שאחדות איכזבו בהחלט, יגיע החשבון לשניים ל70 ש"ח כולל בירה ואספרסו.
# 6056 = ארוחה המבוססת על מנות אלה תגיע ל110 ש"ח לזוג.
# 6072 = המחירים כאן נמוכים (אפשר לקבל ארוחת ערב מלאה לזוג תמורת פחות מ80 ש"ח), אבל האוכל מאכזב.
# 6086 = יחד עם בקבוק של קברנה סוביניון, ארוחה לשניים המבוססת על המנות הללו תעלה כ150 ש"ח כולל שירות.
# 6096 = בסך הכל היה שם מזון מספיק לארבעה אנשים והחשבון, כולל קנקן גדול של יין הבית אדום, היה כ20 ש"ח לאיש.
# 6097 = גם מחירן של המנות העיקריות בתפריט זה, המוגבל בכוונה (נקניקים עם תפוחי אדמה, קיש לוריין או אומצת פילה בגרגרי פלפלים ירוקים), סביר ונע מ12 עד 23 ש"ח.
# 6144 = כולל קנקן של חצי ליטר של יין הבית (אשקלון אדום) וקפה לסיום ארוחת ערב המבוססת על המנות הללו המחיר יהיה כ136 ש"ח, כולל דמי שירות בסך 5 ש"ח, סעיף מעליב בהתחשב במחיר.
# 1959-1 = כ100,000
# 1959-2 = כ100,000
# 2939-8 = ה25.11
# 2939-9 = ה25.11
# 3461-38 = כ150,000
# 3461-39 = כ150,000
# 5359-35 = ל400
# 5359-36 = ל400
# 5542-41 = ל218
# 5542-42 = ל218
# 5903-10 = ה13
# 5903-11 = ה13
# 5913-34 = כ60
# 5913-35 = כ60
# 5913-38 = כ42
# 5913-39 = כ42
# 5913-51 = כ310
# 5913-52 = כ310
# 5913-55 = כ93
# 5913-56 = כ93
# 5919-32 = ל300
# 5919-33 = ל300
# 5950-23 = ה17
# 5950-24 = ה17
# 5950-30 = ה20
# 5950-31 = ה20
# 5992-5 = ה70
# 5992-6 = ה70
# 6008-16 = ה50
# 6008-17 = ה50
# 6042-17 = ל70
# 6042-18 = ל70
# 6056-8 = ל110
# 6056-9 = ל110
# 6072-15 = מ80
# 6072-16 = מ80
# 6086-18 = כ150
# 6086-19 = כ150
# 6096-25 = כ20
# 6096-26 = כ20
# 6097-36 = מ12
# 6097-37 = מ12
# 6144-29 = כ136
# 6144-30 = כ136
# 2046-29 = 18.2
# 2050-11 = 1.9
# 2199-32 = 2.1
# 2233-4 = 10
# 2357-12 = 1.6.09
# 3751-19 = 43,820
# 3848-7 = 19
# 4006-11 = 12.9
# 4009-11 = 23.40
# 4442-2 = 102
# 4973-22 = 14
# 5085-7 = 41
# 5293-36 = 543,000
# 5318-11 = 2,400
# 5339-12 = 63
# 5387-18 = 1.2.91
# 5414-8 = 1,800
# 5522-7 = 2500
# 5529-9 = 2.3
# 5564-10 = 4500
# 5847-24 = 31
# 5913-24 = 97
# 5913-42 = 97
# 5913-59 = 97
