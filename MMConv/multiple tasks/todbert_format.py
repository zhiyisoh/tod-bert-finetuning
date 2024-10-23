#!/usr/bin/env python
# coding: utf-8

# In[1]:


from utils.json_utils import load, load_all, save
from utils.generic_utils import read, list2dict
import os
import json
import sys
import re
from copy import deepcopy
from collections import defaultdict
import random
import ast

path = "C:/Users/Zhiyi/Desktop/NLC/project/tod-bert-finetuning/MMConv/multiple tasks/dataset/"
with open(path + "dialogues.json") as f:
        dialogues = json.load(f)
        
data = load(path + 'evidence_for_delex.json')
data_dict = list2dict(data, 'dialogue_idx')


def slot_info(slot):
    info = slot.split(': ')
    return (info[0].strip(), None) if len(info) == 1 else (info[0].strip(), info[1].strip())

turn_label_key = {
    'agent': 'dialog_act',
    'user': 'turn_label'
}

correct_value = {
    'yes (incl. american express & mastercard)': 'yes',
    'yes (incl. american express)': 'yes',
    'yes (incl. nfc payments & mastercard)': 'yes',
    'yes (incl. nfc payments & visa)': 'yes',
    'yes (incl. visa & american express)': 'yes',
    'yes (incl. visa & mastercard)': 'yes',
    'cocktail': 'cocktails',
    'modearte': 'moderate',
    'free & paid': 'yes',
    'desserts': 'dessert',
    'sessert': 'dessert',
    'free & paid': 'yes',
    'goos': 'good',
    'bar snacks': 'bar snack'
}

def extract_slot(slot):
    slot_conv = {
        'menu': 'menus',
        'drink': 'drinks',
        'musics': 'music',
        'reservation': 'reservations',
        'credit card': 'credit cards',
        'outdoor seatings': 'outdoor seating',
        'dining option': 'dining options',
        'wifi': 'wi-fi'
    }
    slot_name, value = slot_info(slot)
    slot_name = slot_name.lower()
    if slot_name in slot_conv:
        slot_name = slot_conv[slot_name]
    if value is not None:
        value = value.lower()
        if value in correct_value:
            value = correct_value[value]
        if slot_name == 'wi-fi' and value in ('free', 'paid', 'good'):
            value = 'yes'
        if value.replace(' ', '') in ('dontcare', 'don\'tcare', 'donotcare', 'doesnotcare', 'doesntcare', 'doesn\'tcare'):
            value = 'dontcare'
    return slot_name, value


# In[13]:


slot_opts = load( path +'ontology.json')

slot_name_map = {slot_name.lower(): slot_name for slot_name in slot_opts}

wrong_slots = {'delivery', 'open span', 'img_gts'}

telephone_matcher = re.compile('(65|65 |[+]65|[+]65 )?\d{4} ?\d{4}')


def get_candidate_value(slots):
    for slot_name, slot_info in slots.values():
        slot_name = slot_name.lower()
        slot_info

        
def delex_slot(transcript, slot, exclude_slots=set(), evidences={}):
    def dist(span1, span2):
        return min(abs(span1[0] - span2[1]), abs(span1[1] - span2[0]))

    def find_and_replace(transcript, value):
        value_padded = ' ' + value
        transcript_padded = ' ' + transcript
        transcript_list = []
        replace_with = f'[{name}]'
        last_idx = -1
        while True:
            if last_idx != -1:
                if last_idx + len(value_padded) >= len(transcript_padded):
                    break
                to_search = transcript_padded[last_idx + len(value_padded)]
            else:
                to_search = transcript_padded
            match_idx = to_search.find(value_padded)
            if match_idx != -1:
                if match_idx + len(value_padded) < len(transcript_padded) and transcript_padded[match_idx + len(value_padded)].isalnum():
                    last_idx = match_idx
                    continue
                if last_idx != -1:
                    match_idx += last_idx
                transcript_list.append(transcript[:match_idx])
                transcript_list.append(replace_with)
                last_idx = match_idx
            else:
                break
        if last_idx != -1:
            transcript_list.append(transcript[last_idx + len(value_padded) - 1:])
        if transcript_list:
            return ''.join(transcript_list), True
        return transcript, False

    name, value = extract_slot(slot)
    if name not in exclude_slots and name not in wrong_slots and value is not None:
        if slot_opts[slot_name_map[name.lower()]]['type'] == 'yes/no':
            transcript_lower = transcript.lower()
            matches = list(re.finditer(f'(^| ){value}($|[.,;! ])', transcript_lower))
            if matches:
                keyword_idx = transcript_lower.find(name)
                if keyword_idx == -1:
                    for kw in slot_opts[slot_name_map[name.lower()]]['keywords']:
                        keyword_idx = transcript_lower.find(kw)
                        if keyword_idx != -1:
                            keyword = kw
                            break
                else:
                    keyword = name
                if keyword_idx != -1:
                    first_span = matches[0].span()
                    kw_span = (keyword_idx, keyword_idx + len(keyword))
                    min_dist = [0, dist(first_span, kw_span)]
                    for i, match in enumerate(matches):
                        curr_dist = dist(match.span(), kw_span)
                        if curr_dist < min_dist[1]:
                            min_dist = [i, curr_dist]
                    min_span = matches[min_dist[0]].span()
                    t_match = matches[min_dist[0]].group(0)
                    if t_match[0].isalpha():
                        offset_begin = 0
                    else:
                        offset_begin = 1
                    if t_match[-1].isalpha():
                        offset_end = 0
                    else:
                        offset_end = 1
                    return transcript[:min_span[0] + offset_begin] + f'[{name}]' + transcript[min_span[1] - offset_end:]
                else:
                    transcript_list = []
                    if matches[0].group(0)[0].isalpha():
                        offset_begin = 0
                    else:
                        offset_begin = 1
                    transcript_list.append(transcript[:matches[0].start() + offset_begin])
                    replace_with = f'[{name}]'
                    for i, match in enumerate(matches):
                        transcript_list.append(replace_with)
                        if i < len(matches) - 1:
                            if match.group(0)[-1].isalpha():
                                offset_end = 0
                            else:
                                offset_end = 1
                            if matches[i + 1].group(0)[0].isalpha():
                                offset_begin = 0
                            else:
                                offset_begin = 1
                            transcript_list.append(transcript[match.end() - offset_end: matches[i + 1].start() + offset_begin])
                    if matches[-1].group(0)[-1].isalpha():
                        offset_end = 0
                    else:
                        offset_end = 1
                    transcript_list.append(transcript[matches[-1].end() - offset_end:])
                    return ''.join(transcript_list)
            else:
                return transcript
        else:
            value = slot.split(': ')[1]
            if name == 'telephone':
                transcipt = telephone_matcher.sub('[telephone]', transcript)
                value = telephone_matcher.sub('[telephone]', value)
            else:
                result = False
                candidates = [value, value.lower()]
                if name in evidences:
                    evidence = evidences[name]
                    if evidence:
                        if isinstance(evidence, str):
                            evidence = [evidence]
                        for e in evidence:
                            if e:
                                candidates.extend([e, e.lower()])
                for candidate in candidates:
                    if not result:
                        transcript, result = find_and_replace(transcript, candidate)
                    else:
                        break
            return transcript
    return transcript


def do_delex(dialogue, turn_idx, role='agent', exclude_slots=set()):
    def sort_slot(sa):
        if sa[0] == 'venueaddress':
            return 0
        if sa[0] == 'venuename':
            return 1
        return 2

    dialogue_in_data = data_dict[dialogue['id']]
    utt = dialogue['dialogue'][turn_idx][role]
    transcript = utt['transcript']
    if dialogue['id'] == '4131' and turn_idx == 1:
        evidences = {}
    elif dialogue['id'] == '4131' and turn_idx > 1:
        utt_in_data = dialogue_in_data['dialogue'][turn_idx - 1][role]
        evidences = {slot_name.lower(): utt_in_data['slots']['fixed'][slot_name].get('evidence', None) for slot_name in utt_in_data['slots']['fixed']}
    elif dialogue['id'] == '4154' and turn_idx == 7:
        evidences = {}
    elif dialogue['id'] == '4155' and turn_idx == 9:
        evidences = {}
    else:
        try:
            utt_in_data = dialogue_in_data['dialogue'][turn_idx][role]
        except:
            print(dialogue['id'])
        utt_in_data = dialogue_in_data['dialogue'][turn_idx][role]
        evidences = {slot_name.lower(): utt_in_data['slots']['fixed'][slot_name].get('evidence', None) for slot_name in utt_in_data['slots']['fixed']}
    slot_acts = utt[turn_label_key.get(role, 'slot-action-mapping')].items()
    slot_acts = sorted(slot_acts, key=sort_slot)
    for slot, act in slot_acts:
        transcript = delex_slot(transcript, slot, exclude_slots=exclude_slots, evidences=evidences)
    return transcript


# In[14]:

quote = "\""
ctx_token = ''
ectx_token = ''
bst_token = ',\"belief\":'
ebst_token = ''
act_token = ',\"sys_act\":'
eact_token = ''

sys_token = '\"turn_sys\":'
usr_token = '\"turn_usr\":'
img_token = '(image)'
imgsrc_token = '(imagesource)'

role2token = {
    'agent': sys_token,
    'user': usr_token
}

multi_space_matcher = re.compile('\s{2,}')

all_slot_names = {"drinks", "music", "reservations", "dining options", "venueaddress", "menus", "outdoor seating",
                  "venueneigh", "wheelchair accessible", "smoking", "parking", "venuescore",
                  "restroom", "venuename", "price", "telephone", "credit cards", "wi-fi"}

# Convert the set to a list
slot_names_list = list(all_slot_names)

# Get the index of each slot name
slot_name_indices = {name: index for index, name in enumerate(slot_names_list)}

def clean(text):
    # Remove duplicated spaces
    text = multi_space_matcher.sub(r' ', text)
    return text.strip()

def make_sample(dialogue,
                turn_idx,
                history_length=1,
                with_context=True,
                with_images=True,
                with_belief=True,
                with_action=True,
                with_response=False,
                delex=True,
                sort_slots=True,
                sort_func=None,
                with_slot_name=True,
                no_repetition=False,
                accumulate_all_slots=False,
                strict_slot_merge=False):
    ret = []
    
    if with_context:
        ret.append(ctx_token)
        ctx = make_context(dialogue, turn_idx, with_images=with_images)
        if ctx:
            ret.append(ctx)
        ret.append(ectx_token)
        dialogue_history = make_dialogue_history(dialogue, (turn_idx - history_length) if history_length > -1 else 0, turn_idx, with_images=with_images)
        ret.append(",\"dialog_history\":")
        ret.append("[")
        ret.append(dialogue_history)
        ret.append("]")

    if with_belief:
        ret.append(bst_token)
        del_belief = {}
        if accumulate_all_slots:
            bst = []
            slot_gate = []
            slot_values = []
            for i in reversed(range(0, turn_idx)):
                bst.append(make_bstate(dialogue['dialogue'][i]['bstate'], sort_slots=sort_slots, sort_func=sort_func)[0])
                slot_gate = make_bstate(dialogue['dialogue'][i]['bstate'], sort_slots=sort_slots, sort_func=sort_func)[1]
                slot_values = make_bstate(dialogue['dialogue'][i]['bstate'], sort_slots=sort_slots, sort_func=sort_func)[2]
            bst = merge_bstate(bst, sort_slots=sort_slots, sort_func=sort_func, strict=strict_slot_merge)
        else:
            bst = make_bstate(dialogue['dialogue'][turn_idx]['bstate'], sort_slots=sort_slots, sort_func=sort_func)[0] if turn_idx < len(dialogue['dialogue']) else ''
            bst = ast.literal_eval("{"+ bst + "}")
            prev = make_bstate(dialogue['dialogue'][turn_idx - 1]['bstate'], sort_slots=sort_slots, sort_func=sort_func)[0] if turn_idx != 0 else ''
            prev = ast.literal_eval("{"+ prev + "}")
            slot_gate = make_bstate(dialogue['dialogue'][turn_idx]['bstate'], sort_slots=sort_slots, sort_func=sort_func)[1] if turn_idx < len(dialogue['dialogue']) else ''
            slot_values = make_bstate(dialogue['dialogue'][turn_idx]['bstate'], sort_slots=sort_slots, sort_func=sort_func)[2] if turn_idx < len(dialogue['dialogue']) else ''
            if isinstance(bst, dict) and isinstance(prev, dict):
                
                del_belief = {key: value for key, value in bst.items() if key not in prev or prev[key] != value}

        if bst:
            ret.append(str(json.dumps(bst)))
        else: 
            ret.append("{}")
        ret.append(ebst_token)
        ret.append(",\"del_belief\":")
        ret.append(str(json.dumps(del_belief)))
        ret.append(",\"slot_gate\":")
        ret.append(str(slot_gate))
        ret.append(",\"slot_values\":")
        ret.append(str(json.dumps(slot_values)))
        ret.append(",\"slots\":")
        ret.append(str(json.dumps(slot_names_list)))

    if with_action:
        act = make_bstate_sysact(dialogue['dialogue'][turn_idx + 1]['agent']['dialog_act'], delex=False, sort_slots=sort_slots, sort_func=sort_func, with_slot_name=with_slot_name, no_repetition=True) if turn_idx < len(dialogue['dialogue']) - 1 else ''
        if act:
            ret.append(act_token)
            ret.append(quote + act + quote)
        ret.append(eact_token)

    return ' '.join(ret).strip()

def merge_bstate(bstates_reversed, sort_slots=True, sort_func=None, strict=False):
    curr_slot_names = set()
    merged = []
    for bstate in bstates_reversed:
        if bstate:
            slot_texts = bstate.split('; ')
            for slot_text in slot_texts:
                found = False
                for slot_name in all_slot_names:
                    if slot_text.startswith(slot_name):
                        found = True
                        if not strict:
                            curr_slot_names = curr_slot_names.difference(['img_gt'])
                        if slot_name not in curr_slot_names:
                            curr_slot_names.add(slot_name)
                            if slot_text not in merged:
                                merged.append(slot_text)
                if found == False:
                    print(f'1111{slot_texts}1111')
                    print(f'1111{slot_text}1111')
                    raise Exception()
    if sort_slots:
        if sort_func is None:
            merged.sort()
        else:
            merged.sort(key=lambda x: sort_func(x))
    return ', '.join(merged)

def make_slot_comps(name, value, delex=False, with_slot_name=True):
    slot_comps = [name, value]
    if delex:
        slot_comps[1] = None
    if not with_slot_name:
        slot_comps[0] = None
    return [(x.strip() if x is not None else '') for x in slot_comps]

def make_bstate(bstate, with_images=True, delex=False, sort_slots=True, sort_func=None, with_slot_name=True, no_repetition=False):
    all_slots = []
    for slot, act in bstate.items():
        slot = slot.replace('：', ':').replace('；', ':').replace(':', ': ').replace('  ', ' ')
        name, value = extract_slot(slot)
        if value is None:
            value = ''
        if name in wrong_slots:
            continue
        if with_images or name != 'img_gts':
            # if name == 'img_gts':
            #     name = 'img_gt'
            #     for v in value.split(', '):
            #         all_slots.append(make_slot_comps(name, v, delex=delex, with_slot_name=with_slot_name))
            # else:
                all_slots.append(make_slot_comps(name, value, delex=delex, with_slot_name=with_slot_name))
    if sort_slots:
        if sort_func is None:
            all_slots.sort()
        else:
            all_slots.sort(key=lambda x: sort_func(x))
    ret = [f'"{x[0]}" : "{x[1]}"' for x in all_slots]
    if no_repetition:
        ret_set = set()
        new_ret = []
        for r in ret:
            if r not in ret_set:
                ret_set.add(r)
                new_ret.append(r)
        ret = new_ret
    slot_gate = [0 for i in range(len(slot_name_indices))]
    slot_values = ["none" for i in range(len(slot_name_indices))]
    for i in all_slots:
        index = slot_name_indices[i[0]]
        slot_gate[index] = 1
        slot_values[index] = i[1]
    return clean(', '.join(ret)), slot_gate, slot_values

def make_bstate_sysact(bstate, with_images=True, delex=False, sort_slots=True, sort_func=None, with_slot_name=True, no_repetition=True):
    all_slots = []
    for slot, act in bstate.items():
        slot = slot.replace('：', ':').replace('；', ':').replace(':', ': ').replace('  ', ' ')
        name, value = extract_slot(slot)
        if value is None:
            value = ''
        if name in ["delivery"]:
            continue
        if with_images or name != 'img_gts':
            # if name == 'img_gts':
            #     name = 'img_gt'
            #     for v in value.split(', '):
            #         all_slots.append(act)
            # else:
                all_slots.append(act)
    if sort_slots:
        if sort_func is None:
            all_slots.sort()
        else:
            all_slots.sort(key=lambda x: sort_func(x))
    ret = [''.join(x) for x in all_slots]
    if no_repetition:
        ret_set = set()
        new_ret = []
        for r in ret:
            if r not in ret_set:
                ret_set.add(r)
                new_ret.append(r)
        ret = new_ret
    return clean(', '.join(ret))

def make_context(dialogue, current_turn, reverse=False, roles=('agent', 'user'), with_images=True, delex=False):
    # Start from the current turn only, no previous dialogue
    r = range(current_turn, current_turn + 1)
    
    if reverse:
        r = reversed(r)
        
    ctx = []
    for i in r:
        for role in roles:
            role_token = role2token[role]
            if delex:
                transcript = do_delex(dialogue, i, role=role, exclude_slots={
                    'open span', 'img_gts', 'openspan', 'open psan', 'opne span',
                    'open open', 'opan span', 'open sapn', 'delivery', 'open span:',
                    'oprn span', 'openn span', 'open spicy', 'opens span', 'open spam',
                    'oepn span'
                })
            else:
                transcript = dialogue['dialogue'][i][role]['transcript']
            
            images = ''
            image_sources = ''
            if with_images:
                turn_label = dialogue['dialogue'][i][role][turn_label_key.get(role, 'slot-action-mapping')]
                for slot, act in turn_label.items():
                    name, value = extract_slot(slot)
                    if name in wrong_slots:
                        continue
                    # if name == 'img_gts':
                    #     images = value
                    #     image_sources = ', '.join(dialogue['dialogue'][i][role]['imgs'])
                    #     break
            
            if transcript or images:
                ctx.append("," + role_token)
                if transcript:
                    transcript = transcript.replace("\"", '\'')
                    ctx.append(quote + transcript + quote)
                if images:
                    ctx.append(img_token)
                    ctx.append(images)
                if image_sources:
                    ctx.append(imgsrc_token)
                    ctx.append(image_sources)
                    
    return clean(' '.join(ctx).replace('\n', ' '))


def make_dialogue_history(dialogue, lower, upper, reverse=False, roles=('agent', 'user'), with_images=True, delex=False):
    # Start from the current turn only, no previous dialogue
    r = range(max(0, lower), min(len(dialogue['dialogue']), upper))
    if reverse:
        r = reverse(r)
    ctx = []
    for i in r:
        for role in roles:
            role_token = role2token[role]
            if delex:
                transcript = do_delex(dialogue, i, role=role, exclude_slots={'open span', 'img_gts', 'openspan', 'open psan', 'opne span', 'open open', 'opan span', 'open sapn', 'delivery', 'open span:', 'oprn span', 'openn span', 'open spicy', 'opens span', 'open spam', 'oepn span'})
            else:
                transcript = dialogue['dialogue'][i][role]['transcript']
            images = ''
            image_sources = ''
            if with_images:
                turn_label = dialogue['dialogue'][i][role][turn_label_key.get(role, 'slot-action-mapping')]
                for slot, act in turn_label.items():
                    name, value = extract_slot(slot)
                    if name in wrong_slots:
                        continue
            if transcript or images:
                if transcript:
                    transcript = transcript.replace("\"", '\'')
                    ctx.append(quote + transcript + quote)

    return clean(', '.join(ctx).replace('\n', ' '))

# In[15]:


splits = load(path + 'data_split.json')
for split_name, split in splits.items():
    print(f'{split_name}: {len(split)} dialogues')


# In[16]:


def prepare_path(path):
    folder, file = os.path.split(path)
    os.makedirs(folder, exist_ok=True)


input_formats = {
    'dialogpt': {
        'history_length': -1,
        'with_context': True,
        'with_images': True,
        'with_belief': False,
        'with_action': False,
        'with_response': True,
        'delex': False,
        'sort_slots': True,
        'sort_func': None,
        'with_slot_name': True,
        'no_repetition': False,
        'accumulate_all_slots': False,
        'strict_slot_merge': False
    },
    'simpletod': {
        'history_length': 4,
        'with_context': True,
        'with_images': False,
        'with_belief': True,
        'with_action': True,
        'with_response': True,
        'delex': True,
        'sort_slots': True,
        'sort_func': None,
        'with_slot_name': True,
        'no_repetition': False,
        'accumulate_all_slots': False,
        'strict_slot_merge': False
    },
    'simpletod_keep_slots': {
        'history_length': -1,
        'with_context': True,
        'with_images': False,
        'with_belief': True,
        'with_action': True,
        'with_response': True,
        'delex': True,
        'sort_slots': True,
        'sort_func': None,
        'with_slot_name': True,
        'no_repetition': False,
        'accumulate_all_slots': True,
        'strict_slot_merge': False
    }
}

output_format = "simpletod"

for split_name, split in splits.items():
    output = f'resources/{split_name}.txt'
    prepare_path(output)
    with open(output, 'w+', encoding='utf-8') as f:
        for id_ in split:
            dialogue = dialogues[id_]
            for i in range(0, len(dialogue['dialogue'])):
                f.write('{')
                f.write(f'\"ID\": {id_},')
                f.write(f'\"turn_id\":{i}')
                sample = make_sample(dialogue, i, **input_formats[output_format])
                f.write(f'{sample}')
                f.write('},')
                
print('Done!')


# In[ ]:




