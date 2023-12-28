import os
import csv
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from easynmt import EasyNMT
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from google.cloud import translate
from googleapiclient import discovery

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F

import emoji
import re

def get_sentences(filename='./gender_normal_tweets.csv',text_col='tweet',filter_col=None):
    df = pd.read_csv(filename)
    if filter_col == None:
        sents = df[text_col].tolist()
    elif filter_col not in df.keys():
        sents = df[text_col].tolist()
    else:
        # filter out sentences that are not translated
        sents = df[df[filter_col].apply(lambda x: isinstance(x, float))][text_col].tolist() 
    # print(sents)
    print('Total Sentences:',len(df))
    print('To be Translated:',len(sents))
    return sents


def translate_with_MT(sents, modelname='opus-mt'):
    if modelname.startswith('m2m100'):
        trans = []
    
        model = M2M100ForConditionalGeneration.from_pretrained("facebook/"+modelname)
        print('model loaded!')
        tokenizer = M2M100Tokenizer.from_pretrained("facebook/"+modelname)
        print('tokenizer loaded!')
        tokenizer.source_lang = 'ar'

        
        # encoded_ar = tokenizer(sents, padding=True, return_tensors="pt")
        # generated_tokens = model.generate(**encoded_ar, forced_bos_token_id=tokenizer.get_lang_id("en"))
        # trans = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        for sent in sents:
            encoded_ar = tokenizer(sent, padding=True, return_tensors="pt")
            generated_tokens = model.generate(**encoded_ar, forced_bos_token_id=tokenizer.get_lang_id("en"))
            trans_ar = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            print(trans_ar)
            trans.append(trans_ar)

    else:
        model = EasyNMT(modelname)
        trans = model.translate(sents,source_lang='ar',target_lang='en')
    
    return trans


def translate_with_google(sents,source_lang='ar',target_lang='en-US'):
    # Initialize Translation client'

    
    def translate_text(sents, project_id=""):
        """Translating Text."""

        client = translate.TranslationServiceClient()

        location = "global"

        parent = f"projects/{project_id}/locations/{location}"

        # Translate text to English
        # Detail on supported types can be found here:
        # https://cloud.google.com/translate/docs/supported-formats
        response = client.translate_text(
            request={
                "parent": parent,
                "contents": sents,
                "mime_type": "text/plain",  # mime types: text/plain, text/html
                "source_language_code": source_lang,
                "target_language_code": target_lang,
            }
        )

        trans = []
        # Display the translation for each input text provided
        for translation in response.translations:
            # print("Translated text: {}".format(translation.translated_text))
            trans.append(translation.translated_text)

        return trans


    trans = []
    tens = len(sents)//10
    char_count = 0

    # break_flag = False

    if os.path.exists('trans.txt'):
        os.remove('trans.txt')

    with open('trans.txt','a') as f:
    
        for i in tqdm(range(tens)):
            _sents = sents[i*10:i*10+10]
            try:
                _trans = list(translate_text(_sents)) # 10 sents a time
            except: # if character count of 10 sentences is too many, error.
                _trans = []
                # print(len(_sents))
                for sent in _sents:
                    print()
                    print(sent)
                    try:
                        _trans.append(translate_text([sent])[0]) # one at a time
                    except:
                        _trans.append('')
                
            trans += _trans
            f.writelines([t+'\n' for t in _trans])

            for s in _sents:
                if type(s) == str:
                    char_count += len(s)

            for t in _trans:
                if type(t) == str:
                    char_count += len(t)

            # if char_count > 500000:
            #     break_flag = True
            #     print('last sentence idx:',i*10+10-1)
            #     break
        
        _sents = sents[tens*10:]
        # if len(_sents) > 0 and break_flag == False:
        if len(_sents) > 0:
            try:
                _trans = list(translate_text(_sents)) # 10 sents a time
            except: # if character count of 10 sentences is too many, error.
                _trans = []
                # print(len(_sents))
                for sent in _sents:
                    print()
                    print(sent)
                    try:
                        _trans.append(translate_text([sent])[0]) # one at a time
                    except:
                        _trans.append('')
            trans += _trans

        for s in _sents:
            if type(s) == str:
                char_count += len(s)

        for t in _trans:
            if type(t) == str:
                char_count += len(t)

    print(f'Character Count: {char_count}')
    # trans = list(translate_text(sents))

    return trans

def translate_with_papago(sents):
    import sys
    import urllib.request
    client_id = "" # 개발자센터에서 발급받은 Client ID 값
    client_secret = "" # 개발자센터에서 발급받은 Client Secret 값

    if len(sents) > 10000:
        sents = sents[:10000]
    trans = []
    total_len = 0

    for sent in tqdm(sents):
        try:
            encText = urllib.parse.quote(sent)
            data = "source=en&target=ko&text=" + encText
            url = "https://openapi.naver.com/v1/papago/n2mt"
            request = urllib.request.Request(url)
            request.add_header("X-Naver-Client-Id",client_id)
            request.add_header("X-Naver-Client-Secret",client_secret)
            response = urllib.request.urlopen(request, data=data.encode("utf-8"))
            rescode = response.getcode()
            if(rescode==200):
                response_body = response.read()
                _trans = response_body.decode('utf-8')
                jdict = json.loads(_trans)
                # print(type(jdict))
                _trans = jdict['message']['result']['translatedText']
                # print(_trans)
                with open('trans.txt','a') as f:
                    f.write(_trans+'\n')
                trans.append(_trans)
            else:
                print("Error Code:" + rescode)
        except:
            break

    return trans

def preprocess(text):
    # Remove white spaces
    text = text.strip()
    text = re.sub("<LF>","\n",text)
    text = re.sub('[\n]+', ". ", text)
    
    # Mask twitter handles
    text = text.replace('＠','@')
    text = re.sub("(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([_A-Za-z\u4e00-\u9fff]+[A-Za-z0-9-_\u4e00-\u9fff]+)", "@USER", text)
    text = re.sub("@[A-Za-z0-9-_\u4e00-\u9fff]+", "@USER", text)
    
    # Mask urls
    WEB_URL_REGEX = r"""(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))"""
    text = re.sub(WEB_URL_REGEX, "URL", text)
    
    # Remove emojis
    text = emoji.replace_emoji(re.sub('&#[0-9]+;', '', text), replace='')
    
    # Replace htlm characters
    text = re.sub("&lt;","<", re.sub("&gt;",">",re.sub("&amp;","&",text)))

    
    text = text.replace('@user','').replace('@url','').replace('@USER','').replace('URL','').replace('RT','').replace('<url>','').replace('<email>','').replace('<user>','').strip()
    text = re.sub("\s+", ' ', re.sub('\n','. ',text))
    return text.strip()

def remove_special_tokens(sents):
    res = [preprocess(s) for s in sents]
    return res

# def remove_special_tokens(sents):
#     _sents = [re.sub('\n+', '. ', s.replace('@user','').replace('@url','').replace('@USER','').replace('URL','').replace('<LF>','\n').replace('RT','').strip()) for s in sents]
#     res = [emoji.replace_emoji(s, replace='') for s in _sents]
#     # print(res)
#     return res

def add_col(data,input_file,output_file,col='translation',if_translation=True):
    df = pd.read_csv(input_file)
    if if_translation:
        if col in df.columns.tolist():
            existing = df[df[col].apply(lambda x: isinstance(x, str))][col].tolist()
            nan = ['']*(len(df[col])-len(existing)-len(data))
            print('Existing Translations:',len(existing))
            print('New Translations:',len(data))
            print('NaNs:',len(nan))
            print(len(existing),'+',len(data),'+',len(nan),'=',len(existing)+len(data)+len(nan))
            print('Total Sentences:',len(df[col]))
            data = existing+data+nan
        else:
            nan = ['']*(len(df)-len(data))
            print('New Translations:',len(data))
            print('NaNs:',len(nan))
            print(len(data),'+',len(nan),'=',len(data)+len(nan))
            print('Total Sentences:',len(df))
            data = data + nan

    if len(df) == len(data):
        df[col] = data
    else:
        nan = ['']*(len(df)-len(data))
        print('New Hate Scores:',len(data))
        print('NaNs:',len(nan))
        print(len(data),'+',len(nan),'=',len(data)+len(nan))
        print('Total Sentences:',len(df))
        data = data + nan
        df[col] = data
    df.to_csv(output_file,sep=',',encoding='utf-8',index=False)

def get_col(file,col):
    df = pd.read_csv(file)
    series = df[col]
    return series.to_list()

def save_en_sentences(trans,filename='./gender_normal_tweets.csv',modelname='opus-mt'):
    df = pd.read_csv(filename)
    df['translation'] = trans
    filename = filename[:-4]+'-'+modelname+filename[-4:]
    df.to_csv(filename,sep=',',encoding='utf-8',index=False)

def get_normal(filename='./gender_normal_tweets.csv',modelname='opus-mt'):
    filename = filename[:-4]+'-'+modelname+filename[-4:]
    df = pd.read_csv(filename)
    normals = df[df.sentiment == 'normal']

    return normals

def get_normal_trans(filename='./gender_normal_tweets.csv',modelname='opus-mt'):
    normals = get_normal(filename,modelname)
    trans = normals['translation']

    return trans.to_list()

def get_notnormal_en(filename='./en_gender_notnormal_dataset.csv',col='tweet'):
    df = pd.read_csv(filename)
    sents = df[col]

    return sents.to_list()

def json_dump(data,filename):
    with open(filename,'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def read_json(filename):
    with open(filename,'r') as f:
        data = json.load(f)

def convert_json_to_csv(filename):
    with open(filename,'r') as f:
        data = json.load(f)

    output_filename = filename[:-4]+'csv'

    with open(output_filename,'w') as f:
        csv_writer = csv.writer(f)

        if type(data) == list:
            # write header of csv
            keys = data[0].keys()
            csv_writer.writerow(keys)

            for line in data:
                csv_writer.writerow(line.values())

def compare_similarity(filename='./en_gender_notnormal_dataset.csv',ori_col='tweet',comp_col='opus-mt',output_filename='./gender_normal_tweets.csv',sim_col='cos_sim'):
    ori = get_col(filename,ori_col)
    comp = get_col(filename,comp_col)
    
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    ori_embeddings = model.encode(ori,convert_to_tensor=True)
    comp_embeddings = model.encode(comp,convert_to_tensor=True)

    cos = torch.nn.CosineSimilarity(dim=1)
    cos_sim = cos(ori_embeddings,comp_embeddings) 
    print(cos_sim.shape)

    add_col(cos_sim.tolist(),input_file=filename,output_file=output_filename,col=sim_col,if_translation=False)

def detect_hate_perspective(sent):
    API_KEY = ''

    client = discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=API_KEY,
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        static_discovery=False,
    )

    analyze_request = {
        'comment': { 'text': sent },
        "languages": ["en"],
        'requestedAttributes': {'TOXICITY': {},
                                'SEVERE_TOXICITY': {},
                                'IDENTITY_ATTACK': {},
                                'INSULT': {},
                                'PROFANITY': {},
                                'THREAT': {}}
    }

    response = client.comments().analyze(body=analyze_request).execute()
    score = response['attributeScores']['TOXICITY']['spanScores'][0]['score']['value']

    return score

def detect_hate_hateBERT(sent,toxigen_hatebert,tokenizer):
    
    inputs = tokenizer(sent,max_length=512,return_tensors='pt')
    emb = inputs['input_ids']
    att = inputs['attention_mask']
    
    res = toxigen_hatebert(emb,attention_mask=att)
    
    logits = res.logits[0]
    
    probabilities = F.softmax(logits, dim=-1)
    
    score = float(probabilities[1])
    
    return score

def detect_hate_roberta(sent,toxigen_roberta,tokenizer):
    
    inputs = tokenizer(sent,max_length=512,return_tensors='pt')
    emb = inputs['input_ids']
    att = inputs['attention_mask']
    res = toxigen_roberta(emb,attention_mask=att)
    
    logits = res.logits[0]
    
    probabilities = F.softmax(logits, dim=-1)
    score = float(probabilities[1])
    
    return score

def detect_hate(sent,model,tokenizer):
    
    model.eval()
    emb = tokenizer.encode(sent,padding='max_length',max_length=256,truncation=True,return_tensors='pt')
    with torch.no_grad():
        res = model(emb,token_type_ids=None)
    
    logits = res.logits[0]
    
    probabilities = F.softmax(logits, dim=-1)
    
    
    argmax = torch.argmax(logits)
    
    score = float(probabilities[1])
    
    return score

def detect_hate_arabert(input_file,output_file,col):
    tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv02-twitter")
    num_added_toks = tokenizer.add_tokens(['<LF>','@USER','URL'])
    model = BertForSequenceClassification.from_pretrained('aubmindlab/bert-base-arabertv02-twitter', num_labels=2)
    model.resize_token_embeddings(len(tokenizer))
    model.cuda()
    checkpoint = torch.load('./finetune/mlma_arabert_base_model_best.pth.tar')

    model.load_state_dict(checkpoint['state_dict'])

    df = pd.read_csv(input_file,encoding='utf-8')
    sents = df[col]

    scores = []
    
    for s in tqdm(sents):
        inputs = tokenizer(sent,return_tensors='pt')
        emb = inputs['input_ids']
        att = inputs['attention_mask']
        res = model(emb,attention_mask=att)
        logits = res.logits[0]
        prob = F.softmax(logits,dim=-1)
        score = float(prob[1])
        scores.append(score)

    

    df['arabert_osact'] = scores
    df.to_csv(output_file,encoding='utf-8',index=False)

def detect_hate_sents(sents,lang):
    # pers_score = []
    if lang == 'en':
        bert_score = []

        bert_checkpoint = './finetune/HateBERT_SBIC_best.pth.tar'
        
        device = torch.device('cpu')

        bert_base = AutoModelForSequenceClassification.from_pretrained('./finetune/HateBERT_hateval', num_labels=2)
        bert_base_tokenizer = AutoTokenizer.from_pretrained('./finetune/HateBERT_hateval')
        print("=> loading checkpoint '{}'".format(bert_checkpoint))
        checkpoint = torch.load(bert_checkpoint,map_location=device)
        state_dict = checkpoint['state_dict']
        bert_base.load_state_dict(remove_module(state_dict))
        bert_base.eval()

        for i,sent in enumerate(tqdm(sents)):
            if type(sent) != str:
                break
            bert_score.append(detect_hate(sent,bert_base,bert_base_tokenizer))
            
        return bert_score

    elif lang == 'kr':
        bert_base_scores = []
        roberta_base_scores = []
        roberta_large_scores = []

        bert_base_checkpoint = './finetune/klue_bert_base_best.pth.tar'
        roberta_base_checkpoint = './finetune/klue_roberta_base_best.pth.tar'
        roberta_large_checkpoint = './finetune/klue_roberta_large_best.pth.tar'

        device = torch.device('cpu')

        bert_base = AutoModelForSequenceClassification.from_pretrained('klue/bert-base', num_labels=2)
        bert_base_tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')
        print("=> loading checkpoint '{}'".format(bert_base_checkpoint))
        checkpoint = torch.load(bert_base_checkpoint,map_location=device)
        state_dict = checkpoint['state_dict']
        bert_base.load_state_dict(remove_module(state_dict))
        bert_base.eval()
        
        roberta_base = AutoModelForSequenceClassification.from_pretrained('klue/roberta-base', num_labels=2)
        roberta_base_tokenizer = AutoTokenizer.from_pretrained('klue/roberta-base')
        print("=> loading checkpoint '{}'".format(roberta_base_checkpoint))
        checkpoint = torch.load(roberta_base_checkpoint,map_location=device)
        state_dict = checkpoint['state_dict']
        roberta_base.load_state_dict(remove_module(state_dict))
        roberta_base.eval()

        roberta_large = AutoModelForSequenceClassification.from_pretrained('klue/roberta-large', num_labels=2)
        roberta_large_tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large')
        print("=> loading checkpoint '{}'".format(roberta_large_checkpoint))
        checkpoint = torch.load(roberta_large_checkpoint,map_location=device)
        state_dict = checkpoint['state_dict']
        roberta_large.load_state_dict(remove_module(state_dict))
        roberta_large.eval()


        for i,sent in enumerate(tqdm(sents)):
            if type(sent) != str:
                break
            bert_base_scores.append(detect_hate(sent,bert_base,bert_base_tokenizer))
            roberta_base_scores.append(detect_hate(sent,roberta_base,roberta_base_tokenizer))
            roberta_large_scores.append(detect_hate(sent,roberta_large,roberta_large_tokenizer))

        return bert_base_scores,roberta_base_scores,roberta_large_scores

    else:
        return

def get_metrics_df(filename,score_col,label_col):
    df = pd.read_csv(filename)
    df = df[df['translation'].apply(lambda x: isinstance(x, str))]
    scores = df[score_col]
    labels = df[label_col]
    preds = [1 if score>=0.5 else 0 for score in scores.tolist()]
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    accuracy = accuracy_score(labels, preds)


    print(f"***Results for {score_col}***")
    print(f"accuracy : {accuracy: .2f}")
    print(f"Precision : {precision: .2f}")
    print(f"Recall : {recall: .2f}")
    print(f"F1 : {f1: .2f}")
    print()

def remove_module(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

def draw_piechart(df,col):
    not_toxic = 0
    slight = 0
    toxic = 0
    for score in df[col].tolist():
        if score < 0.5:
            not_toxic += 1
        elif score < 0.8:
            slight += 1
        else:
            toxic += 1

    fig, ax = plt.subplots(figsize=(15, 5), subplot_kw=dict(aspect="equal"))

    data = [not_toxic,slight,toxic]
    ingredients = ["Not Toxic","Slightly Toxic","Toxic"]
    colors = ['limegreen','gold','red']

    def func(pct, allvals):
        absolute = int(np.round(pct/100.*np.sum(allvals)))
        if absolute == 0:
            return ''
        return "{:.1f}%\n({:d} tweets)".format(pct, absolute)


    wedges, texts, autotexts = ax.pie(data, autopct=lambda pct: func(pct, data),colors=colors,
                                      wedgeprops=dict(linewidth=0.7, edgecolor='w'),textprops=dict(color="black"))

    ax.legend(wedges, ingredients,
              title="Ingredients",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1))

    plt.setp(autotexts, size=8, weight="bold")

    ax.set_title(f"{col} Hate Score Distribution")

    plt.show()

def draw_hist(df,col,title=None,show_counts=True):
    bins = [l/10 for l in list(range(0,11))]
    bins[-1] += 0.000001
    data = df[col].tolist()

    counts, edges, bars = plt.hist(data,edgecolor='white',bins=bins)
    
    if title == None:
        plt.title(f'{col} Score Histogram')
    else:
        plt.title(title)
    if show_counts:
        plt.bar_label(bars)

    # plt.ylim(0,80)
    if not os.path.exists('./figures'):
        os.mkdir('./figures')
    # plt.show()
    if title == None:
        plt.savefig(f'./figures/{col} Score Histogram')
    else:
        plt.savefig(f'./figures/{title}')

    plt.clf()


def draw_charts(df,cols,hist_show_counts=False):
    df = df[df['translation'].apply(lambda x: isinstance(x, str))]
    for col in cols:
        draw_hist(df,col,hist_show_counts)
        draw_piechart(df,col)

if __name__ == "__main__":
    sentence = 'I wish there was a law to prevent the harem from tweeting about sports.'
    print(sentence)
    print()
    toxigen_hatebert = AutoModelForSequenceClassification.from_pretrained("tomh/toxigen_hatebert")
    tokenizer_hatebert = AutoTokenizer.from_pretrained("tomh/toxigen_hatebert",device=0)
    toxigen_hatebert.resize_token_embeddings(len(tokenizer_hatebert))

    toxigen_roberta = AutoModelForSequenceClassification.from_pretrained("tomh/toxigen_roberta")
    tokenizer_roberta = AutoTokenizer.from_pretrained("tomh/toxigen_roberta",device=0)

    print('Google\'s Perspective API: ')
    detect_hate_perspective(sentence)
    print()
    print('HateBERT: ')
    detect_hate_hateBERT(sentence,toxigen_hatebert,tokenizer_hatebert)
    print()
    print('RoBERTa: ')
    detect_hate_roberta(sentence,toxigen_roberta,tokenizer_roberta)
