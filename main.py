from utils import *

def translate(input_file, output_file, 
                source_lang='ar', 
                target_lang='en-US', 
                text_col=None, 
                st_removed_col='ST_removed',
                trans_col='translation',
                remove_special_tok=True,
                google=True):

    if remove_special_tok == True:
        sents = get_sentences(filename=input_file,text_col=text_col,filter_col=trans_col)
        sents_wo_special_tok = remove_special_tokens(sents)
        print(sents_wo_special_tok[:10])
        add_col(sents_wo_special_tok,input_file,input_file,st_removed_col,if_translation=False)
        trans = translate_with_google(sents_wo_special_tok,source_lang,target_lang)

    else:
        sents = get_sentences(filename=input_file,text_col=text_col,filter_col=trans_col)
        print(sents[:10])
        if target_lang == 'ko' and google == False:
            trans = translate_with_papago(sents)
        else:
            trans = translate_with_google(sents,source_lang,target_lang)

    if len(trans) != len(sents):
        # _trans = trans + ['']*(len(sents) - len(trans))
        # trans = _trans
        print('# of Translated Sentences:',len(trans))
    
    add_col(trans,input_file,output_file,trans_col,if_translation=True)

def translate_and_check(input_file, trans_output_file, cos_sim_output_file,
                source_lang='ar', 
                target_langs=['en-US','ko'], 
                text_col=None, 
                st_removed_col='ST_removed',
                remove_special_tok=True,
                google=True):

    remove_sp_token_done = False

    for target_lang in target_langs:
        translate(input_file,trans_output_file,source_lang=source_lang,target_lang=target_lang,text_col=text_col,st_removed_col=st_removed_col,trans_col=f'{target_lang}_translation',remove_special_tok=(not remove_sp_token_done) and remove_special_tok,google=True)
        translate(trans_output_file,trans_output_file,source_lang=target_lang,target_lang=source_lang,text_col=f'{target_lang}_translation',trans_col=f'{target_lang}-{source_lang}_back-translation',remove_special_tok=False,google=True)
        remove_sp_token_done = True
        input_file = trans_output_file
        text_col = st_removed_col

    for target_lang in target_langs:
        if remove_special_tok:
            ori_col = st_removed_col
        else:
            ori_col = text_col
        compare_similarity(filename=trans_output_file,ori_col=ori_col,comp_col=f'{target_lang}-{source_lang}_back-translation',output_filename=cos_sim_output_file,sim_col=f'{target_lang}-bt_cos_sim')
        trans_output_file = cos_sim_output_file    

    for target_lang in target_langs:
        df = pd.read_csv(cos_sim_output_file,encoding='utf-8')
        draw_hist(df=df,col=f'{target_lang}-bt_cos_sim',title=f'{target_lang}-{source_lang} Back Translation Cosine Similarity',show_counts=True)

def detect_hate(input_file, output_file, col, lang, if_translation=True):
    sents = get_col(file=input_file,col=col)

    if lang == 'en':
        bert_score = detect_hate_sents(sents,lang)
        add_col(data=bert_score,input_file=input_file,output_file=output_file,col='HateBERT_SBIC',if_translation=if_translation)


    elif lang == 'kr':
        bert_base, roberta_base, roberta_large = detect_hate_sents(sents,lang)
        add_col(data=bert_base,input_file=input_file,output_file=output_file,col='BERT_base',if_translation=if_translation)
        add_col(data=roberta_base,input_file=output_file,output_file=output_file,col='RoBERTa_base',if_translation=if_translation)
        add_col(data=roberta_large,input_file=output_file,output_file=output_file,col='RoBERTa_large',if_translation=if_translation)

if __name__ == '__main__':

    input_file = FILE_TO_TRANSLATE
    output_file = FILE_TO_SAVE_TRANSLATION
    cos_sim_file = FILE_TO_SAVE_COS-SIM_SCORES
  
    translate_and_check(input_file, output_file, cos_sim_file,
                source_lang='ko', 
                target_langs=['en-US'], 
                text_col=COLUMN_NAME_TO_SAVE_TRANSLATION, # column name within the output_file for translation results to be saved 
                st_removed_col=COLUMN_NAME_TO_TRANSLATE, # column name from the input_file for translation
                remove_special_tok=False, # want to remove the special tokens such as '@USER'?
                google=True) # use Google Translate for translation?
  
    compare_similarity(filename=output_file,
                        ori_col=COLUMN_NAME_TO_BE_COMPARED_1, 
                        comp_col=COLUMN_NAME_TO_BE_COMPARED_2,
                        output_filename=cos_sim_file,
                        sim_col=COLUMN_NAME_TO_SAVE_COS-SIM_SCORES) 
    
   
    
