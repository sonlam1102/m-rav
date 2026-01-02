from tkinter import scrolledtext
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration, LlavaNextProcessor, LlavaNextForConditionalGeneration
import re
import torch
import argparse
from read_data import get_dataset
from PIL import Image
from tqdm import tqdm, trange
import json
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
# from llm2vec import LLM2Vec

import logging
logging.disable(logging.WARNING)


def load_peft_model_text(peft_model_name, device="auto", quantile=True, flash_attention=True):
    processor = AutoTokenizer.from_pretrained(
        peft_model_name,
        padding_side="left",
        truncation_side="left",
        token=""
    )

    quantization_config = BitsAndBytesConfig(
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        load_in_4bit=True,
        load_in_8bit=False,
    )

    atten_type = "flash_attention_2" if flash_attention else "eager" 
    if quantile:
        model = AutoModelForCausalLM.from_pretrained(
        peft_model_name,
        quantization_config=quantization_config,
        token="",
        device_map=device,
        attn_implementation=atten_type,
    )
    else:
        model = AutoModelForCausalLM.from_pretrained(
        peft_model_name,
        token="",
        device_map=device,
        attn_implementation=atten_type,
    )

    return processor, model


@torch.inference_mode()
def do_inference_text(model, processor, prompt, new_token=10):
    inputs = processor(prompt, return_tensors="pt").to(model.device)
    # inputs = processor.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)

    model.generation_config.pad_token_id = processor.pad_token_id
    output_ids = model.generate(
        **inputs,
        max_new_tokens=new_token,
        do_sample=False,
    )
    return processor.decode(output_ids[0])


def read_augmented_data(data):
    for d in data:
        if len(d['alignment']) > 0:
            for a in d['alignment']:
                temp = a['alignment'].split("\n\n\n\n")[-1]
                a['clean_alignment'] = temp.replace('<|eot_id|>', '').strip()
    return data

def retrieve_verification_results(data):
    def filter_results(response):
        response = response.split("<RESPONSE>")[-1]
        if ("supported" in response or "Supported" in response) and "not supported" not in response:
            return "supported"
        elif ("refuted" in response or "Refuted" in response) or "not supported" in response:
            return "refuted"
        else:
            return "NEI"

    label2inx = {
        "supported": 2,
        "NEI": 1,
        "refuted": 0
    }

    ground_truth = []
    predict = []
    for d in data:
        predict.append(label2inx[filter_results(d['results'])])
        ground_truth.append(label2inx[d['label']])
        d['predict'] = filter_results(d['results'])
    
    return ground_truth, predict, data


def make_verification_prompt(claim, text_evidence, image_guides):
    def make_image_description_evidence(image_explaination):
        expl = image_explaination.split("<EXPLANATION>")[-1]
        expl = expl.replace("<FINAL ANSWER>:", "")
        expl = expl.replace("     ", " ")
        expl = expl.replace("\n", "")
        expl = expl.replace(": ", "")

        hyp = image_explaination.split("<EXPLANATION>")[0]
        hyp = hyp.replace("\n", "")
        hyp = hyp.replace("<HYPOTHESIS>:", "")
        # hyp = "Not consistent" if "is not consistent" in hyp else "Consistent"
        return expl, hyp

    if len(image_guides) > 0:
        img_guides = ""
        eidx = 1
        for im in image_guides:
            # rel_score, rel_deg = llm_relevant(l2v, claim, im['text'], make_image_description_evidence(im['clean_alignment'])[0])
            img_guides += f"""
                Evidence {eidx}: 
                    Text: {im['text']}
                    Image: {make_image_description_evidence(im['clean_alignment'])[0]}
                    Consistency: {make_image_description_evidence(im['clean_alignment'])[1]}
                    Relevance score: {im['relevance_score']}
                """
            eidx += 1          
        prompt = f"""
        Is it true that: {claim}?
        The evidence: 
            {img_guides}

        To verify the truthfulness of the claim, please following these steps:
        STEP 1: Consult the relevance between the claim and each given evidence based on the relevance score.
        STEP 2: Think and conclude the truthfulness of the claim based on the relevance and logical of the evidence. If the evidence does not help concluding the claim is supported or refuted, it may be not enough information.

        The truthfulness must be only one of three value: supported, refuted, or not enough information. Please think step-by-step carefully and response only the truthfulness of the claim. 
        <RESPONSE>: 
        """
    else:
        evidences = ""
        eidx = 1
        for t in text_evidence:
            # rel_score, rel_deg = llm_relevant(l2v, claim, t, None)
            evidences += f"""
            Evidence {eidx}: 
                Text: {t['text']}
                Relevance score: {t['relevance_score']}
            """
            eidx += 1
        prompt = f"""
        Is it true that: {claim}?
        The evidence: 
            {evidences}
        
        To verify the truthfulness of the claim, please following these steps:
        STEP 1: Consult the relevance between the claim and each given evidence based on the relevance score.
        STEP 2: Think and conclude the truthfulness of the claim based on the relevance and logical of the evidence. If the evidence does not help concluding the claim is supported or refuted, it may be not enough information.

        The truthfulness must be only one of three value: supported, refuted, or not enough information. Please think step-by-step carefully and response only the truthfulness of the claim. 
        <RESPONSE>:
        """
    return prompt


def make_verification_prompt_no_rel(claim, text_evidence, image_guides):
    def make_image_description_evidence(image_explaination):
        expl = image_explaination.split("<EXPLANATION>")[-1]
        expl = expl.replace("<FINAL ANSWER>:", "")
        expl = expl.replace("     ", " ")
        expl = expl.replace("\n", "")
        expl = expl.replace(": ", "")

        hyp = image_explaination.split("<EXPLANATION>")[0]
        hyp = hyp.replace("\n", "")
        hyp = hyp.replace("<HYPOTHESIS>:", "")
        return expl, hyp

    if len(image_guides) > 0:
        img_guides = ""
        eidx = 1
        for im in image_guides:
            img_guides += f"""
                Evidence {eidx}: 
                    Text: {im['text']}
                    Image: {make_image_description_evidence(im['clean_alignment'])[0]}
                    Consistency: {make_image_description_evidence(im['clean_alignment'])[1]}
                """
            eidx += 1          
        prompt = f"""
        Is it true that: {claim}?
        The evidence: 
            {img_guides}

        To verify the truthfulness of the claim, please following these steps:
        STEP 1: Consult the relevance between the claim and each given evidence based on the relevance score.
        STEP 2: Think and conclude the truthfulness of the claim based on the relevance and logical of the evidence. If the evidence does not help concluding the claim is supported or refuted, it may be not enough information.

        The truthfulness must be only one of three value: supported, refuted, or not enough information. Please think step-by-step carefully and response only the truthfulness of the claim. 
        <RESPONSE>: 
        """
    else:
        evidences = ""
        eidx = 1
        for t in text_evidence:
            evidences += f"""
            Evidence {eidx}: 
                Text: {t['text']}
                Relevance score: {t['relevance_score']}
            """
            eidx += 1
        prompt = f"""
        Is it true that: {claim}?
        The evidence: 
            {evidences}
        
        To verify the truthfulness of the claim, please following these steps:
        STEP 1: Consult the relevance between the claim and each given evidence based on the relevance score.
        STEP 2: Think and conclude the truthfulness of the claim based on the relevance and logical of the evidence. If the evidence does not help concluding the claim is supported or refuted, it may be not enough information.

        The truthfulness must be only one of three value: supported, refuted, or not enough information. Please think step-by-step carefully and response only the truthfulness of the claim. 
        <RESPONSE>:
        """
    return prompt


def make_verification_prompt_no_aligment(claim, aligment):
    evidences = ""
    eidx = 1
    for t in aligment:
        evidences += f"""
        Text: {t['text']}
        Relevance score: {t['relevance_score']}
        """
        eidx += 1
    prompt = f"""
    Is it true that: {claim}?
    The evidence: 
        {evidences}
    
    To verify the truthfulness of the claim, please following these steps:
    STEP 1: Consult the relevance between the claim and each given evidence based on the relevance score.
    STEP 2: Think and conclude the truthfulness of the claim based on the relevance and logical of the evidence. If the evidence does not help concluding the claim is supported or refuted, it may be not enough information.

    The truthfulness must be only one of three value: supported, refuted, or not enough information. Please think step-by-step carefully and response only the truthfulness of the claim. 
    <RESPONSE>:
    """
    return prompt


def make_verification_prompt_no_evidence(claim):
    prompt = f"""
    Is it true that: {claim}?
    
    To verify the truthfulness of the claim, please following these steps:
    STEP 1: Consult the relevance between the claim and each given evidence based on the relevance score.
    STEP 2: Think and conclude the truthfulness of the claim based on the relevance and logical of the evidence. If the evidence does not help concluding the claim is supported or refuted, it may be not enough information.

    The truthfulness must be only one of three value: supported, refuted, or not enough information. Please think step-by-step carefully and response only the truthfulness of the claim. 
    <RESPONSE>:
    """
    return prompt


def create_verification_prompt(dataset, model, processor, path, new_token=10):
    results = []
    new_dataset = read_augmented_data(dataset)
    print("---performing verification .....----")
    for sample in tqdm(new_dataset):
        prompt = make_verification_prompt(sample['claim'], sample['text_evidence_new'], sample['alignment'])
        # prompt = make_verification_prompt_no_rel(sample['claim'], sample['text_evidence_new'], sample['alignment'])
        # prompt = make_verification_prompt_no_aligment(sample['claim'], sample['alignment'])
        # prompt = make_verification_prompt_no_evidence(sample['claim'])
        # print(prompt)
        # raise Exception

        try:
            results.append({
                **sample,
                'results': do_inference_text(model, processor, prompt, new_token)
            })
            # print(results)
            # raise Exception
        except Exception as e:
            # raise e
            print(e)
            print(sample['claim_id'])
            results.append({
                **sample,
                'results': "This claim is supported"
            })
    return results

def parser_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--path', type=str, default="/home/s2320014/data/mocheg/val")
    parser.add_argument('--path', type=str, default="/home/sonlt/drive/data/mocheg/val")
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--system', default=False, action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parser_args()
    processor, model = load_peft_model_text("/data/huggingface_models/Qwen2.5-32B-Instruct")
    with open("./factify_claim_llama3.2_test_new.json", "r") as f:
        dataset = json.load(f)
    f.close()

    # MAIN
    # # train 
    results = create_verification_prompt(dataset, model, processor, args.path, new_token=20)
    g, p, new_results = retrieve_verification_results(results)

    # with open('./factify_verification_dev_llama-3.1-70B.json', 'w', encoding='utf-8') as f:
    with open('./factify_verification_test_no_aligment.json', 'w', encoding='utf-8') as f:
        json.dump(new_results, f, ensure_ascii=False, indent=4)
    f.close()

    print("Test result micro: {}\n".format(f1_score(g, p, average='micro')))
    print("Test result macro: {}\n".format(f1_score(g, p, average='macro')))
    print("Test result Accuracy: {}\n".format(accuracy_score(g, p)))
    print(confusion_matrix(g, p, labels=[0, 1, 2]))