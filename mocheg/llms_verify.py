from tkinter import scrolledtext
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration, LlavaNextProcessor, LlavaNextForConditionalGeneration
import re
import torch
import argparse
from read_data import get_dataset
from PIL import Image
from tqdm import tqdm, trange
import json
import heapq
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
# from llm2vec import LLM2Vec
import numpy as np

import logging
logging.disable(logging.WARNING)


def load_peft_model_text(peft_model_name, device="auto", quantile=True, flash_attention=True):
    processor = AutoTokenizer.from_pretrained(
        peft_model_name,
        model_max_length=2048,
        padding_side="left",
        truncation_side="left",
        token="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"
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
        token="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",
        device_map=device,
        attn_implementation=atten_type,
    )
    else:
        model = AutoModelForCausalLM.from_pretrained(
        peft_model_name,
        token="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",
        device_map=device,
        attn_implementation=atten_type,
    )

    return processor, model


def load_peft_model_vision2(peft_model_name, device="auto", quantile=True, flash_attention=True):
    processor = AutoProcessor.from_pretrained(
        peft_model_name,
        token="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"
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

    if quantile:
        model = LlavaForConditionalGeneration.from_pretrained(
        peft_model_name,
        quantization_config=quantization_config,
        token="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",
        device_map=device,
        use_flash_attention_2=flash_attention,
        low_cpu_mem_usage=True, 
    )
    else:
        model = LlavaForConditionalGeneration.from_pretrained(
        peft_model_name,
        token="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",
        device_map=device,
        use_flash_attention_2=flash_attention,
        low_cpu_mem_usage=True, 
    )

    return processor, model


def load_peft_model_vision3(peft_model_name, device="auto", quantile=True, flash_attention=True, image_token="<image>"):
    processor = LlavaNextProcessor.from_pretrained(
        peft_model_name,
        token="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",
        padding_side="left",
        image_token=image_token
    )
    processor.tokenizer.padding_side = "left"

    quantization_config = BitsAndBytesConfig(
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        load_in_4bit=True,
        load_in_8bit=False
    )

    if quantile:
        model = LlavaNextForConditionalGeneration.from_pretrained(
        peft_model_name,
        quantization_config=quantization_config,
        token="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",
        device_map=device,
        use_flash_attention_2=flash_attention,
    )
    else:
        model = LlavaNextForConditionalGeneration.from_pretrained(
        peft_model_name,
        token="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",
        device_map=device,
        use_flash_attention_2=flash_attention,
    )

    return processor, model


def load_peft_model_vision4(peft_model_name, device="auto", quantile=True, flash_attention=True, image_token="<image>"):
    processor = AutoProcessor.from_pretrained(
        peft_model_name,
        token="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",
        padding_side="left",
        image_token=image_token
    )
    processor.tokenizer.padding_side = "left"

    quantization_config = BitsAndBytesConfig(
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        load_in_4bit=True,
        load_in_8bit=False,
    )

    if quantile:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
        peft_model_name,
        quantization_config=quantization_config,
        token="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",
        device_map=device,
        use_flash_attention_2=flash_attention,
    )
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
        peft_model_name,
        token="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",
        device_map=device,
        use_flash_attention_2=flash_attention,
    )

    return processor, model


def make_prompt(text_evidence):
    prompt = f"""
    <|image|><|begin_of_text|>{text_evidence} \n

    Please generate a short paragraph describing the about the consistency of the image based on the given text following this template:
    \n
    <HYPOTHESIS>: Please determining whether the image is consistent with the text or not.
    <EXPLANATION>: Explanation the aligment between the image hypothesis and the text.
    <FINAL ANSWER>: Give one paragraph describing the consistency of the image and text based on the explanation.
    """

    return prompt


@torch.inference_mode()
def do_inference_vision(model, processor, prompt, images, new_token=10):
    image_data = []
    for img in images:
        temp = Image.open(img)
        keep = temp.copy()
        image_data.append(keep.resize((keep.width // 2, keep.height // 2), Image.LANCZOS))
        temp.close()
                          
    if len(image_data) > 0:
        inputs = processor(images=image_data, text=prompt, return_tensors="pt").to(model.device)
    else:
        image = Image.new('RGB', (10, 10), color='black')
        # inputs = processor(text=prompt, return_tensors="pt", padding=True).to(model.device)
        inputs = processor(images=[image], text=prompt, return_tensors="pt").to(model.device)
    
    output_ids = model.generate(
        **inputs,
        max_new_tokens=new_token,
        do_sample=False
    )
    return processor.decode(output_ids[0])


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


@torch.inference_mode()
def do_inference_vision_text_only(model, processor, prompt, new_token=10):
    inputs = processor(text=prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=new_token,
        do_sample=False,
    )
    # return processor.decode(output_ids[0])
    return processor.decode(output_ids[0][2:], skip_special_tokens=True)


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


def make_verification_prompt(claim, text_evidence, image_guides, path):
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
    
    # def make_image_description_evidence(image_explaination):
    #     expl = image_explaination
    #     expl = expl.replace("<", "")
    #     expl = expl.replace(">", "")

    #     return expl

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
        # prompt = [
        #     {
        #         "role": "user",
        #         "content": "You are an assistant to perform checking the truthfulness of a claim. \n The claim is: {} \n Here are collected evidence about the claim. Please consulting the consistency of these evidences for verifying the truthfulness of the claim: \n {}".format(claim, img_guides)
        #     },
        #     {
        #         "role": "system",
        #         "content": "Based on those given clues, please determining the truthfulness of the claim. The results must be one of these three values: refuted, supported or not enoght information. \n <RESPONSE>:"
        #     }
        # ]
        # print(prompt)
        # raise Exception
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
        # prompt = [
        #     {
        #         "role": "user",
        #         "content": "You are an assistant to perform checking the truthfulness of a claim. \n The claim is: {} \n Here are collected text evidence about the claim. Please consulting the consistency of these evidences for verifying the truthfulness of the claim: \n {}".format(claim, evidences)
        #     },
        #     {
        #         "role": "system",
        #         "content": "Based on those given clues, please determining the truthfulness of the claim. The results must be one of these three values: refuted, supported or not enoght information. \n <RESPONSE>:"
        #     }
        # ]
        # print(prompt)
        # raise Exception
    return prompt


def make_verification_prompt_no_aug(claim, text_evidence, no_evidence=False):
    if not no_evidence:
        evidences = ""
        eidx = 1
        lst_txt_evidence = [list(te.values())[0] if isinstance(te, dict) else te for te in text_evidence]
        for t in lst_txt_evidence:
            # rel_score, rel_deg = llm_relevant(l2v, claim, t, None)
            evidences += f"""
                Evidence {eidx}: {t}
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
    else:
        prompt = f"""
            Is it true that: {claim}?
            
            Let verify the truthfulness of the claim based on your knowledge.
            The truthfulness must be only one of three value: supported, refuted, or not enough information. Please think step-by-step carefully and response only the truthfulness of the claim. 
            <RESPONSE>:
        """
    # print(prompt)
    # raise Exception
    return prompt


def make_verification_prompt_with_image(claim, text_evidence, image_guides, path):
    def get_image_path_new(img_p, path):
        name = img_p.split('/')[-1]
        return path + "/images/" + name
    
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
    
    lst_images = []
    if len(image_guides) > 0:
        img_guides = ""
        eidx = 1
        
        for im in image_guides:
            # rel_score, rel_deg = llm_relevant(l2v, claim, im['text'], make_image_description_evidence(im['clean_alignment'])[0])
            img_guides += f"""
                Evidence {eidx}: 
                    Text: {im['text']}
                    Image: <image>
                    Description: {make_image_description_evidence(im['clean_alignment'])[0]}
                    Consistency: {make_image_description_evidence(im['clean_alignment'])[1]}
                    Relevance score: {im['relevance_score']}
                """
            eidx += 1
            lst_images.append(get_image_path_new(im['image'], path))          
        prompt = f"""
        Is it true that: {claim}?
        Here are the evidence for checking: 
            {img_guides}

        To verify the truthfulness of the claim, please following these steps:
        STEP 1: Consult the relevance between the claim and each given evidence based on the relevance score.
        STEP 2: Think and conclude the truthfulness of the claim based on the relevance and logical of the evidence.

        The truthfulness must be only one of three value: supported, refuted, or not enough information. Please think step-by-step carefully and response only the truthfulness of the claim. 
        <RESPONSE>: 
        """
        # print(prompt)
        # raise Exception
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
        Here are the evidence for checking: 
            {evidences}
        To verify the truthfulness of the claim, please following these steps:
        STEP 1: Consult the relevance between the claim and each given evidence based on the relevance score.
        STEP 2: Think and conclude the truthfulness of the claim based on the relevance and logical of the evidence.

        The truthfulness must be only one of three value: supported, refuted, or not enough information. Please think step-by-step carefully and response only the truthfulness of the claim. 
        <image>
        <RESPONSE>:
        """
        # print(prompt)
        # raise Exception
    return prompt, lst_images


def make_verification_prompt_with_image_no_aug(claim, text_evidence, image_evidence, path):
    def get_image_path_new(img_p, path):
        name = img_p.split('/')[-1]
        return path + "/images/" + name
    
    lst_images = []
    if len(image_evidence) > 0:
        lst_txt_evidence = [list(te.values())[0] for te in text_evidence]
        evidence_guides = ""
        for t in lst_txt_evidence:
            evidence_guides += f"""
                Text: {t}
            """
        lst_im_path = [list(im.values())[0] for im in image_evidence]

        for im in lst_im_path:
            evidence_guides += f"""
                Image: <image>
            """
            lst_images.append(get_image_path_new(im, path))          
        
        prompt = f"""
        Is it true that: {claim}?
        Here are the evidence for checking: 
            {evidence_guides}

        To verify the truthfulness of the claim, please following these steps:
        STEP 1: Consult the relevance between the claim and each given evidence based on the relevance score.
        STEP 2: Think and conclude the truthfulness of the claim based on the relevance and logical of the evidence.

        The truthfulness must be only one of three value: supported, refuted, or not enough information. Please think step-by-step carefully and response only the truthfulness of the claim. 
        <RESPONSE>: 
        """
        # print(lst_images)
        # print(prompt)
        # raise Exception
    else:
        evidences = ""
        for t in text_evidence:
            evidences += f"""
                Text: {t}
                """
        prompt = f"""
            Is it true that: {claim}?
            The evidence: 
                {evidences}
            
            To verify the truthfulness of the claim, please following these steps:
            STEP 1: Consult the relevance between the claim and each given evidence based on the relevance score.
            STEP 2: Think and conclude the truthfulness of the claim based on the relevance and logical of the evidence. If the evidence does not help concluding the claim is supported or refuted, it may be not enough information.

            The truthfulness must be only one of three value: supported, refuted, or not enough information. Please think step-by-step carefully and response only the truthfulness of the claim.
            <image> 
            <RESPONSE>:
            """
        # print(prompt)
        # raise Exception
    return prompt, lst_images


def create_verification_prompt(dataset, model, processor, path, new_token=10, no_aug=False, no_evidence=False):
    results = []
    new_dataset = read_augmented_data(dataset)
    print("---performing verification .....----")
    for sample in tqdm(new_dataset):
        if not no_aug:
            prompt = make_verification_prompt(sample['claim'], sample['text_evidence_new'], sample['alignment'], path)
        else:
            prompt = make_verification_prompt_no_aug(sample['claim'], sample['text_evidence'], no_evidence=no_evidence)
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



def create_verification_prompt_vision_text_only(dataset, model, processor, path, new_token=10):
    results = []
    new_dataset = read_augmented_data(dataset)
    print("---performing verification .....----")
    for sample in tqdm(new_dataset):
        prompt = make_verification_prompt(sample['claim'], sample['text_evidence_new'], sample['alignment'], path)
        try:
            results.append({
                **sample,
                'results': do_inference_vision_text_only(model, processor, prompt, new_token)
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


def create_verification_prompt_vision_multiimages(dataset, model, processor, path, new_token=10, no_aug=False):
    results = []
    new_dataset = read_augmented_data(dataset)
    print("---performing verification .....----")
    for sample in tqdm(new_dataset):
        if not no_aug:
            prompt, lst_imgs = make_verification_prompt_with_image(sample['claim'], sample['text_evidence_new'], sample['alignment'], path)
        else:
            prompt, lst_imgs = make_verification_prompt_with_image_no_aug(sample['claim'], sample['text_evidence'], sample['image_evidence'], path)
        try:
            results.append({
                **sample,
                'results': do_inference_vision(model, processor, prompt, lst_imgs, new_token)
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


def llm_relevant(model, claim, text, image):
    def get_level_of_relevance(score):
        if score <= 0.2:
            return "Slight"
        elif score > 0.2 and score <= 0.4:
            return "Fair"
        elif score > 0.4 and score <= 0.6:
            return "Moderate"
        elif score > 0.6 and score <= 0.8:
            return "Substaintial"
        else:
            return "Perfect"
    
    claim_prompt = "Give a claim, retrieve relevant evidence that help verify for the query: " + claim
    if image is not None:
        evidence = f"""
            Text: {text}
            Image: {image}
        """
    else:
        evidence = f"""
            Text: {text}
        """

    q_reps = model.encode([claim_prompt], show_progress_bar = False)
    d_reps = model.encode([evidence], show_progress_bar = False)

    q_reps_norm = torch.nn.functional.normalize(q_reps, p=2, dim=1)
    d_reps_norm = torch.nn.functional.normalize(d_reps, p=2, dim=1)
    cos_sim = torch.mm(q_reps_norm, d_reps_norm.transpose(0, 1))

    score = cos_sim[0].detach().item()
    return score, get_level_of_relevance(score)


def integrate_relevant_score(dataset, l2v):
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

    new_dataset = read_augmented_data(dataset)

    for sample in tqdm(new_dataset):
        claim = sample['claim']
        text_evidence = sample['text_evidence']
        align_image = sample['alignment']

        text_evidence_new = []

        if len(align_image) > 0:
            for im in align_image:
                rel_score, rel_deg = llm_relevant(l2v, claim, im['text'], make_image_description_evidence(im['clean_alignment'])[0])
                im['relevance_score'] = rel_score
                im['relevance_deg'] = rel_deg
        else:
            for t in text_evidence:
                rel_score, rel_deg = llm_relevant(l2v, claim, t, None)
                text_evidence_new.append({
                    "text": t,
                    "relevance_score": rel_score,
                    "relevance_deg": rel_deg
                })
        sample['text_evidence_new'] = text_evidence_new
    
    return new_dataset


def create_verification_prompt_system(dataset, model, processor, path, new_token=10, no_aug=False, no_evidence=False, top_k=5):
    results = []
    new_dataset = read_augmented_data(dataset)
    print("---performing verification .....----")
    for sample in tqdm(new_dataset):
        if not no_aug:
            assert len(sample['text_evidence_new']) == 0
            new_alignment = get_top_k_evidence(sample['alignment'], top_k=top_k)
            # print(new_alignment)
            # print(len(new_alignment))
            # raise Exception
            prompt = make_verification_prompt(sample['claim'], sample['text_evidence_new'], new_alignment, path)
        else:
            prompt = make_verification_prompt_no_aug(sample['claim'], sample['text_evidence'], no_evidence=no_evidence)
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


def create_verification_prompt_vision_multiimages_system(dataset, model, processor, path, new_token=10, no_aug=False, top_k=5):
    results = []
    new_dataset = read_augmented_data(dataset)
    print("---performing verification .....----")
    for sample in tqdm(new_dataset):
        if not no_aug:
            assert len(sample['text_evidence_new']) == 0
            new_alignment = get_top_k_evidence(sample['alignment'], top_k=top_k)
            prompt, lst_imgs = make_verification_prompt_with_image(sample['claim'], sample['text_evidence_new'], new_alignment, path)
        else:
            prompt, lst_imgs = make_verification_prompt_with_image_no_aug(sample['claim'], sample['text_evidence'], sample['image_evidence'], path)
        try:
            results.append({
                **sample,
                'results': do_inference_vision(model, processor, prompt, lst_imgs, new_token)
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

def get_top_k_evidence(evidence_samples, top_k):
    lst_rel_score = []
    lst_choosen_evidences = []
    for sample in evidence_samples:
        lst_rel_score.append(sample['relevance_score'])
    
    lst_rel_score = np.array(lst_rel_score)
    idx_top_k = heapq.nlargest(top_k, range(len(lst_rel_score)), lst_rel_score.take)

    for idx in idx_top_k:
        lst_choosen_evidences.append(evidence_samples[idx])

    return lst_choosen_evidences


if __name__ == '__main__':
    args = parser_args()
    # processor, model = load_peft_model_text("meta-llama/Llama-3.1-70B-Instruct")
    # processor, model = load_peft_model_text("meta-llama/Llama-3.1-8B-Instruct")
    # processor, model = load_peft_model_text("mistralai/Mixtral-8x7B-Instruct-v0.1")
    # processor, model = load_peft_model_text("meta-llama/Llama-3.2-3B-Instruct")
    processor, model = load_peft_model_text("/SSD_data1/huggingface_models/Llama-3.1-70B-Instruct")
    # processor, model = load_peft_model_text("Qwen/Qwen2.5-32B-Instruct")

    # processor, model = load_peft_model_vision3("llava-hf/llava-v1.6-vicuna-7b-hf", flash_attention=True)
    # processor, model = load_peft_model_vision3("llava-hf/llava-v1.6-vicuna-13b-hf", flash_attention=True)
    # processor, model = load_peft_model_vision3("llava-hf/llava-v1.6-34b-hf", flash_attention=True)
    # processor, model = load_peft_model_vision3("llava-hf/llava-next-72b-hf", flash_attention=True)

    # l2v = LLM2Vec.from_pretrained(
    #     "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
    #     peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
    #     device_map="auto",
    #     torch_dtype=torch.bfloat16,
    # )

    # # intergrate relevance
    # with open("./mocheg_claim_llama3.2_dev_system_0-200.json", "r") as f:
    #     dataset1 = json.load(f)
    # f.close()

    # with open("./mocheg_claim_llama3.2_dev_system_200-400.json", "r") as f:
    #     dataset2 = json.load(f)
    # f.close()

    # with open("./mocheg_claim_llama3.2_dev_system_400-600.json", "r") as f:
    #     dataset3 = json.load(f)
    # f.close()


    # dataset = dataset1 + dataset2 + dataset3

    # print(len(dataset))
    # new_data = integrate_relevant_score(dataset, l2v)
    # with open('./mocheg_claim_llama3.2_dev_system_new.json', 'w', encoding='utf-8') as f:
    #     json.dump(new_data, f, ensure_ascii=False, indent=4)
    # f.close()

    ## MAIN

    # train gold
    with open("./data/mocheg_claim_llama3.2_test_new.json", "r") as f:
        dataset = json.load(f)
    f.close()
    results = create_verification_prompt(dataset, model, processor, args.path, new_token=20, no_aug=True)
    # results = create_verification_prompt_vision_text_only(dataset, model, processor, args.path, new_token=50)
    # results = create_verification_prompt_vision_multiimages(dataset, model, processor, args.path, new_token=10, no_aug=True)
    g, p, new_results = retrieve_verification_results(results)

    with open('./mocheg_verification_test_llama3.1-70B-direct.json', 'w', encoding='utf-8') as f:
        json.dump(new_results, f, ensure_ascii=False, indent=4)
    f.close()

    # # # train system
    # with open("./mocheg_claim_llama3.2_test_system_new_label.json", "r") as f:
    #     dataset = json.load(f)
    # f.close()
    # results = create_verification_prompt_system(dataset, model, processor, args.path, new_token=20, no_aug=False, top_k=5)
    # # results = create_verification_prompt_vision_multiimages_system(dataset, model, processor, args.path, new_token=8, no_aug=False, top_k=5)
    # g, p, new_results = retrieve_verification_results(results)

    # with open('./mocheg_verification_test_qwen2.5-14B-system(1).json', 'w', encoding='utf-8') as f:
    #     json.dump(new_results, f, ensure_ascii=False, indent=4)
    # f.close()


    # # # testing 
    # with open("./mocheg_verification_test_mistral8x7B-system-direct.json", "r") as f:
    #     results = json.load(f)
    # f.close()
    # g, p, new_results = retrieve_verification_results(results)


    print("Test result micro: {}\n".format(f1_score(g, p, average='micro')))
    print("Test result macro: {}\n".format(f1_score(g, p, average='macro')))
    print("Test result Accuracy: {}\n".format(accuracy_score(g, p)))
    print(confusion_matrix(g, p, labels=[0, 1, 2]))

    # Dump 
    # # text = "Caecilians is taken from the Latin Caecus meaning 'blind', and have rudimentary eyes which can only detect light or dark. Capable of seeing the world only in black and white, Donald Trump has claimed that climate change is a hoax by the Chinese. The dermophis genus grows an extra layer of skin which their young use their teeth to peel off and eat, a behaviour known as dermatrophy. As a method of ensuring their children survive in life Donald Trump prefers granting them high roles in the Oval Office. The amphibians live almost entirely underground, believed to have lost their limbs at least 60 million years ago, as an adaptation to burrowing. Burrowing its head underground helps Donald Trump when avoiding scientific consensus on anthropogenic climate change and also appointed several energy lobbyists to the Environment Agency, where their job is to regulate the energy industry. Caecilians have tentacles used in a sensory capacity to help them find prey. 'This Thing Has Tentacles We Have No Idea About' was said by Juliette Kayyem, a former federal prosecutor and a Homeland Security official in the Obama administration talking special council Mueller's investigation into alleged interference of Russia in US elections."
    # # image = "The image is consistent with the text because the image shows a caecilian with a Donald Trump hairstyle and the text describes the similarities between the two. The image shows a caecilian with a blonde comb-over hairstyle, which is a distinctive feature of Donald Trump's appearance. The text also describes the similarities between the two, including their ability to see the world in black and white, their tendency to burrow underground, and their use of tentacles to find prey. The image and text are aligned in their depiction of the caecilian and Donald Trump, suggesting that the image is consistent with the text. The image is consistent with the text, as it shows a caecilian with a Donald Trump hairstyle and describes the similarities between the two. The image and text are aligned in their depiction of the caecilian and Donald Trump, suggesting that the image is consistent with the text. The image and text work together to create a humorous and satirical commentary on the similarities between the two, highlighting their shared characteristics and behaviors. Overall, the image is consistent with the text and effectively conveys the message that the caecilian and Donald Trump share many similarities."

    # # claim = "A newly discovered species of amphibian was named 'Dermophis donaldtrumpi' as an unflattering reference to the U.S. President."

    # # score, deg = llm_relevant(l2v, claim, text, image)

    # with open("./mocheg_claim_llama3.2_dev_system_new.json", "r") as f:
    #     dump_system = json.load(f)
    # f.close()

    # with open("./mocheg_claim_llama3.2_dev.json", "r") as f:
    #     gold_test = json.load(f)
    # f.close()

    # assert len(dump_system) == len(gold_test)
    # for i in range(0, len(dump_system)):
    #     dump_system[i]['label'] = gold_test[i]['label']

    # with open('./mocheg_claim_llama3.2_dev_system_new_label.json', 'w', encoding='utf-8') as f:
    #     json.dump(dump_system, f, ensure_ascii=False, indent=4)
    # f.close()
