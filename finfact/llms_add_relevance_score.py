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
from llm2vec import LLM2Vec

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

    if quantile:
        model = AutoModelForCausalLM.from_pretrained(
        peft_model_name,
        quantization_config=quantization_config,
        token="",
        device_map=device,
        use_flash_attention_2=flash_attention,
    )
    else:
        model = AutoModelForCausalLM.from_pretrained(
        peft_model_name,
        token="",
        device_map=device,
        use_flash_attention_2=flash_attention,
    )

    return processor, model


def load_peft_model_vision2(peft_model_name, device="auto", quantile=True, flash_attention=True):
    processor = AutoProcessor.from_pretrained(
        peft_model_name,
        token="",
        model_max_length=1024
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
        token="",
        device_map=device,
        use_flash_attention_2=flash_attention,
        low_cpu_mem_usage=True, 
    )
    else:
        model = LlavaForConditionalGeneration.from_pretrained(
        peft_model_name,
        token="",
        device_map=device,
        use_flash_attention_2=flash_attention,
        low_cpu_mem_usage=True, 
    )

    return processor, model


def load_peft_model_vision3(peft_model_name, device="auto", quantile=True, flash_attention=True, image_token="<image>"):
    processor = LlavaNextProcessor.from_pretrained(
        peft_model_name,
        token="",
        image_token=image_token,
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
    
    atten_type = "flash_attention_2" if flash_attention else "eager" 
    if quantile:
        model = LlavaNextForConditionalGeneration.from_pretrained(
        peft_model_name,
        quantization_config=quantization_config,
        token="",
        device_map=device,
        attn_implementation=atten_type,
    )
    else:
        model = LlavaNextForConditionalGeneration.from_pretrained(
        peft_model_name,
        token="",
        device_map=device,
        attn_implementation=atten_type,
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
        image_data.append(keep.resize((keep.width // 3, keep.height // 3), Image.LANCZOS))
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
        do_sample=False,
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
        if ("true" in response or "True" in response) and "not true" not in response:
            return "true"
        elif ("false" in response or "False" in response) or "not true" in response:
            return "false"
        else:
            return "NEI"

    label2inx = {
        "true": 2,
        "NEI": 1,
        "false": 0
    }

    ground_truth = []
    predict = []
    for d in data:
        predict.append(label2inx[filter_results(d['results'])])
        ground_truth.append(label2inx[d['label']])
        d['predict'] = filter_results(d['results'])
    
    return ground_truth, predict, data


def make_verification_prompt(claim, text_evidence, image_guides, path):
    # def make_image_description_evidence(image_explaination):
    #     image_explaination = image_explaination.replace("**FINAL ANSWER:**", "<FINAL ANSWER>:")
    #     image_explaination = image_explaination.replace("**HYPOTHESIS:**", "<HYPOTHESIS>:")
    #     image_explaination = image_explaination.replace("**EXPLANATION:**", "<EXPLANATION>:")

    #     expl = image_explaination.split("<EXPLANATION>")[-1]
    #     expl = expl.replace("<FINAL ANSWER>:", "")
    #     expl = expl.replace("     ", " ")
    #     expl = expl.replace("\n", "")
    #     expl = expl.replace(": ", "")

    #     hyp = image_explaination.split("<EXPLANATION>")[0]
    #     hyp = hyp.replace("\n", "")
    #     hyp = hyp.replace("<HYPOTHESIS>:", "")
    #     # hyp = "Not consistent" if "is not consistent" in hyp else "Consistent"
    #     return expl, hyp

    def make_image_description_evidence(image_explaination):
        image_explaination = image_explaination.replace("**FINAL ANSWER:**", "<FINAL ANSWER>:")
        image_explaination = image_explaination.replace("**HYPOTHESIS:**", "<HYPOTHESIS>:")
        image_explaination = image_explaination.replace("**EXPLANATION:**", "<EXPLANATION>:")

        expl = image_explaination.split("<EXPLANATION>: ")[-1]
        expl = expl.split("<FINAL ANSWER>: ")[-1]

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
        STEP 1: Reasoning the relevance between the claim and each given evidence based on the inner consistency between each evidence.
        STEP 2: Think and conclude the truthfulness of the claim based on the relevance and logical of the evidence. If the evidence does not help concluding the claim is true or false, it may be not enough information.

        The truthfulness must be only one of three value: true, false, or not enough information. Please think step-by-step carefully and response only the truthfulness of the claim. 
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
        STEP 1: Reasoning the relevance between the claim and each given evidence based on the inner consistency between each evidence.
        STEP 2: Think and conclude the truthfulness of the claim based on the relevance and logical of the evidence. If the evidence does not help concluding the claim is true or false, it may be not enough information.

        The truthfulness must be only one of three value: true, false, or not enough information. Please think step-by-step carefully and response only the truthfulness of the claim. 
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


def make_verification_prompt_no_aug(claim, text_evidence):
    evidences = ""
    eidx = 1
    for t in text_evidence:
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

        The truthfulness must be only one of three value: true, false, or not enough information. Please think step-by-step carefully and response only the truthfulness of the claim. 
        <RESPONSE>:
        """
    # print(prompt)
    # raise Exception
    return prompt


def make_verification_prompt_with_image(claim, text_evidence, image_guides, path):
    def get_image_path_new(img_p, path):
        name = img_p.split('/')[-1]
        return path + "/" + name
    
    # def make_image_description_evidence(image_explaination):
    #     image_explaination = image_explaination.replace("**FINAL ANSWER:**", "<FINAL ANSWER>:")
    #     image_explaination = image_explaination.replace("**HYPOTHESIS:**", "<HYPOTHESIS>:")
    #     image_explaination = image_explaination.replace("**EXPLANATION:**", "<EXPLANATION>:")

    #     expl = image_explaination.split("<EXPLANATION>")[-1]
    #     expl = expl.replace("<FINAL ANSWER>:", "")
    #     expl = expl.replace("     ", " ")
    #     expl = expl.replace("\n", "")
    #     expl = expl.replace(": ", "")

    #     hyp = image_explaination.split("<EXPLANATION>")[0]
    #     hyp = hyp.replace("\n", "")
    #     hyp = hyp.replace("<HYPOTHESIS>:", "")
    #     return expl, hyp

    def make_image_description_evidence(image_explaination):
        image_explaination = image_explaination.replace("**FINAL ANSWER:**", "<FINAL ANSWER>:")
        image_explaination = image_explaination.replace("**HYPOTHESIS:**", "<HYPOTHESIS>:")
        image_explaination = image_explaination.replace("**EXPLANATION:**", "<EXPLANATION>:")

        expl = image_explaination.split("<EXPLANATION>: ")[-1]
        expl = expl.split("<FINAL ANSWER>: ")[-1]

        hyp = image_explaination.split("<EXPLANATION>")[0]
        hyp = hyp.replace("\n", "")
        hyp = hyp.replace("<HYPOTHESIS>:", "")
        # hyp = "Not consistent" if "is not consistent" in hyp else "Consistent"
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

        The truthfulness must be only one of three value: true, false, or not enough information. Please think step-by-step carefully and response only the truthfulness of the claim. 
        <RESPONSE>: 
        """
        # print(prompt)
        # print(lst_images)
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

        The truthfulness must be only one of three value: true, false, or not enough information. Please think step-by-step carefully and response only the truthfulness of the claim. 
        <image>
        <RESPONSE>:
        """
        # print(prompt)
        # raise Exception
    return prompt, lst_images


def make_verification_prompt_with_image_no_aug(claim, text_evidence, image_evidence, path):
    def get_image_path_new(img_p, path):
        name = img_p.split('/')[-1]
        return path + "/" + name
    
    lst_images = []

    # Filter corrupted images (not found in the images_new folders)
    new_image_evidence = []
    for ime in image_evidence:
        try:
            Image.open(ime)
            new_image_evidence.append(ime)
        except Exception as e:
            # print(ime)
            pass
    # end 
    
    image_evidence = new_image_evidence

    if len(image_evidence) > 0:
        evidence_guides = ""
        for t in text_evidence:
            evidence_guides += f"""
                Text: {t}
            """

        for im in image_evidence:
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

        The truthfulness must be only one of three value: true, false, or not enough information. Please think step-by-step carefully and response only the truthfulness of the claim. 
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

            The truthfulness must be only one of three value: true, false, or not enough information. Please think step-by-step carefully and response only the truthfulness of the claim.
            <image> 
            <RESPONSE>:
            """
        # print(prompt)
        # raise Exception
    return prompt, lst_images


def create_verification_prompt(dataset, model, processor, path, new_token=10, no_aug=False):
    results = []
    new_dataset = read_augmented_data(dataset)
    print("---performing verification .....----")
    for sample in tqdm(new_dataset):
        # prompt = make_verification_prompt(sample['claim'], sample['text_evidence_new'], sample['alignment'], path)
        if not no_aug:
            prompt = make_verification_prompt(sample['claim'], sample['text_evidence_new'], sample['alignment'], path)
        else:
            prompt = make_verification_prompt_no_aug(sample['claim'], sample['text_evidence'])
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
                'results': "This claim is true"
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
                'results': "This claim is true"
            })
    return results


def create_verification_prompt_vision_multiimages(dataset, model, processor, path, new_token=10, no_aug=False):
    results = []
    new_dataset = read_augmented_data(dataset)
    print("---performing verification .....----")
    for sample in tqdm(new_dataset):
        # prompt, lst_imgs = make_verification_prompt_with_image(sample['claim'], sample['text_evidence_new'], sample['alignment'], path)
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
                'results': "This claim is true"
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
    # def make_image_description_evidence(image_explaination):
    #     image_explaination = image_explaination.replace("**FINAL ANSWER:**", "<FINAL ANSWER>:")
    #     image_explaination = image_explaination.replace("**HYPOTHESIS:**", "<HYPOTHESIS>:")
    #     image_explaination = image_explaination.replace("**EXPLANATION:**", "<EXPLANATION>:")

    #     expl = image_explaination.split("<EXPLANATION>")[-1]
    #     expl = expl.replace("<FINAL ANSWER>:", "")
    #     expl = expl.replace("     ", " ")
    #     expl = expl.replace("\n", "")
    #     expl = expl.replace(": ", "")

    #     hyp = image_explaination.split("<EXPLANATION>")[0]
    #     hyp = hyp.replace("\n", "")
    #     hyp = hyp.replace("<HYPOTHESIS>:", "")
    #     return expl, hyp

    def make_image_description_evidence(image_explaination):
        image_explaination = image_explaination.replace("**FINAL ANSWER:**", "<FINAL ANSWER>:")
        image_explaination = image_explaination.replace("**HYPOTHESIS:**", "<HYPOTHESIS>:")
        image_explaination = image_explaination.replace("**EXPLANATION:**", "<EXPLANATION>:")

        expl = image_explaination.split("<EXPLANATION>: ")[-1]
        expl = expl.split("<FINAL ANSWER>: ")[-1]

        hyp = image_explaination.split("<EXPLANATION>")[0]
        hyp = hyp.replace("\n", "")
        hyp = hyp.replace("<HYPOTHESIS>:", "")
        # hyp = "Not consistent" if "is not consistent" in hyp else "Consistent"
        return expl, hyp

    new_dataset = read_augmented_data(dataset)

    for sample in tqdm(new_dataset):
        claim_text = sample['claim']

        text_evidence = sample['text_evidence']
        align_image = sample['alignment']

        text_evidence_new = []

        if len(align_image) > 0:
            for im in align_image:
                rel_score, rel_deg = llm_relevant(l2v, claim_text, im['text'], make_image_description_evidence(im['clean_alignment'])[0])
                im['relevance_score'] = rel_score
                im['relevance_deg'] = rel_deg
        else:
            for t in text_evidence:
                rel_score, rel_deg = llm_relevant(l2v, claim_text, t, None)
                text_evidence_new.append({
                    "text": t,
                    "relevance_score": rel_score,
                    "relevance_deg": rel_deg
                })
        sample['text_evidence_new'] = text_evidence_new
    
    return new_dataset



if __name__ == '__main__':
    args = parser_args()
    with open("./result_dump/finfact_claim_new.json", "r") as f:
        dataset = json.load(f)
    f.close()

    ##  intergrate relevance
    l2v = LLM2Vec.from_pretrained(
        "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
        peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    new_data = integrate_relevant_score(dataset, l2v)
    with open('./finfact_claim_new.json', 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)
    f.close()

