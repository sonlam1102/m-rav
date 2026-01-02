from transformers import AutoTokenizer, AutoModelForCausalLM, MllamaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration, LlavaNextProcessor, LlavaNextForConditionalGeneration
import re
import torch
import argparse
from read_data import get_dataset
from PIL import Image
from tqdm import tqdm, trange
import json
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import sys

import logging
logging.disable(logging.WARNING)

def clean_data(text):
    if str(text) == 'nan':
        return text
    text = re.sub("(<p>|</p>|@)+", '', text)
    return text.strip()


def encode_one_sample(sample):
    claim_id = sample[0]
    claim = sample[1]
    text_evidence = sample[3]
    image_evidence = sample[4]
    label = sample[5]

    label2idx_new = {
        'Support_Text': 'supported',
        'Support_Multimodal': 'supported',
        'Insufficient_Text': 'NEI',
        'Insufficient_Multimodal': 'NEI',
        'Refute': 'refuted',
    }

    encoded_sample = {}
    encoded_sample["claim_id"] = str(claim_id)
    encoded_sample["claim"] = claim
    encoded_sample["label"] = label2idx_new[label]
    encoded_sample['text_evidence'] = text_evidence.tolist()
    encoded_sample['image_evidence'] = image_evidence

    return encoded_sample


class ClaimVerificationDataset(torch.utils.data.Dataset):
    def __init__(self, claim_verification_data):
        self._data = claim_verification_data
        # self._processor = processor

        self._encoded = []
        for d in self._data:
            self._encoded.append(encode_one_sample(d))

    def __len__(self):
        return len(self._encoded)

    def __getitem__(self, idx):
        return self._encoded[idx]

    def to_list(self):
        return self._encoded


def load_peft_model_vision(peft_model_name, device="auto"):
    processor = AutoProcessor.from_pretrained(
        peft_model_name,
        model_max_length=2000,
        padding_side="left",
        truncation_side="left",
        token="",
    )

    quantization_config = BitsAndBytesConfig(
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        load_in_4bit=True,
    )

    model = MllamaForConditionalGeneration.from_pretrained(
        peft_model_name,
        quantization_config=quantization_config,
        token="",
        device_map="auto",
        use_flash_attention_2=False,
    )

    return processor, model

# def load_peft_model_vision2(peft_model_name, device="auto"):
#     processor = AutoProcessor.from_pretrained(
#         peft_model_name,
#         model_max_length=2048,
#         padding_side="left",
#         truncation_side="left",
#         token="",
#         trust_remote_code=True,
#     )

#     quantization_config = BitsAndBytesConfig(
#         llm_int8_threshold=6.0,
#         llm_int8_has_fp16_weight=False,
#         bnb_4bit_compute_dtype=torch.bfloat16,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type="nf4",
#         load_in_4bit=True,
#     )

#     model = AutoModelForImageTextToText.from_pretrained(
#         peft_model_name,
#         quantization_config=quantization_config,
#         token="",
#         device_map=device,
#         # _attn_implementation='eager',
#         trust_remote_code=True
#     )

#     return processor, model


def load_peft_model_vision2(peft_model_name, device="auto", quantile=True, flash_attention=True):
    processor = AutoProcessor.from_pretrained(
        peft_model_name,
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


def load_peft_model_vision3(peft_model_name, device="auto", quantile=True, flash_attention=True):
    processor = LlavaNextProcessor.from_pretrained(
        peft_model_name,
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
        model = LlavaNextForConditionalGeneration.from_pretrained(
        peft_model_name,
        quantization_config=quantization_config,
        token="",
        device_map=device,
        use_flash_attention_2=flash_attention,
    )
    else:
        model = LlavaNextForConditionalGeneration.from_pretrained(
        peft_model_name,
        token="",
        device_map=device,
        use_flash_attention_2=flash_attention,
    )

    return processor, model


def load_peft_model_text(peft_model_name, device="auto", quantile=True, flash_attention=True):
    processor = AutoTokenizer.from_pretrained(
        peft_model_name,
        model_max_length=2048,
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
        load_in_4bit=False,
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
def do_inference_vision(model, processor, prompt, image):
    image_data = Image.open(image)

    inputs = processor(image_data, prompt, return_tensors="pt").to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.2,    
        top_p=0.5,
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

@torch.inference_mode()
def do_inference_vision_verification(model, processor, prompt, image):
    prompt = processor.apply_chat_template(prompt, add_generation_prompt=True)
    if len(image) > 0:
        image_data = [Image.open(img) for img in image]
        inputs = processor(images=image_data, text=prompt, padding=True, return_tensors="pt").to(model.device)
    else:
        inputs = processor(text=prompt, padding=True, return_tensors="pt").to(model.device)

    output_ids = model.generate(
        **inputs,
        eos_token_id=processor.tokenizer.eos_token_id, 
        max_new_tokens=5,
        do_sample=False,
    )
    # output_ids = output_ids[:, inputs['input_ids'].shape[1]:]
    return processor.decode(output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)


def create_align_form(dataset, model, processor, path):
    def get_image_path_new(img_p, path):
        name = img_p.split('/')[-1]
        return path + "/images/" + name
    
    results = []
    print("---performing.....----")
    for sample in tqdm(dataset):
        align_text_image = []
        try:
            align_text_image.append({
            'text': sample['text_evidence'][0],
            'image': sample['image_evidence'],
            'alignment': do_inference_vision(model, processor, make_prompt(sample['text_evidence'][0]), sample['image_evidence'])
            })
        except Exception as e:
            print(e)
            align_text_image = []

        results.append({
            **sample,
            'alignment': align_text_image
        })

    return results


def create_align_form_system(dataset, model, processor, path):
    def get_image_path_new(img_p, path):
        name = img_p.split('/')[-1]
        return path + "/images/" + name
    
    results = []
    print("---performing.....----")
    for sample in tqdm(dataset):
        align_text_image = []
        if len(sample['image_evidence']) > 0:
            for ie in sample['image_evidence']:
                for te in sample['text_evidence']:
                    image = get_image_path_new([*ie.values()][0], path)
                    align_text_image.append({
                        'text': [*te.values()][0],
                        'image': image,
                        'alignment': do_inference_vision(model, processor, make_prompt([*te.values()][0]), image)
                    })
        results.append({
            **sample,
            'alignment': align_text_image
        })

    return results


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
        if "supported" in response:
            return "supported"
        elif "refuted" in response:
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
            img_guides += f"""
            Text evidence: {im['text']}
            Image evidence: {make_image_description_evidence(im['clean_alignment'])[0]}
            Evidence consistency: {make_image_description_evidence(im['clean_alignment'])[1]}
            
            """
            eidx += 1
        
        prompt = f"""
        You are an assistant that help judging the truthfulness of a claim.
        The claim is: {claim}
        You should reasoning about the given evidence for judging the truthfulness of a claim.
        Here are the evidence for verifying the claim:
        {img_guides}
        Please think and determine the final truthfulness of the claim based on the given evidence. Truthfulness must be one of these three values only: refuted, supported, or not enough information.
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
            evidences += f"""
            Text evidence: {t}

            """
            eidx += 1
        prompt = f"""
        You are an assistant that help judging the truthfulness of a claim.
        The claim is: {claim}
        You should reasoning about the given evidence for judging the truthfulness of a claim.
        Here are the evidence for verifying the claim:
        {evidences}
        Please think and determine the final truthfulness of the claim based on the given evidence. Truthfulness must be one of these three values only: refuted, supported, or not enough information.
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


def create_verification_prompt(dataset, model, processor, path, new_token=10):
    results = []
    new_dataset = read_augmented_data(dataset)
    print("---performing verification .....----")
    for sample in tqdm(new_dataset):
        prompt = make_verification_prompt(sample['claim'], sample['text_evidence'], sample['alignment'], path)
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


def make_verification_prompt_vision(claim, text_evidence, image_guides, path):
    def parse_image_path(img_evidence):
        return path + "/images/" + img_evidence.split("/")[-1]

    images_list = []
    if len(image_guides) > 0:
        content = [
            {"type": "text", "text": "You are an assistant to perform checking the truthfulness of a claim."},
            {"type": "text", "text": "The claim is: {}.".format(claim)},
            {"type": "text", "text": "Please consulting these evidences for verifying the claim: "},
        ]
        for im in image_guides:
            content.append({"type": "text", "text": im['text']})
            content.append({"type": "image"})
            content.append({"type": "text", "text": im['alignment']})
            
            images_list.append(parse_image_path(im['image']))
    else:
        content = [
            {"type": "text", "text": "You are an assistant to perform checking the truthfulness of a claim."},
            {"type": "text", "text": "The claim is: {}.".format(claim)},
            {"type": "text", "text": "Please consulting these evidences for verifying the claim: "},
        ]
        for t in text_evidence:
            content.append({"type": "text", "text": t})
    
    content.append(
        {"type": "text", "text": "Based on those given clues, please determining the truthfulness of the claim. The results must be one of these three values: refuted, supported or not enoght information. \n<RESPONSE>:"},
    )
    prompt = [
        {
            "role": "user",
            "content": content
        }
    ]
    return prompt, images_list


def create_verification_prompt_vision_text_only(dataset, model, processor, path, new_token=10):
    results = []
    new_dataset = read_augmented_data(dataset)
    print("---performing verification .....----")
    for sample in tqdm(new_dataset):
        prompt = make_verification_prompt(sample['claim'], sample['text_evidence'], sample['alignment'], path)
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

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="/home/s2320014/data")
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--system', default=False, action='store_true')
    parser.add_argument('--demo', default=False, action='store_true')
    parser.add_argument('--limit', default=False, action='store_true')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=0)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parser_args()
    # processor, model = load_peft_model_vision("meta-llama/Llama-3.2-90B-Vision-Instruct")

    # print("Running on gold evidence")
    # train, val, test = get_dataset(args.path)
    # dev_claim = ClaimVerificationDataset(val)
    # test_claim = ClaimVerificationDataset(test)

    # print(len(dev_claim))
    # print(len(test_claim))

    # if args.limit:
    #     print("Run with limit: {}-{}".format(args.start, args.end))
    #     test_claim = test_claim[args.start:args.end]
    #     dev_claim = dev_claim[args.start:args.end]

    # if not args.test:
    #     print("Dev")
    #     result_dev = create_align_form(dev_claim, model, processor, args.path)
    #     with open('./factify_claim_llama3.2_dev.json' if not args.limit else './factify_claim_llama3.2_dev_{}-{}.json'.format(args.start, args.end), 'w', encoding='utf-8') as f:
    #         json.dump(result_dev, f, ensure_ascii=False, indent=4)
    #     f.close()
    # else:
    #     print("Test")
    #     result_test = create_align_form(test_claim, model, processor, args.path)
    #     with open('./factify_claim_llama3.2_test.json' if not args.limit else './factify_claim_llama3.2_test_{}-{}.json'.format(args.start, args.end), 'w', encoding='utf-8') as f:
    #         json.dump(result_test, f, ensure_ascii=False, indent=4)
    #     f.close()
        
    # processor, model = load_peft_model_text("meta-llama/Llama-3.1-8B-Instruct")
    processor, model = load_peft_model_vision2("llava-hf/llava-1.5-7b-hf")
    with open("./factify_claim_llama3.2_test_0-3500.json", "r") as f:
        dataset1 = json.load(f)
    f.close()

    with open("./factify_claim_llama3.2_test_3500-7500.json", "r") as f:
        dataset2 = json.load(f)
    f.close()

    dataset = dataset1 + dataset2

    results = create_verification_prompt_vision_text_only(dataset, model, processor, args.path, new_token=10)

    g, p, new_results = retrieve_verification_results(results)

    with open('./factify_verification_test_llava-7B.json', 'w', encoding='utf-8') as f:
        json.dump(new_results, f, ensure_ascii=False, indent=4)
    f.close()

    print("Test result micro: {}\n".format(f1_score(g, p, average='micro')))
    print("Test result macro: {}\n".format(f1_score(g, p, average='macro')))
    print("Test result Accuracy: {}\n".format(accuracy_score(g, p)))
    print(confusion_matrix(g, p, labels=[0, 1, 2]))