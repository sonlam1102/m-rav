import pandas as pd
import cv2
import glob
from tqdm import tqdm, trange

DATA_PATH = '/home/s2320014/data'


def read_text_corpus(path):
    train = path + '/train/Corpus2.csv'
    dev = path + '/val/Corpus2.csv'
    test = path + '/test/Corpus2.csv'

    train_data = pd.read_csv(train, low_memory=False)
    val_data = pd.read_csv(dev, low_memory=False)
    test_data = pd.read_csv(test, low_memory=False)

    return (train_data, val_data, test_data)


def read_image(path):
    # imdir = 'path/to/files/'
    ext = ['jpg', 'jpeg', 'png']

    files = []
    images = []
    names = []
    claim = []
    for e in ext:
        for img in glob.glob(path + "/*." + e):
            files.append(img)

    for f in files:
        names.append(f.split('/')[-1])
        claim.append(int(f.split('/')[-1].split('-')[0]))
        images.append(cv2.imread(f))
    
    return pd.DataFrame({
        'claim_id': claim,
        'id': names,
        'image': images
    })


def read_image_path_only(path):
    ext = ['jpg', 'jpeg', 'png']

    files = []
    images = []
    names = []
    claim = []
    for e in ext:
        for img in glob.glob(path + "/*." + e):
            files.append(img)

    for f in files:
        names.append(f.split('/')[-1])
        claim.append(int(f.split('/')[-1].split('-')[0]))
        images.append(f)

    # images = [cv2.imread(file) for file in files]
    #
    return pd.DataFrame({
        'claim_id': claim,
        'id': names,
        'image': images
    })


def read_images_corpus(path):
    train = path + '/train/images'
    dev = path + '/val/images'
    test = path + '/test/images'

    train_images = read_image_path_only(train)
    dev_images = read_image_path_only(dev)
    test_images = read_image_path_only(test)

    return (train_images, dev_images, test_images)


def retrieve_data_for_verification(train_text, train_images):
    claim_ids = train_text['claim_id'].values
    claim_ids = list(set(claim_ids))

    claim_data = []
    for claim_id in tqdm(claim_ids):
        df = train_text.loc[(train_text.claim_id == claim_id)]
        text_evidences = df['Evidence'].values
        image_evidences = train_images.loc[(train_images.claim_id == claim_id)]['image'].values

        claim_object = (df['Claim'].values[0], text_evidences, image_evidences, df['cleaned_truthfulness'].values[0], claim_id)
        claim_data.append(claim_object)

    return claim_data


def get_dataset(path):
    train_text, dev_text, test_text = read_text_corpus(path)
    train_image, dev_image, test_image = read_images_corpus(path)

    val_claim = retrieve_data_for_verification(dev_text, dev_image)
    train_claim = retrieve_data_for_verification(train_text, train_image)
    test_claim = retrieve_data_for_verification(test_text, test_image)

    return train_claim, val_claim, test_claim


if __name__ == '__main__':
    train_text, dev_text, test_text = read_text_corpus(DATA_PATH)
    train_image, dev_image, test_image = read_images_corpus(DATA_PATH)

    val_claim = retrieve_data_for_verification(dev_text, dev_image)
    train_claim = retrieve_data_for_verification(train_text, train_image)
    test_claim = retrieve_data_for_verification(test_text, test_image)
