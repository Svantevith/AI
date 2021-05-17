import os
import re
from datetime import datetime
from typing import NoReturn, Union

import numpy as np
import pandas as pd
from PIL import UnidentifiedImageError

from neural_networks.imagenet import IHappyPlant
from scraping.export_excel.export_excel import save_excel
from scraping.image_scraping.image_scraper import get_data_from_image
from scraping.web_scraping.web_scraper import get_data_from_web

"""
Plant Diseases, Garden Pests, and Possible Treatments [web]
@ https://www.fondation-louisbonduelle.org/

14 Common Plant Diseases and How to Treat Them [images]
@ https://www.proflowers.com/
"""


def get_urls(file: str) -> list:
    with open(file, 'r') as f:
        return [line.strip('\n') for line in f.readlines()]


DATABASE_DIR = r'D:\PyCharm Professional\Projects\HappyPlant\data\database'
DATABASE_PATH = os.path.join(DATABASE_DIR, 'HappyPlantDatabase.xlsx')
URL_PATH = os.path.join(DATABASE_DIR, r'disease_urls.txt')
FONDATION_URL, PRO_FLOWERS_URL = get_urls(URL_PATH)[:2]


def find_words(line: str) -> list:
    words = re.findall('[A-Za-z\\-]+', line)
    return [line, *words] if len(words) > 1 else words


def format_text(text: Union[str, list], indent: int = 1, enum: bool = False) -> str:
    if isinstance(text, str):
        text = text.split('\n')
    tab = '\t' * indent
    if enum:
        return ''.join([f'\n{tab} [{i}] {word}' for i, word in enumerate(text, start=1)])
    return ''.join([f'\n{tab} - {word}' for word in text])


def extract_keywords(text: str) -> Union[list, np.ndarray]:
    if re.search('\\w+', text):
        keywords = re.split('\\s?,\\s?', text)
        if len(keywords) == 1:
            return find_words(keywords[0])
        return np.ravel([find_words(keyword) for keyword in keywords])
    return []


def get_output_path():
    output_folder = r'D:\PyCharm Professional\Projects\HappyPlant\data\export'
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    file_name = f"happy_plant_disease_queries_{datetime.strftime(datetime.now(), '%Y%m%d%H%M')}.xlsx"
    return os.path.join(output_folder, file_name)


def show_description(df: pd.DataFrame, matches: pd.Series, keywords: list) -> NoReturn:
    if not matches.empty:
        diseases = matches[matches == matches.max()].index
        found = diseases.shape[0]
        print(f"\n[🌷] {found} suitable match{'es' if found > 1 else ''} found based on provided keywords:")
        print(f'{format_text(keywords, indent=1)}')
        for i, disease in enumerate(diseases, start=1):
            row = df.loc[df['Disease'] == disease].reset_index(inplace=False)
            print(
                f'\n\t[{i} 💮] '
                + f'Potential disease, accompanying symptoms, possible reasons and appropriate treatment hints:\n'
                + f'\n\t\t[🦠] The plant disease indicated by the provided keywords is most likely:\n'
                + f'\t\t{format_text(disease, indent=2)}.\n'
                + '\n\t\t[🌡️] The symptoms and damage to the plant may be as following:\n'
                + f"\t\t{format_text(row.loc[0, 'Symptoms'], indent=2)}\n"
                + '\n\t\t[🔬] Possible cause of appearance of these diseases or pests is favorably:\n'
                + f"\t\t{format_text(row.loc[0, 'Reason'], indent=2)}\n"
                + '\n\t\t[🌸] The appropriate treatment involves:\n'
                + f"\t\t{format_text(row.loc[0, 'Treatment'], indent=2)}"
            )
        if re.match('y', input("\n[🦕] Would you like to save your session? y/[n]: "), re.IGNORECASE):
            queries = df[df['Disease'].isin(diseases)]
            output_path = get_output_path()
            save_excel(queries, output_path)
    elif keywords:
        print(
            "[💧] Unfortunately we haven't found any suitable matches for provided keywords:"
            + format_text(keywords, indent=1)
        )


def get_matches(df: pd.DataFrame, keywords: list) -> pd.Series:
    matches = {}
    for keyword in keywords:
        for idx in df.index:
            disease = df.loc[idx, 'Disease']
            if re.match(f'{keyword}$', disease, re.IGNORECASE):
                return pd.Series({disease: 1}, dtype=int)
            for col in df.columns:
                found = len(re.findall(keyword, df.loc[idx, col], re.IGNORECASE))
                if found > 0:
                    matches[disease] = matches.get(disease, 1) + found
    return pd.Series(matches, dtype=int)


def concat_cells(x: str) -> str:
    lines = re.split('\\.\\s', ' '.join(x))
    return '.\n'.join(lines) if len(lines) > 1 else lines[0]


def get_database(load_from_database: bool = True, save: bool = False) -> pd.DataFrame:
    if load_from_database:
        return pd.read_excel(DATABASE_PATH)
    df_from_web = get_data_from_web(FONDATION_URL)
    df_from_image = get_data_from_image(PRO_FLOWERS_URL)
    database = pd.concat([df_from_image, df_from_web]).groupby('Disease').agg(lambda x: concat_cells(x)).reset_index()
    if save:
        database.to_excel(DATABASE_PATH, index=False, engine='xlsxwriter')
    return database


def plant_disease_recognition() -> NoReturn:
    query = ''
    features = ['Search Engine', 'Image Net']
    output_message = "[🥑] Thank you for trusting us - HappyPlant is keeping track on your plant's health conditions 💚"
    print(f"\n[🧠] iHappyPlant - intelligent plant disease recognition & treatment:")
    print(format_text(features, indent=1, enum=True))
    try:
        choice = int(input('\n[💦] Choose feature: '))
        print(f'[🚀] Launching {features[choice - 1]}...\n')
        if choice == 1:
            query = input('[🌱] Please provide comma-separated keywords describing the visual defects of your plant: ')
        elif choice == 2:
            i_happy_plant = IHappyPlant()
            user_input = input('[🌱] Please provide path to the image presenting pathological leaf: ')
            y_pred = i_happy_plant.predict_disease(user_input)
            if re.match('Healthy', y_pred, re.IGNORECASE):
                print(f"\n\t[🍒] It seems like your plant is fit as a fiddle!")
                return
            elif re.match('Background', y_pred, re.IGNORECASE):
                print(f"\n\t[🐻] Unfortunately, iHappyPlant could not identify any plant on the uploaded image.")
                return
            else:
                query = y_pred.replace('_', ' ')
    except (ValueError, IndexError, UnidentifiedImageError, OSError) as err:
        print(
            f'[❌] Wrong input provided: {err}\n'
            f'[🍁] We hope your plant is safe and sound!'
        )
    else:
        keywords = extract_keywords(query)
        if keywords:
            data = get_database(load_from_database=True)
            matches = get_matches(data, keywords)
            show_description(data, matches, keywords)
        else:
            print("[🌺] You haven't provided any valid keywords. We hope your plant is right as rain!")
    finally:
        print(f"\n\n{output_message}")


if __name__ == '__main__':
    plant_disease_recognition()
